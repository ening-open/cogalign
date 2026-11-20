import argparse
import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import h5py
import torchmetrics

def read_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def read_h5(path, dataset: str = "data"):
    with h5py.File(path, 'r') as f:
        return f[dataset][:]

def load_volume(volume_path, volume_trial1_path, volume_trial2_path, volume_trial3_path):
    volumes = read_h5(volume_path)
    volumes_trial1 = read_h5(volume_trial1_path)
    volumes_trial2 = read_h5(volume_trial2_path)
    volumes_trial3 = read_h5(volume_trial3_path)
    return volumes, (volumes_trial1, volumes_trial2, volumes_trial3)

def load_hidden_state(path) -> dict[str, np.ndarray]:
    all_hidden_states = read_pkl(path)
    image_pad_hidden_states, vision_end_hidden_states, im_end_hidden_states = [], [], []

    for i, hidden_states in tqdm(enumerate(all_hidden_states), total=len(all_hidden_states), desc="Loading hidden states (image_pad, vision_end, im_end[user])"):
        assert f"image_{i}.png" == hidden_states["info"][0] and len(hidden_states["info"]) == 1

        image_pad_hs = hidden_states["image_pad"]
        vision_end_hs = hidden_states["vision_end"]
        im_end_hs = hidden_states["im_end"]["user"]

        image_pad_hidden_states.append(image_pad_hs)
        vision_end_hidden_states.append(vision_end_hs)
        im_end_hidden_states.append(im_end_hs)

    image_pad_hidden_states = np.array(image_pad_hidden_states)
    vision_end_hidden_states = np.array(vision_end_hidden_states)
    im_end_hidden_states = np.array(im_end_hidden_states)

    # image_pad_hidden_states (982, num_tokens, hidden_size) average
    image_pad_hidden_states = np.mean(image_pad_hidden_states, axis=1, keepdims=False)

    # vision_end_hidden_states (982, 1, hidden_size)
    vision_end_hidden_states = vision_end_hidden_states.reshape((-1, vision_end_hidden_states.shape[-1]))

    # im_end_hidden_states (982, hidden_size)
    return {
        "image_pad": image_pad_hidden_states,
        "vision_end": vision_end_hidden_states,
        "im_end": im_end_hidden_states
    }

def compute_rdm_torch(X):
    corr = torch.corrcoef(X)
    rdm = 1 - corr
    rdm = torch.nan_to_num(rdm, nan=1.0)  # correlation 0 -> distance 1
    return rdm

def compute_hs_rdm(hidden_states):
    hs_rdms = {}
    for hs_name, hs in hidden_states.items():
        hs = torch.from_numpy(hs).cuda()
        hs_rdms[hs_name] = compute_rdm_torch(hs)
    return hs_rdms

def compute_rsa_torch(rdm1, rdm2) -> torch.Tensor:
    assert rdm1.shape == rdm2.shape
    n = rdm1.shape[-1]
    mask = torch.triu(torch.ones(n, n, device=rdm1.device, dtype=torch.bool), diagonal=1)
    vec1 = rdm1[mask]
    vec2 = rdm2[mask]
    res = torchmetrics.functional.spearman_corrcoef(vec1, vec2)
    return res.cpu()

def np2torch(ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.from_numpy(ndarray).cuda().to(dtype=dtype)

def rsa_searchlight(args):
    subject, tag = args.subject, args.tag

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    hidden_state_path = args.hidden_state
    # load hidden_states
    hidden_states = load_hidden_state(hidden_state_path)

    volume_path = args.volume
    volume_trial1_path, volume_trial2_path, volume_trial3_path = args.volume_trial1, args.volume_trial2, args.volume_trial3

    # load volume in xyz format
    print(f"===> Loading volumes, volumes trial1, volumes trial2, volumes trial3")
    volumes, (volumes_trial1, volumes_trial2, volumes_trial3) = load_volume(volume_path, volume_trial1_path, volume_trial2_path, volume_trial3_path)  # (n_sample, x, y, z)

    # compute hidden_state rdm
    print("===> Computing hidden states RDM")
    hidden_state_rdms = compute_hs_rdm(hidden_states)

    # searchlight
    # volume to torch
    vols= np2torch(volumes)
    vols_trial1, vols_trial2, vols_trial3 = np2torch(volumes_trial1), np2torch(volumes_trial2), np2torch(volumes_trial3)

    xyz_shape = vols.shape[1:]  # (x, y, z)

    print(f"===> Preparing Coordinate")
    brain_mask = vols[0] != 0
    center_coords = torch.nonzero(brain_mask, as_tuple=False)
    n_centers = center_coords.shape[0]

    print(f"===> Preparing Meshgrid `nxn`")
    radius = 3
    rs = torch.arange(-radius, radius + 1).cuda()
    grid_x, grid_y, grid_z = torch.meshgrid(rs, rs, rs, indexing="ij")
    offsets = torch.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], dim=1)

    # prepare output_dict
    print(f"===> Preparing scores, scores_norm output map [shape: {xyz_shape}]")
    outputs_dict = {
        "score": {k: np.zeros(xyz_shape, dtype=np.float32) for k in hidden_state_rdms.keys()},
        "score_norm": {k: np.zeros(xyz_shape, dtype=np.float32) for k in hidden_state_rdms.keys()},
        "noise_ceiling": np.zeros(xyz_shape, dtype=np.float32)
    }

    print(f"===> Running Searchlight RSA (n_centers={n_centers}, radius={radius})")
    for n_i in tqdm(range(n_centers), desc="Computing RSA"):
        cx, cy, cz = center_coords[n_i]  # center voxel
        neighbor_coords = center_coords[n_i] + offsets

        # valida neighbors mask
        valid_mask = (
                (neighbor_coords[:, 0] >= 0) & (neighbor_coords[:, 0] < xyz_shape[0]) &
                (neighbor_coords[:, 1] >= 0) & (neighbor_coords[:, 1] < xyz_shape[1]) &
                (neighbor_coords[:, 2] >= 0) & (neighbor_coords[:, 2] < xyz_shape[2])
        )
        valid_neighbors = neighbor_coords[valid_mask]
        nx, ny, nz = valid_neighbors[:, 0], valid_neighbors[:, 1], valid_neighbors[:, 2]

        voxels_sphere = vols[:, nx, ny, nz]
        voxels_trial1_sphere = vols_trial1[:, nx, ny, nz]
        voxels_trial2_sphere = vols_trial2[:, nx, ny, nz]
        voxels_trial3_sphere = vols_trial3[:, nx, ny, nz]

        rdm_sphere = compute_rdm_torch(voxels_sphere)
        rdm_trial1_sphere = compute_rdm_torch(voxels_trial1_sphere)
        rdm_trial2_sphere = compute_rdm_torch(voxels_trial2_sphere)
        rdm_trial3_sphere = compute_rdm_torch(voxels_trial3_sphere)

        rdm_mean_sphere = (rdm_trial1_sphere + rdm_trial2_sphere + rdm_trial3_sphere) / 3
        r1 = compute_rsa_torch(rdm_mean_sphere, rdm_trial1_sphere)
        r2 = compute_rsa_torch(rdm_mean_sphere, rdm_trial2_sphere)
        r3 = compute_rsa_torch(rdm_mean_sphere, rdm_trial3_sphere)

        noise_ceiling = max([r1, r2, r3])

        # compute Searchlight RSA for all hidden states
        for hs_rdm_name, hs_rdm in hidden_state_rdms.items():
            r = compute_rsa_torch(rdm_sphere, hs_rdm)
            r_norm = r / (noise_ceiling + 1e-8)

            # fill to output_dict
            outputs_dict["score"][hs_rdm_name][cx, cy, cz] = r.item()
            outputs_dict["score_norm"][hs_rdm_name][cx, cy, cz] = r_norm.item()

        outputs_dict["noise_ceiling"][cx, cy, cz] = noise_ceiling.item()

    # save metrics
    print(f"Saving score, score_norm, noise_ceiling to: {output_dir}")
    for metrics_name, metrics_datas in outputs_dict.items():
        if metrics_name == "noise_ceiling":
            output_filename = f"{subject}_{tag}_{metrics_name}.npy"
            np.save(os.path.join(output_dir, output_filename), metrics_datas)
            continue

        for hs_rdm_name in hidden_state_rdms.keys():
            metrics_data = metrics_datas[hs_rdm_name]
            output_filename = f"{subject}_{tag}_{hs_rdm_name}_{metrics_name}.npy"
            np.save(os.path.join(output_dir, output_filename), metrics_data)

def main():
    parser = argparse.ArgumentParser(description="Representational Similarity Analysis (RSA)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # searchlight parser
    searchlight_parser = subparsers.add_parser("searchlight", help="Searchlight RSA")
    ## common args
    searchlight_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    searchlight_parser.add_argument("--hidden-state", type=str, required=True, help="Hidden state absolute path (*.pkl)")
    searchlight_parser.add_argument("--subject", type=str, required=True, help="Subject ID")
    searchlight_parser.add_argument("--volume", type=str, required=True, help="Volume absolute path (fmri_shared_1k_all_rois_4d.h5)")
    searchlight_parser.add_argument("--volume-trial1", type=str, required=True, help="Volume trial1 absolute path (fmri_shared_1k_all_rois_4d_trial_1.h5)")
    searchlight_parser.add_argument("--volume-trial2", type=str, required=True, help="Volume trial2 absolute path (fmri_shared_1k_all_rois_4d_trial_2.h5)")
    searchlight_parser.add_argument("--volume-trial3", type=str, required=True, help="Volume trial3 absolute path (fmri_shared_1k_all_rois_4d_trial_3.h5)")
    searchlight_parser.add_argument("--tag", type=str, default="tag", help="Output file tag")

    # searchlight parser
    args = parser.parse_args()

    if args.command == "searchlight":
        rsa_searchlight(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
