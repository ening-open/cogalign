import argparse
import os
import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr

from cogalign.utils.io import read_json, save_mat

def get_object_simi_matrix(datas, num_object):
    simi_matrix = np.zeros((num_object, num_object), dtype=int)
    total_matrix = np.zeros((num_object, num_object), dtype=int)

    for i, data in enumerate(datas):
        object_ids = data
        C = object_ids[-1]
        if C == -1:
            # parse error
            continue
        AB = [object_ids[0], object_ids[1]]

        # fill simi matrix
        simi_matrix[AB[0], AB[1]] += 1
        simi_matrix[AB[1], AB[0]] += 1

        # fill total matrix
        total_matrix[object_ids[0], object_ids[1]] += 1
        total_matrix[object_ids[0], object_ids[2]] += 1
        total_matrix[object_ids[1], object_ids[2]] += 1
        total_matrix[object_ids[1], object_ids[0]] += 1
        total_matrix[object_ids[2], object_ids[0]] += 1
        total_matrix[object_ids[2], object_ids[1]] += 1

    result = np.zeros((num_object, num_object), dtype=float)
    for i in range(num_object):
        for j in range(i + 1, num_object):
            if total_matrix[i, j] > 0:
                score = simi_matrix[i,j] / total_matrix[i,j]
                result[i,j] = score
                result[j,i] = score
    np.fill_diagonal(result, 1)
    return result, simi_matrix, total_matrix

def get_simi(matrix1, matrix2, diag=True, mode="pearson"):
    vec1, vec2 = None, None
    if diag:
        vec1 = matrix1[np.triu_indices(matrix1.shape[0], k=1)]
        vec2 = matrix2[np.triu_indices(matrix2.shape[0], k=1)]
    else:
        raise NotImplementedError
    corr = 0.0
    if mode == "pearson":
        corr, p_value = pearsonr(vec1, vec2)
    elif mode == "euclidean":
        corr = euclidean(vec1, vec2)
    elif mode == "cosine":
        corr = 1 - cosine(vec1, vec2)
    return corr

def compute_rsm(data_path, output_dir, output_fn):
    num_object = 1854
    all_datas = read_json(data_path)

    rsm, _, _ = get_object_simi_matrix(all_datas, num_object=num_object)
    n_mid = len(all_datas) // 2

    rsm_half1, _, _ = get_object_simi_matrix(all_datas[:n_mid], num_object=num_object)
    rsm_half2, _, _ = get_object_simi_matrix(all_datas[n_mid:], num_object=num_object)

    os.makedirs(output_dir, exist_ok=True)
    rsm_output_path = os.path.join(output_dir, f"RSM_{output_fn}_48object.mat")
    save_mat({"RSM1854_triplet": rsm}, rsm_output_path, mode="mat")


    rsm_half_output_path = os.path.join(output_dir, f"RSM_{output_fn}_48object_half.mat")
    save_mat({
        "RSM1854_triplet_split1": rsm_half1,
        "RSM1854_triplet_split2": rsm_half2
    }, rsm_half_output_path, mode="mat")

def main():
    parser = argparse.ArgumentParser(description="Compute model choice-based RSMs.")
    parser.add_argument("--data-path", required=True, help="Input JSON with encoded triplets.")
    parser.add_argument("--output-dir", required=True, help="Directory for the RSM files.")
    parser.add_argument("--model-tag", required=True, help="Model tag to use in output names.")
    args = parser.parse_args()
    compute_rsm(args.data_path, args.output_dir, args.model_tag)


if __name__ == "__main__":
    main()
