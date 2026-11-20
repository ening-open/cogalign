import copy
import os
import argparse
import numpy as np
import cortex
import configparser
import shutil

def update_cortex_cfg(cfg_path, filestore=None):
    cfg_dir = os.path.dirname(cfg_path)

    # create bak
    bak_dir = os.path.join(cfg_dir, ".bak")
    shutil.copy2(cfg_path, bak_dir)
    print(f"===> create cfg backup to {bak_dir}")

    config = configparser.ConfigParser()
    config.optionxform = str  # keep raw file format
    config.read(cfg_path)

    # if "basic" not in config:
    #     config.add_section('basic')

    if filestore is not None and os.path.isdir(filestore):
        bak_filestore = config["basic"].get("filestore", "unknow")
        config["basic"]["filestore"] = filestore
        print(f"===> basic.filestore from `{bak_filestore}` to `{filestore}`")

    # write to cfg
    print(f"===> update cfg ({cfg_path})")
    with open(cfg_path, "w") as configfile:
        config.write(configfile)

def set_cortex_filestore(filestore_path):
    print(f"===> Setting cortex database: {filestore_path}")
    cfg_path = cortex.options.usercfg
    update_cortex_cfg(cfg_path, filestore=filestore_path)
    print(f"===> Finished setting cortex filestore: {cortex.db.filestore}")

def visualize_rsa_searchlight(args):
    filestore_path = args["filestore"]
    set_cortex_filestore(filestore_path)

    subject = args["subject"]
    xfm = args["xfm"]  # func1pt8_to_anat0pt8_autoFSbbr

    ds = cortex.Dataset()

    # load human
    human_data_path = args["human_data"]
    raw_human_data = np.load(human_data_path).astype(float)

    human_data = np.transpose(raw_human_data, (2,1,0))
    human_data[human_data == 0] = np.nan
    human_vol = cortex.Volume(human_data, subject, xfm, vmin=0, vmax=np.max(np.nan_to_num(raw_human_data)), cmap="gist_heat_r")
    ds.append(**{f"{subject}_human": human_vol})  # add to dataset

    baseline_data_path = args["baseline_data"]
    compare_data_path = args["compare_data"]

    baseline_name = os.path.splitext(os.path.basename(baseline_data_path))[0]
    compare_name = os.path.splitext(os.path.basename(compare_data_path))[0]

    raw_baseline_data = np.load(baseline_data_path).astype(float)
    raw_compare_data = np.load(compare_data_path).astype(float)

    baseline_data = np.transpose(raw_baseline_data, (2,1,0))
    compare_data = np.transpose(raw_compare_data, (2,1,0))

    raw_diff_data = compare_data - baseline_data  # bak

    diff_data = copy.deepcopy(raw_diff_data)
    diff_data[diff_data == 0] = np.nan
    #diff_data[diff_data <= 0] = 0.0001

    vmin = 0.0
    vmax = 0.15  # 0.15 for 4b
    #diff_vol = cortex.Volume(diff_data, subject, xfm, vmin=vmin, vmax=vmax, cmap="gist_heat_r")
    diff_vol = cortex.Volume(diff_data, subject, xfm, vmin=np.min(np.nan_to_num(diff_data)), vmax=np.max(np.nan_to_num(diff_data)), cmap="RdYlBu_r")  # RdYlBu_r
    ds.append(**{f"{subject}_{compare_name}_gap_{baseline_name}": diff_vol})

    # show
    cortex.webshow(ds, recache=True, port=12345)

    # print(f"subject: {subject}, vim: {np.min(np.nan_to_num(raw_baseline_data))}, vmax: {np.max(np.nan_to_num(raw_baseline_data))}")
    # print(f"subject: {subject}, vim: {np.min(np.nan_to_num(raw_compare_data))}, vmax: {np.max(np.nan_to_num(raw_compare_data))}")
    # print(f"subject: {subject}, vim: {np.min(np.nan_to_num(diff_data))}, vmax: {np.max(np.nan_to_num(diff_data))}")
    input()

def main():
    parser = argparse.ArgumentParser(description="Visualize human and model searchlight RSA maps.")
    parser.add_argument("--filestore", required=True, help="Pycortex filestore directory")
    parser.add_argument("--subject", required=True, help="Pycortex subject ID")
    parser.add_argument("--xfm", default="func1pt8_to_anat0pt8_autoFSbbr", help="Pycortex transform name")
    parser.add_argument("--human-data", required=True, help="Human RSA volume (.npy)")
    parser.add_argument("--baseline-data", required=True, help="Baseline model RSA volume (.npy)")
    parser.add_argument("--compare-data", required=True, help="Comparison model RSA volume (.npy)")
    visualize_rsa_searchlight(vars(parser.parse_args()))

if __name__ == '__main__':
    main()
