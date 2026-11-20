import glob
import json
import os
import pickle
import zipfile
from pathlib import Path
import numpy as np
from typing import Union
import scipy
from tqdm import tqdm

def unzip_file(zip_file, extract_path):
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        zipf.extractall(extract_path)

def unzip_files(zip_files, extract_path):
    for zip_file in tqdm(zip_files):
        unzip_file(zip_file, extract_path)

def unzip_folder(floader, extract_path):
    zip_files = glob.glob(os.path.join(floader, "*.zip"))
    unzip_files(zip_files, extract_path)

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_json(data: Union[list, dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def read_txt(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def get_image_files(folder_path, recursive=False):
    extension = [".jpg", ".jpeg", ".png"]

    path = Path(folder_path)
    glob = path.rglob if recursive else path.glob

    image_files = [
        os.path.relpath(str(p), folder_path) for p in glob("*")
        if p.is_file() and p.suffix in extension
    ]
    return image_files

def save_mat(data, output_path, mode="np"):
    if mode == "np":
        np.save(output_path, data)
    elif mode == "mat":
        scipy.io.savemat(output_path, data)

def read_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def save_pkl(data, output_path):
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

def read_np(path):
    with open(path, "rb") as f:
        data = np.load(f)
    return data
