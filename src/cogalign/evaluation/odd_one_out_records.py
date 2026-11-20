import copy
import string
import os
import json
from tqdm import tqdm
import glob
from ..utils.io import read_json, save_json

def remove_punctuation(text):
    trans = str.maketrans('', '', string.punctuation + "\n")
    text = text.translate(trans)
    return text

def extract_ans(text, options):
    text = text.lower()
    ans_sid = text.rfind("answer")
    ans_parts = remove_punctuation(text[ans_sid:])
    ans = None
    for opt in options:
        opt = opt.lower()
        if opt in ans_parts.split(" "):
            ans = opt
    return ans

def compute_accuracy(datas):
    true_num, false_num, error_num = 0, 0, 0
    error_samples = []

    for data in tqdm(datas, desc="Computing accuracy"):
        idx = data["id"]
        response = data["response"]
        option = data["option"]
        label = data["label_choice"].lower()
        ans = extract_ans(response, option)
        if ans is None:
            error_num += 1
            error_samples.append(data)
        elif ans == label:
            true_num += 1
        else:
            false_num += 1

    accuracy = true_num / (true_num + false_num + error_num)
    accuracy_loose = true_num / (true_num + false_num)
    return (accuracy, accuracy_loose), (true_num, false_num, error_num), error_samples

def merge_o1o_infer(re_pattern, recursive=False):
    paths = glob.glob(re_pattern, recursive=recursive)
    datas = []
    for path in tqdm(paths, desc="Merging o1o infer"):
        datas.extend(read_json(path))
    return datas

def filter_o1o_infer_remainder(raw_datas, fin_datas):
    """ Filter o1o inference results from total data and get the remainder which is not saved """
    assert len(raw_datas) >= len(fin_datas)

    def list2dict(list_datas):
        dict_datas = {ld["id"]: ld for ld in list_datas}
        return dict_datas

    raw_datas = list2dict(raw_datas)
    fin_datas = list2dict(fin_datas)
    raw_keys = set(raw_datas.keys())
    fin_keys = set(fin_datas.keys())
    remainder_keys = list(raw_keys - fin_keys)
    remainder_datas = [raw_datas[rk] for rk in remainder_keys]
    return remainder_datas

def encode_o1o_object(objects, is_image_name: bool = True):
    objects = [os.path.basename(obj) for obj in objects]
    object_ids = copy.deepcopy(objects)
    if is_image_name:
        objects = [os.path.splitext(obj)[0] for obj in objects]
        object_ids = [int(obj.split("_")[-2])-1 for obj in objects]  # start from 0
    return object_ids

def encode_o1o_infer(datas):
    encoded_datas, error_datas = [], []
    for data in tqdm(datas, desc="Encoding o1o infer"):
        idx = data["id"]
        objects = data["image"]
        response = data["response"]
        option = data["option"]
        ans = extract_ans(response, option)

        object_ids = encode_o1o_object(objects, is_image_name=True)
        pred_object_id = encode_o1o_object(objects[option.index(ans)], is_image_name=True) if ans is not None and ans in option else None

        encoded_datas.append({"id": idx, "triplet": object_ids, "choose": pred_object_id})

        if pred_object_id is None:
            data["triplet"] = object_ids
            error_datas.append(data)
    return encoded_datas, error_datas

def count_tokens(datas: list[str], tokenizer):
    tokens = []
    for data in tqdm(datas, desc="Counting tokens:"):
        tokens.append(len(tokenizer.encode(data)))
    total_tokens = sum(tokens)
    avg_tokens = total_tokens / len(datas)
    return avg_tokens, total_tokens, len(datas)
