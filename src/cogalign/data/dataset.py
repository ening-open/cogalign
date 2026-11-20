from dataclasses import dataclass
import os
import os.path as osp
import random
from itertools import islice, zip_longest

from PIL import Image
import torch
from ..utils.io import read_json, save_json

def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 2) → AB CD EF G
    if n < 1:
        raise ValueError('n must be at least one')
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError('batched(): incomplete batch')
        yield batch

@dataclass
class DatasetConfig:
    image_dir: str = ""
    train: str = ""
    eval: str = ""
    test: str = ""
    log_dir: str = ""

class CogAlignDataset(torch.utils.data.Dataset):
    def __init__(self, config: DatasetConfig, split: str = "train", **kwargs):
        self.rng = random.Random(42)
        self.split = split
        self.config = config
        self.batch_size = kwargs.get("batch_size", 8*1*1)  # default 8 GPUs x 1 (per gpu batch size) x 1 (gradient accumulation)
        self.image_dir = self.config.image_dir
        self.rank = kwargs.get("rank", 0)

        # rank 0
        if self.rank == 0:
            datas = self.load_data()
            # reorganize data_loaded
            self.datas = self.reorganize_data(datas)
            # cache to log_dir to read for rank: 1...n
            self.cache_data(config.log_dir)
        else:
            self.datas = self.load_cached_data(config.log_dir)

    def cache_data(self, cache_dir: str = os.path.expanduser("~")):
        os.makedirs(cache_dir, exist_ok=True)
        cache_data_path = os.path.join(cache_dir, f"{self.split}_data.json")
        # cache data
        save_json(self.datas, cache_data_path)
        print(f"===> Rank-{self.rank}: Cache {self.split}_data.json to {cache_dir}")

        self.cache_dir, self.cache_data_path = cache_dir, cache_data_path

    def load_cached_data(self, cache_dir):
        cache_data_path = os.path.join(cache_dir, f"{self.split}_data.json")
        cached_datas = read_json(cache_data_path)
        print(f"===> Rank-{self.rank}: Load {self.split}_data.json from `cached data` ({cache_data_path}) ")
        return cached_datas

    def load_data(self):
        data_path = getattr(self.config, self.split)
        datas = read_json(data_path)
        print(f"===> Rank-{self.rank}: Load {self.split}_data.json from {data_path}")
        return datas

    def save_data(self):
        # save data to ensure experiment
        output_dir = self.config.log_dir
        os.makedirs(output_dir, exist_ok=True)
        save_json(self.datas, osp.join(output_dir, f"{self.split}_data.json"))
        print(f"====== finished saving {self.split} data to {output_dir}!!! ======")

    def allocate_batch(self, batch_size, ratios):
        scaled = [r * batch_size for r in ratios]
        floored = [int(x) for x in scaled]
        remainder = batch_size - sum(floored)
        indices = sorted(range(len(scaled)), key=lambda i: -(scaled[i] - floored[i]))
        for i in indices[:remainder]:
            floored[i] += 1
        return [max(1, x) for x in floored]

    def reorganize_data(self, data_dict: dict):
        print(f"===> Rank-{self.rank}: Reorganize data!!! (Batch_size {self.batch_size})")
        groups = {
            "single-image": {
                "task": {"allava_instruct_vflan4v", "allava_instruct_laion4v", "llavar_gpt4_20k", "coco"},
                "data": [], "num": 0,
            },
            "multi-image": {
                "task": {"aesop", "clevr_change", "docvqa", "dreamsim", "iedit", "magicbrush", "spot_the_diff", "tqa", "vist", "webqa", "ocr_vqa"},
                "data": [], "num": 0,
            },
            "behavior": {
                "task": {"o1o"}, "data": [], "num": 0,
            }
        }

        tasks = sorted(set().union(*(group["task"] for group in groups.values())))
        tasks_spt = list(data_dict.keys())
        for task in tasks:
            if task not in tasks_spt:
                print(f"`{task}` not in your provided `{tasks_spt}`")
                continue
            task_num = data_dict[task]["total"]
            task_datas = [{**item, "source": task} for item in data_dict[task]["data"]]  # add source which is directory basename /{task}/images
            for group_name, group in groups.items():
                if task in group["task"]:
                    group["num"] += task_num
                    group["data"].extend(task_datas)
                    break
        groups_data, groups_num = {}, {}
        for group_name, group in groups.items():
            groups_data[group_name] = group["data"].copy()
            self.rng.shuffle(groups_data[group_name])  # shuffle
            groups_num[group_name] = group["num"]

        total_num = sum(groups_num.values())
        si_ratio = groups_num["single-image"] / total_num
        mi_ratio = groups_num["multi-image"] / total_num
        bhv_ratio = groups_num["behavior"] / total_num

        # reorganize data by rations
        si_bs, mi_bs, bhv_bs = self.allocate_batch(self.batch_size, [si_ratio, mi_ratio, bhv_ratio])
        # single/multi/behavioral_datas batched
        si_datas, mi_datas, bhv_datas = batched(groups_data["single-image"], si_bs), batched(groups_data["multi-image"], mi_bs), batched(groups_data["behavior"], bhv_bs)

        datas = []
        for si_data, mi_data, bhv_data in zip_longest(si_datas, mi_datas, bhv_datas, fillvalue=[None]):
            mix_data = [item for item in si_data + mi_data + bhv_data if item is not None]
            self.rng.shuffle(mix_data)
            datas.extend(mix_data)
        return datas

    def __len__(self):
        return len(self.datas)

    def convert_img(self, image):
        if image.mode == "RGB":
            return image
        image = image.convert("RGB")
        return image

    def __getitem__(self, idx):
        data = self.datas[idx]
        source = data["source"]
        images = [self.convert_img(Image.open(osp.join(self.image_dir, imp))) for imp in data["image"]]
        conv_pairs = data["conversations"]
        if conv_pairs[0]["role"] == "system":
            system_prompt = conv_pairs[0]["text"]
            conv_pairs = conv_pairs[1:]
        else:
            system_prompt = None
        user_input = conv_pairs[0]["text"]
        assist = conv_pairs[1]["text"]
        return {
            "image": images,
            "system": system_prompt,
            "input": user_input,
            "output": assist,
        }
