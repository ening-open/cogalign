import inspect

from .dataset import CogAlignDataset, DatasetConfig
from .qwen3_collator import Qwen3VLCollator

DATASET = {
    "base": [DatasetConfig, CogAlignDataset],
}

def make_supervised_data(name, **kwargs):
    assert name in DATASET.keys(), f"Dataset: {name} is not supported."

    CONFIG_CLASS, DATASET_CLASS = DATASET[name]
    config_sig = inspect.signature(CONFIG_CLASS.__init__)
    config_kwargs_k = set(config_sig.parameters.keys()) - {"self"}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_kwargs_k}
    config = CONFIG_CLASS(**config_kwargs)
    train_set = DATASET_CLASS(config, split="train", **kwargs)
    eval_set = DATASET_CLASS(config, split="eval", **kwargs)
    return config, (train_set, eval_set)

__all__ = ["make_supervised_data", "Qwen3VLCollator"]
