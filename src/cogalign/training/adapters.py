import os

import torch
from transformers import AutoProcessor, MODEL_MAPPING
from transformers import Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

def merge_adapter_hf(base_model, adapter_name):
    peft_model = PeftModel.from_pretrained(base_model, adapter_name)
    peft_model = peft_model.merge_and_unload()
    return peft_model

def merge_adapter_qwen25vl(model_name, adapter_name, output_dir):
    processor = AutoProcessor.from_pretrained(model_name)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cpu")
    model_merged = merge_adapter_hf(base_model, adapter_name)
    # save
    output_dir = os.path.join(output_dir, "adapter_merged")
    os.makedirs(output_dir, exist_ok=True)
    processor.save_pretrained(output_dir)
    model_merged.save_pretrained(output_dir)

