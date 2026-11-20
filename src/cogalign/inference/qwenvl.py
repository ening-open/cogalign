import copy
import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from glob import glob

from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
import random
import numpy as np
import logging

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

try:
    from vllm import LLM, SamplingParams
except ImportError:
    vllm = None

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def read_image(image_path):
    with Image.open(image_path) as img:
        return img.copy()

def cache_image(image_dir, image_cache):
    extensions = ("jpg", "png")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(image_dir, "**", f"*.{ext}"), recursive=True))

    images_cache_map = {}

    """
    if image_cache > 0:
        read_img_cache = lru_cache(maxsize=image_cache)(read_image)
        for image_path in tqdm(image_paths, desc="Caching images"):
            images_cache_map[image_path] = read_img_cache(image_path)
    else:
        images_cache_map = None
    """
    images_cache_map = None
    return images_cache_map


def build_prompt(data, image_dir, images_cache_map: dict = None, min_pixels: int = 1280*28*28, max_pixels: int = 5120*28*28):
    image_fns = data["image"]
    conversation = data["conversation"]
    messages = []

    if len(conversation) > 1:
        messages.append({"role": "system", "content": conversation[0]["content"],})

    user = conversation[-1]["content"]
    # build messages
    contents = []
    for img_fn in image_fns:
        img_path = os.path.join(image_dir, img_fn)
        img = read_image(img_path) if images_cache_map is None else images_cache_map[img_path]
        contents.append({
            "type": "image",
            "image": img,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        })
    contents.append({"type": "text", "text": user})
    messages.append({"role": "user", "content": contents})
    return messages


def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return {
        "prompt": text,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs
    }

def run_inference(args):
    # ========= raw data =========
    raw_datas = read_json(args.data)
    print(f"===> Load data from {args.data}")
    # ========= raw data =========

    # ========= cache image =========
    image_dir, image_cache = args.image_dir, args.image_cache
    images_cache_map = cache_image(image_dir, image_cache)
    if images_cache_map is not None:
        print(f"===> Cache images (num={len(images_cache_map.keys())})")
    else:
        print(f"===> Not Cache images")
    # ========= cache image =========

    # ========= set up generation parameters (vLLM) =========
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stop_token_ids=[],
    )

    print(f"===> Generation parameters (vLLM):")
    print(f"max_tokens={sampling_params.max_tokens}")
    print(f"temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, top_k={sampling_params.top_k}")
    print(f"repetition_penalty={sampling_params.repetition_penalty}")
    print(f"presence_penalty={sampling_params.presence_penalty}")

    if sampling_params.presence_penalty > 0:
        print(f"Anti-repetition enabled (presence_penalty={sampling_params.presence_penalty})")

    if sampling_params.temperature <= 0.02 and sampling_params.top_k == 1:
        print(f"Using FAST greedy-like decoding")
    else:
        print(f"Using sampling decoding (slower but more diverse)")
    # ========= set up generation parameters (vLLM) =========

    # ========= load processor =========
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    print("===> Load Processor")
    # ========= load processor =========

    # ========= initialize vLLM =========
    print(f"===> Initialize vLLM with model: {args.model_path}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")

    llm = LLM(
        model=args.model_path,
        #dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": args.max_images_per_prompt},
        #max_num_seqs=args.save_interval,
        seed=42,
    )

    print("===> Initialize vLLM successfully")
    # ========= initialize vLLM =========

    # ========= run batch inference =========
    print("=" * 80)
    print("Running vLLM batch inference (automatic optimization)")
    print("=" * 80)
    raw_datas_num = len(raw_datas)
    model_tag = args.model_tag
    save_interval = args.save_interval
    output_dir = args.output_dir

    for inv_start_id in range(0, raw_datas_num, save_interval):
        inv_end_id = min(inv_start_id+save_interval, raw_datas_num)

        # prepare interval data
        raw_datas_inv = raw_datas[inv_start_id:inv_end_id]

        # ========= prepare inputs =========
        inv_raws, inv_inputs, inv_messages = [], [], []
        for _, raw_data_inv in enumerate(tqdm(raw_datas_inv, total=len(raw_datas_inv), desc="Building prompts (interval)")):
            messages = build_prompt(raw_data_inv, image_dir, images_cache_map=images_cache_map, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            inputs = prepare_inputs_for_vllm(messages, processor)

            inv_raws.append(raw_data_inv)
            inv_messages.append(messages)
            inv_inputs.append(inputs)
        print(f"===> Prepare inputs (start={inv_start_id}, end={inv_end_id-1}, num={len(inv_inputs)})")
        # ========= prepare inputs =========

        # ========= run batch inference =========
        inv_outputs = llm.generate(inv_inputs, sampling_params=sampling_params)
        inv_results = []

        for inv_raw, inv_output in zip(inv_raws, inv_outputs):
            response = inv_output.outputs[0].text
            # response_final = str(response).split("</think>")[-1].strip()

            inv_result = copy.deepcopy(inv_raw)
            inv_result["response"] = response
            inv_results.append(inv_result)
        # save
        output_path = os.path.join(output_dir, f"{model_tag}_response_{inv_start_id}_{inv_end_id-1}.json")
        save_json(output_path, inv_results)
        # ========= run batch inference =========

def build_batch_prompt(datas, image_dir, images_cache_map: dict = None, min_pixels: int = 1280*28*28, max_pixels: int = 5120*28*28):
    batch_messages = []
    for data in datas:
        batch_messages.append(build_prompt(data, image_dir, images_cache_map, min_pixels, max_pixels))
    return batch_messages

def prepare_inputs_for_hf(messages, processor, is_batch=False):
    if not is_batch:
        messages = [messages]

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs

def run_hf_inference(args):
    hf_logger = logging.getLogger("transformers")
    hf_logger.setLevel(logging.ERROR)

    set_seed(42)

    # ========= raw data =========
    raw_datas = read_json(args.data)
    print(f"===> Load data from {args.data}")
    # ========= raw data =========

    # ========= cache image =========
    image_dir, image_cache = args.image_dir, args.image_cache
    images_cache_map = cache_image(image_dir, image_cache)
    if images_cache_map is not None:
        print(f"===> Cache images (num={len(images_cache_map.keys())})")
    else:
        print(f"===> Not Cache images")
    # ========= cache image =========

    # ========= set up generation parameters (HF) =========
    generation_kwargs = {
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    print(f"===> Generation parameters (HF):")
    print(f"max_tokens={args.max_new_tokens}")
    print(f"temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}")
    # ========= set up generation parameters (HF) =========

    # ========= load processor =========
    processor = AutoProcessor.from_pretrained(args.model_path)
    processor.tokenizer.padding_side = "left"
    print("===> Load Processor")
    # ========= load processor =========

    # ========= initialize HF model =========
    model_tag = args.model_tag

    print(f"===> Initialize HF ({model_tag}) with model: {args.model_path}")
    print(f"GPU count: {torch.cuda.device_count()}")

    model_kwargs = {
        "attn_implementation": "flash_attention_2",
        "dtype": torch.bfloat16,
        "device_map": "auto",
    }

    if "qwen25vl" in model_tag:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    elif "qwen3vl_moe" in model_tag:
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    elif "qwen3vl" in model_tag:
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_path, **model_kwargs)
    else:
        raise ValueError(f"Unknown model tag {model_tag}")

    model.eval()
    print(f"===> Initialize HF model ({model_tag}) successfully")
    # ========= initialize HF model =========

    # ========= run batch inference =========
    print("=" * 80)
    print("Running HF batch inference")
    print("=" * 80)
    batch_size = args.batch_size
    raw_datas_num = len(raw_datas)
    model_tag = args.model_tag
    save_interval = args.save_interval
    output_dir = args.output_dir

    # ========= run interval inference =========
    for inv_start_id in range(0, raw_datas_num, save_interval):
        inv_end_id = min(inv_start_id+save_interval, raw_datas_num)

        # prepare interval data
        raw_datas_inv = raw_datas[inv_start_id:inv_end_id]

        # ========= prepare inputs =========
        inv_raws, inv_inputs, inv_messages = [], [], []
        raw_datas_inv_num = len(raw_datas_inv)
        for batch_start_id in tqdm(range(0, raw_datas_inv_num, batch_size), desc="Building prompts (batch)"):
            batch_end_id = min(batch_start_id+batch_size, raw_datas_inv_num)
            raw_datas_inv_batch = raw_datas_inv[batch_start_id:batch_end_id]
            batch_messages = build_batch_prompt(raw_datas_inv_batch, image_dir, images_cache_map=images_cache_map, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            batch_inputs = prepare_inputs_for_hf(batch_messages, processor, is_batch=True)

            inv_raws.append(raw_datas_inv_batch)
            inv_messages.append(batch_messages)
            inv_inputs.append(batch_inputs)

        print(f"===> Prepare inputs (start={inv_start_id}, end={inv_end_id-1}, num={len(inv_inputs)})")
        print(f"===> Batch num={len(inv_inputs)}, batch size={batch_size}")
        # ========= prepare inputs =========

        # ========= run batch inference =========
        inv_results = []
        for batch_id, batch_inputs in tqdm(enumerate(inv_inputs), total=len(inv_inputs), desc="Batch Inference", leave=True):
            inputs = {k: v.to(model.device) for k, v in batch_inputs.items()}
            # batch_inputs = batch_inputs.to(model.device)
            ## https://huggingface.co/Qwen/Qwen2.5-VL-nB-Instruct
            with torch.no_grad():
                generated_ids = model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(batch_inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            ##

            # flatten batch and save one by one
            raw_datas_inv_batch = inv_raws[batch_id]
            for inv_raw, inv_output in zip(raw_datas_inv_batch, output_texts):
                response = inv_output
                inv_result = copy.deepcopy(inv_raw)
                inv_result["response"] = response
                inv_results.append(inv_result)

            #del inputs, generated_ids, generated_ids_trimmed
            #torch.cuda.empty_cache()
        # ========= run batch inference =========
        # save
        output_path = os.path.join(output_dir, f"{model_tag}_hf_response_{inv_start_id}_{inv_end_id-1}.json")
        save_json(output_path, inv_results)
    # ========= run interval inference =========

def main():
    parser = argparse.ArgumentParser(description="QwenX-VL inference with vLLM or HuggingFace")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with vLLM")
    infer_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    infer_parser.add_argument("--data", type=str, help="The absolute path of total data file")
    infer_parser.add_argument("--image-dir", type=str, help="The absolute path of image directory")
    infer_parser.add_argument("--image-cache", type=int, help="The maximum number of images to cache. (default: 0)", default=0)
    infer_parser.add_argument("--model-tag", type=str, required=True, help="model tag (save uuid)")
    infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")

    # vLLM specific parameters
    infer_parser.add_argument("--tensor-parallel-size", type=int, default=None,
                              help="Tensor parallel size (default: number of GPUs)")
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                              help="GPU memory utilization (0.0-1.0, default: 0.9)")
    infer_parser.add_argument("--max-model-len", type=int, default=128000,
                              help="Maximum model context length (default: 128000, balance between performance and memory)")
    infer_parser.add_argument("--max-images-per-prompt", type=int, default=16,
                              help="Maximum images per prompt (default: 16)")

    # Generation parameters
    infer_parser.add_argument("--max-new-tokens", type=int, default=32768,
                              help="Maximum number of tokens to generate (default: 2048)")
    infer_parser.add_argument("--temperature", type=float, default=0.7,
                              help="Temperature for sampling (default: 0.7 for greedy-like decoding)")
    infer_parser.add_argument("--top-p", type=float, default=0.8,
                              help="Top-p for sampling (default: 0.8 for greedy-like decoding)")
    infer_parser.add_argument("--top-k", type=int, default=20,
                              help="Top-k for sampling (default: 20 for greedy decoding)")
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0,
                              help="Repetition penalty (default: 1.0, increase to 1.2-1.5 to reduce repetition)")
    infer_parser.add_argument("--presence-penalty", type=float, default=1.5,
                              help="Presence penalty (default: 1.5, range: 0.0-2.0, penalize tokens that have already appeared)")

    infer_parser.add_argument("--min-pixels", type=int, default=1280*28*28, help="Qwen2.5-VL processor min pixels (default: ~1M pixels)")
    infer_parser.add_argument("--max-pixels", type=int, default=5120*28*28, help="Qwen2.5-VL processor max pixels (default: ~4M pixels)")
    infer_parser.add_argument("--save-interval", type=int, default=256, help="Save response every interval ")

    # infer with hf
    hf_infer_parser = subparsers.add_parser("hf_infer", help="Run inference with HF")
    hf_infer_parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    hf_infer_parser.add_argument("--data", type=str, help="The absolute path of total data file")
    hf_infer_parser.add_argument("--image-dir", type=str, help="The absolute path of image directory")
    hf_infer_parser.add_argument("--image-cache", type=int, help="The maximum number of images to cache. (default: 0)",
                              default=0)
    hf_infer_parser.add_argument("--model-tag", type=str, help="model tag (default: qwen25vl)", default="qwen25vl")
    hf_infer_parser.add_argument("--model-path", type=str, required=True, help="Path to the model")

    # Generation parameters
    hf_infer_parser.add_argument("--max-new-tokens", type=int, default=32768, help="Maximum number of tokens to generate (default: 2048)")
    hf_infer_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (default: 0.7 for greedy-like decoding)")
    hf_infer_parser.add_argument("--top-p", type=float, default=0.8, help="Top-p for sampling (default: 0.8 for greedy-like decoding)")
    hf_infer_parser.add_argument("--top-k", type=int, default=20, help="Top-k for sampling (default: 20 for greedy decoding)")
    hf_infer_parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size (default: 16)")

    hf_infer_parser.add_argument("--min-pixels", type=int, default=1280 * 28 * 28, help="Qwen2.5-VL processor min pixels (default: ~1M pixels)")
    hf_infer_parser.add_argument("--max-pixels", type=int, default=5120 * 28 * 28, help="Qwen2.5-VL processor max pixels (default: ~4M pixels)")
    hf_infer_parser.add_argument("--save-interval", type=int, default=256, help="Save response every interval")

    args = parser.parse_args()

    # Automatically set tensor_parallel_size
    if args.command == "infer" and args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()
        print(f"Auto-set tensor_parallel_size to {args.tensor_parallel_size}")

    if args.command == "infer":
        run_inference(args)
    elif args.command == "hf_infer":
        run_hf_inference(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
