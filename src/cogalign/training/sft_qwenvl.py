import glob
import random
from dataclasses import dataclass
from typing import Union
import os
import json
import re

from cogalign.data import make_supervised_data, Qwen3VLCollator
from cogalign.training.trainer import SeqSFTTrainer

import accelerate
import numpy as np
import torch

import transformers
from transformers import HfArgumentParser, AutoProcessor, BitsAndBytesConfig, AutoTokenizer
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
from trl import SFTConfig


@dataclass
class Arguments:
    seed: int = 42
    output_dir: str = None

    dataset: str = None
    image_dir: str = None
    train_data: str = None
    eval_data: str = None

    model_type: str = "qwen3vl"  # or qwen3vl_moe
    model_name: str = None
    attn_impl: str = "flash_attention_2"

    tune_vision: bool = False
    tune_merger: bool = False
    tune_llm: bool = False

    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_pattern: str = r".*language_model.*(q|k|v|o|gate|up|down)_proj.*"

    deepspeed: int = 2
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    offload_optimizer: bool = False
    offload_param: bool = False

    lr: float = 5e-05
    epochs: int = 1
    skip_eval: bool = False

    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "linear"

    max_pixels: int = 28 * 28 * 576
    min_pixels: int = 28 * 28 * 16

    use_4bit_quantization: bool = False
    gradient_checkpointing: bool = False
    do_eval: bool = True
    swanlab_enable: bool = False

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_json(data: Union[list, dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def safe_save_hf_model(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def safe_save_lora_model(trainer: transformers.Trainer, output_dir: str):
    model_merged = trainer.model.merge_and_unload()
    merge_output_dir = os.path.join(output_dir, "checkpoint_trained")
    os.makedirs(merge_output_dir, exist_ok=True)
    model_merged.save_pretrained(merge_output_dir)

def set_model(model_args, model):
    if model_args.tune_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_merger:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

def set_ds(zero: int, **kwargs):
    per_device_train_batch_size = kwargs.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = kwargs.get("gradient_accumulation_steps", 1)
    offload_optimizer = kwargs.get("offload_optimizer", False)

    zero_optim = {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": "auto",
        "contiguous_gradients": True
    }
    if offload_optimizer:
        zero_optim["offload_optimizer"] = {"device": "cpu", "pin_memory": True}

    if zero == 3:
        zero_optim = {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        }
        offload_param = kwargs.get("offload_param", False)
        if offload_param:
            zero_optim["offload_param"] = {"device": "cpu", "pin_memory": True}

    ds_config = {
        "bf16": {
            "enabled": True,
        },
        "zero_optimization": zero_optim,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "steps_per_print": 1000,
        "train_micro_batch_size_per_gpu": per_device_train_batch_size,
        "wall_clock_breakdown": False
    }
    return ds_config


def main():
    parser = HfArgumentParser((Arguments, ))
    args, *_ = parser.parse_args_into_dataclasses()
    print(args)

    seed = args.seed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # `accelerate launch` to get process device
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    output_dir = args.output_dir
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(seed)

    model_type, model_name = args.model_type, args.model_name
    per_device_train_batch_size, per_device_eval_batch_size = args.per_device_train_batch_size, args.per_device_eval_batch_size

    # ------------------ dataset ------------------
    max_pixels, min_pixels = args.max_pixels, args.min_pixels
    processor = AutoProcessor.from_pretrained(model_name, max_pixels=max_pixels, min_pixels=min_pixels)
    processor.tokenizer.padding_side = "right"  # right or left is same, because `pad` token will be masked
    # inference need left pad

    # max_pixels, min_pixels = 28 * 28 * 64, 28 * 28 * 1
    collator = Qwen3VLCollator(
        processor=processor,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )

    dataset = args.dataset
    image_dir, train_data, eval_data = args.image_dir, args.train_data, args.eval_data

    with accelerate.PartialState().main_process_first():
        data_config, (train_dataset, eval_dataset) = make_supervised_data(
            name=dataset, image_dir=image_dir, train=train_data, eval=eval_data, test=eval_data, log_dir=log_dir,
            batch_size=per_device_train_batch_size * world_size * args.gradient_accumulation_steps, rank=local_rank,
        )
    # ------------------ dataset ------------------

    # ------------------ model ------------------
    # Qwen-3-VL or Qwen-3-VL-Moe
    attn_impl = args.attn_impl
    model_kwargs = {
        "attn_implementation": attn_impl,
        "dtype": torch.bfloat16,
        # "device_map": "auto",
    }

    bnb_config = None
    if args.use_4bit_quantization:
        print(f"********* Local Rank: {local_rank} 4-Bit quantization! *********")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage="bfloat16",
        )
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = {"": f"cuda:{local_rank}"}

    if re.search(r"a\d+b", model_type):
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        print("========= Model Architecture ========")
        print("===> Qwen3VLMoeForConditionalGeneration")
        print("========= Model Architecture ========")
    elif "qwen3vl" in model_type:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}, supported model types (moe, qwen3vl)")

    ### lora
    lora_enable = args.lora_enable
    if lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print(f"********* Local Rank: {local_rank} LoRA enabled *********")

        for p in model.parameters():
            p.requires_grad = False

        r, lora_alpha, lora_dropout, lora_pattern = args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_pattern
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=[mn for mn, _ in model.named_modules() if re.match(lora_pattern, mn)],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        if args.tune_merger:
            for n, p in model.visual.merger.named_parameters():
                p.requires_grad = True
    else:
        set_model(args, model)

        if local_rank == 0:
            print(f"========= Trainable Parameters ===========")
            print(f"===> Trainable parameters: {model.num_parameters(only_trainable=True):,}")
            print(f"========= Trainable Parameters ===========")
    ### lora

    # save model training info
    model_info = []
    for n, p in model.named_parameters():
        model_info.append({"name": n, "requires_grad": p.requires_grad})
    save_json(model_info, os.path.join(args.output_dir, "model_info.json"))
    # ------------------ model ------------------

    # ------------------ deepspeed ------------------
    zero = args.deepspeed
    gradient_accumulation_steps = args.gradient_accumulation_steps
    offload_optimizer, offload_param = args.offload_optimizer, args.offload_param
    ds_config = set_ds(
        zero,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        offload_optimizer=offload_optimizer,
        offload_param=offload_param,
    )
    # ------------------ deepspeed ------------------

    # ------------------ gradient_checkpointing ------------------
    gradient_checkpointing = args.gradient_checkpointing
    gradient_checkpointing_kwargs = {"use_reentrant": False} if gradient_checkpointing else None

    ## from Qwen3-VL github
    # if gradient_checkpointing:
    #     if hasattr(model, "enable_input_require_grads"):
    #         model.enable_input_require_grads()
    #     else:
    #         def make_inputs_require_grad(module, input, output):
    #             output.requires_grad_(True)
    #         model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    # ------------------ gradient_checkpointing ------------------

    lr = args.lr
    epochs = args.epochs
    lr_scheduler_type = args.lr_scheduler_type
    warmup_ratio = args.warmup_ratio

    do_eval = args.do_eval
    eval_strategy = "epoch" if do_eval else "no"
    eval_on_start = not args.skip_eval

    sft_args = SFTConfig(
        seed=seed,
        do_eval=do_eval,
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch_fused",
        eval_strategy=eval_strategy,
        eval_on_start=eval_on_start,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy=eval_strategy,
        learning_rate=lr,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        bf16=True,
        ## if use zero-3, adopt Qwen3-VL gradient_checkpointing
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        dataset_kwargs={"skip_prepare_dataset": True},  # enable to use passed train and eval dataset
        remove_unused_columns=False,
        deepspeed=ds_config,
        label_names=["labels"],
        report_to="none",
    )

    # ------------------ callbacks ------------------
    callbacks = []
    if args.swanlab_enable:
        from swanlab.integration.transformers import SwanLabCallback

        swanlab = SwanLabCallback(
            project="coglm",
            experiment_name=model_type,
            logdir=args.output_dir
        )
        callbacks.append(swanlab)
    # ------------------ callbacks ------------------

    # ------------------ Train ------------------
    # SeqSFTTrainer: train with training dataset without shuffling
    trainer = SeqSFTTrainer(
        model=model,
        args=sft_args,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        callbacks=callbacks,
    )

    if list(glob.glob(os.path.join(args.output_dir, "checkpoint-*"))):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # save
    try:
        model.config.use_cache = True
        processor.save_pretrained(output_dir)

        safe_save_hf_model(trainer, output_dir=output_dir)

        if lora_enable and (args.tune_merger or args.tune_vision or args.tune_llm):
            print("="*80)
            print("===> Existing unfreezed parameters (not belong to LoRA module), save model to `checkpoint_trained` directory totally")
            print("===> Existing unfreezed parameters (not belong to LoRA module), save model to `checkpoint_trained` directory totally")
            print("===> Existing unfreezed parameters (not belong to LoRA module), save model to `checkpoint_trained` directory totally")
            print("=" * 80)
            safe_save_lora_model(trainer, output_dir=output_dir)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
