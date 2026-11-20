# Training and inference scripts

Run these examples from the activated Python environment after installing CogAlign and the corresponding extras. Use Bash. Paths may be relative to the current working directory. [Accelerate launch reference](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch)

## Supervised fine-tuning

`sft_qwenvl.sh` launches the current training module with Accelerate and forwards the SFT options. To select your own launcher configuration, set `ACCELERATE_CONFIG_FILE`.

```bash
bash scripts/sft_qwenvl.sh 6 --help
```

`sft_qwen3vl_4b_it_align.sh` uses the supplied 4B settings, including a global training batch size of 144 (`6 × 4 × 6`), ZeRO stage 2, and cosine scheduling. `base` is the dataset loader name for the prepared mixed JSON. The `SWANLAB_ENABLE` environment variable defaults to `false` for local runs. Set it to `true` to enable tracking after installing the tracking extra. To change the batch size, set `NUM_PROCESSES`, `TRAIN_BATCH_SIZE`, and `GRADIENT_ACCUMULATION_STEPS` as a group. Adjust `EVAL_BATCH_SIZE` for evaluation memory.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/sft_qwen3vl_4b_it_align.sh \
  Qwen/Qwen3-VL-4B-Instruct /path/to/images \
  /path/to/train.json /path/to/eval.json runs/qwen3vl-4b-align
```

Use a fresh output directory for a new run. The SFT module can resume from checkpoints in an existing output directory.

## Inference

`infer_qwenvl.sh` accepts `infer` for vLLM and `hf_infer` for Transformers. It creates the output directory, checks for existing prediction shards with the same backend and tag, and forwards additional options to the selected backend. Use a fresh output directory or tag for a new run. For Transformers inference, use a model tag containing `qwen3vl` for dense Qwen3-VL checkpoints or `qwen3vl_moe` for Qwen3-VL MoE checkpoints.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_qwenvl.sh infer \
  runs/o1o-vllm /path/to/o1o.json /path/to/images /path/to/checkpoint qwen3vl_4b_aligned \
  --tensor-parallel-size 1 --max-model-len 8192 --max-new-tokens 2048 \
  --temperature 0 --top-p 1 --top-k 1 --presence-penalty 0

CUDA_VISIBLE_DEVICES=0 bash scripts/infer_qwenvl.sh hf_infer \
  runs/o1o-hf /path/to/o1o.json /path/to/images /path/to/checkpoint qwen3vl_4b_aligned \
  --batch-size 4 --max-new-tokens 2048
```

`infer_qwen3vl_4b_it_align_48obj.sh` runs the supplied vLLM 48-object settings on one GPU. It looks for files such as `o1o_48object_base_split_01.json` and writes predictions under the corresponding `01/chunks/` output directory. Pass a comma-separated list of split IDs. Use the [reproduction guide](../docs/reproduction.md) for the RSM-to-RDM and scoring steps.

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/infer_qwen3vl_4b_it_align_48obj.sh \
  01,02 /path/to/48object-splits /path/to/images /path/to/checkpoint runs/48object
```
