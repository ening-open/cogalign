# Reproduce the CogAlign paper

Table and figure numbers follow [arXiv:2610.41951v1](https://arxiv.org/html/2610.41951v1), the public manuscript accompanying the NeurIPS 2026 poster. Reported scores below are reference values from the paper. A rerun can start with prepared data or saved predictions. See [data and checkpoints](data.md) for input formats and [shell examples](../scripts/README.md) for the current launchers.

| Paper result | Start from | Remaining inputs for the paper result |
| --- | --- | --- |
| **Table 1 and Table 9** · MMMU and MMBench | LMMs-Eval with baseline and aligned Qwen3-VL checkpoints | Original harness revision, model adapter and task config, checkpoint revisions |
| **Table 2** · THINGS O1O human consistency | `scripts/infer_qwenvl.sh` and `cogalign.evaluation.odd_one_out` | Exact 453,642-triplet split, prepared prompts, checkpoints, and paired McNemar run |
| **Table 3** · Prompting versus alignment | Same O1O pipeline with condition-specific prompts and scoring | Original `d*` assignments, prepared prompts and original condition-specific parsers |
| **Table 4** · Architecture transfer | O1O scoring with saved Qwen2.5-VL and Gemma3 predictions | Exact one-sixth split, model-family training commands, weights and predictions |
| **Table 5** · Rationale diagnostic | Kimi judging and local rationale metrics | Original sampled IDs, rationale files, and archived judge outputs |
| **Tables 10–11** · Supervision ablations | SFT with one prepared dataset per condition | Original dimension-only, choice-only and generic-CoT datasets and run configs |
| **Figures 4–5** · NSD fMRI RSA | Extract raw model features, then run searchlight on converted features and preprocessed fMRI | Stimulus manifest, hidden-state conversion, volume/ROI preparation, and figure/statistics scripts |
| **Figure 6** · 48-object behavioral RSA | `cogalign.evaluation.choice_rsm` and the bundled Octave workflow | Original ordered encoded choices or predictions with the original object-ID mapping |

## Environment and run records

Use Python 3.10+ and Bash. Install the extras for the experiment you plan to run. For GPU workflows, select a compatible PyTorch/CUDA setup first. SFT and hidden-state extraction use FlashAttention 2. The `device_map=auto` paths also need `accelerate`. Octave is required for the 48-object MATLAB scripts.

```bash
python -m pip install -e .
# Add the extras you need in the matching environment.
python -m pip install -e '.[train]'
python -m pip install -e '.[vllm]'
```

For each run, record the source commit, input file hashes, split and ordered sample IDs, checkpoint and processor revisions, prompt version, package versions, GPU count and memory, and the full command. Save the resolved generation or training parameters alongside the artifacts. The repository does not include a lockfile for the original CUDA environment.

## Train CogAlign from the prepared hybrid split

The public paper uses 80,000 THINGS alignment examples and 229,329 general multimodal examples. Base the prepared training and validation JSON on the original sample IDs. Use the original SPoSE embedding, rationale generator, and prompt versions if rebuilding the split. The checkout starts from prepared JSON, with the assistant target containing the rationale followed by the human choice. [Data schemas](data.md#build-or-obtain-the-sft-inputs) · [Paper Appendices G and H.1](https://arxiv.org/html/2610.41951v1#A7)

This example starts the supplied Qwen3-VL-4B-Instruct recipe. Replace the paths with your prepared inputs and pinned base checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
  bash scripts/sft_qwen3vl_4b_it_align.sh \
  /path/to/qwen3-vl-4b-instruct /path/to/images \
  /path/to/train.json /path/to/validation.json runs/qwen3vl-4b-align
```

The launcher uses six processes, a per-device train batch of 4 and 6 accumulation steps, giving a global batch of 144. It selects ZeRO-2, seed 42, learning rate `1e-5`, a cosine schedule, 3% warmup, and one epoch. It freezes vision and trains the merger and language model. The paper reports six NVIDIA A800 80 GB GPUs, AdamW, and zero weight decay. Use a fresh output directory for a new run, because the trainer resumes if it finds checkpoint directories. For the 8B or 30B-A3B runs, use the [generic SFT launcher](../scripts/sft_qwenvl.sh) with the original model-specific settings. [Paper Table 7](https://arxiv.org/html/2610.41951v1#A8)

## O1O inference and scoring

The public paper compares the unaligned model with Direct and standard CoT prompts against CogAlign. Prepare separate Direct and CoT records for exactly the same 453,642 IDs, labels and image choices. The standard CoT input is used for both the baseline CoT and aligned runs. At inference the aligned model does not receive the assigned dimension or training rationale. Use the public prompt appendix to construct inputs and the original prompt files for an exact comparison. [Paper Table 2 and Appendix J](https://arxiv.org/html/2610.41951v1#S4)

The public evaluation settings are vLLM, BF16, temperature 0, top-p 1, top-k −1, repetition penalty 1.05 and 8,192 maximum new tokens. The script below applies those CLI flags for a 4B CoT comparison. Supply the archived values of the other settings, including presence penalty, image preprocessing bounds and context length, before claiming a numerical reproduction. Verify the engine's resolved dtype in its startup log. The current launcher has no explicit dtype switch. [Paper Table 8](https://arxiv.org/html/2610.41951v1#A8)

```bash
IMAGE_DIR=/path/to/things-images
COT_JSON=/path/to/o1o-cot-heldout.json
BASE_MODEL=/path/to/qwen3-vl-4b-instruct
ALIGNED_MODEL=/path/to/qwen3-vl-4b-aligned

for condition in baseline aligned; do
  model="$BASE_MODEL"
  if [ "$condition" = aligned ]; then model="$ALIGNED_MODEL"; fi
  bash scripts/infer_qwenvl.sh infer "runs/o1o-$condition" \
    "$COT_JSON" "$IMAGE_DIR" "$model" "qwen3vl_4b_$condition" \
    --temperature 0 --top-p 1 --top-k -1 \
    --repetition-penalty 1.05 --max-new-tokens 8192 \
    --presence-penalty "${PRESENCE_PENALTY:?set from the archived run}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:?set from the archived run}" \
    --max-model-len "${MAX_MODEL_LEN:?set from the archived run}" \
    --min-pixels "${MIN_PIXELS:?set from the archived run}" \
    --max-pixels "${MAX_PIXELS:?set from the archived run}"
done
```

Run the baseline once more with the Direct input file and a separate output directory and tag. Each vLLM run writes `<tag>_response_<first>_<last>.json` shards. Use a fresh directory or tag for each condition. Replay existing shards by skipping generation and pointing the following command at their paths. The checks reject missing, duplicate or unmatched IDs and mismatched labels or option order before scoring.

```bash
python - <<'PY'
from cogalign.evaluation.odd_one_out import (
    build_correctness,
    calculate_bootstrap_ci,
    merge_infer_result,
)

patterns = {
    "baseline": "runs/o1o-baseline/qwen3vl_4b_baseline_response_*.json",
    "aligned": "runs/o1o-aligned/qwen3vl_4b_aligned_response_*.json",
}
reference = None
for condition, pattern in patterns.items():
    records = merge_infer_result(pattern)
    signature = [(r["id"], r["label_choice"], r["option"]) for r in records]
    if len(signature) != 453642 or len({record[0] for record in signature}) != 453642:
        raise ValueError(f"{condition}: expected 453,642 unique predictions")
    if reference is not None and signature != reference:
        raise ValueError(f"{condition}: IDs, labels or option order differ")
    reference = signature
    mean, low, high, _ = calculate_bootstrap_ci(build_correctness(records))
    print(f"{condition}: {mean:.2%} (95% CI {low:.2%} to {high:.2%})")
PY
```

The paper’s 4B reference is 41.48% Direct, 53.37% CoT and 56.81% CogAlign. The scorer estimates bootstrap intervals. For the reported paired McNemar test, recover the original statistic and sample inclusion rules. Confirm the original answer parser before comparing results, particularly for standalone Direct answers and malformed outputs.

For **Table 3**, hold the unaligned 4B checkpoint and test IDs fixed for the three prompt variants, then compare their scores with the aligned 4B checkpoint. The public scores for S-CoT, `d*`-CoT, FS-CoT and CogAlign are 53.37%, 49.01%, 52.70% and 56.81%. Appendix H.4 uses more permissive parsing for the two dimension-prompt conditions. Use its archived parser and failure policy for that table. **Table 4** evaluates Qwen2.5-VL-7B and Gemma3-12B on the same one-sixth subset, approximately 75.6K triplets. **Tables 10–11** additionally vary the prepared SFT target, so each condition needs its own training input and checkpoint. [Paper §4.4 and Appendices H.4–H.5](https://arxiv.org/html/2610.41951v1#S4.SS4)

## MMMU and MMBench

Run MMMU validation and MMBench English test for both baseline and aligned checkpoints. The paper uses the defaults of its LMMs-Eval revision. Install the original revision in a separate environment if available and retain its model adapter, task config, and evaluation defaults. Use the same revision and flags for all compared checkpoints. The example uses current task identifiers from upstream and shows the command shape. [LMMs-Eval tasks](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) · [Paper Tables 1 and 9](https://arxiv.org/html/2610.41951v1#S4)

```bash
python -m lmms_eval --model qwen3_vl \
  --model_args pretrained=/path/to/qwen3-vl-checkpoint \
  --tasks mmmu_val,mmbench_en_test --output_path runs/lmms-eval/baseline
```

Repeat with the aligned checkpoint and its own output directory. Retain the task-level metrics for Table 1 and the MMBench category breakdown for Table 9.

## 48-object behavioral RSA

Figure 6 compares the human RSM with Qwen2.5-VL-7B CoT, CogAlign and Centaur on approximately 91.56K triplets from the 48-object subset. The checked-in 48-object inference example uses Qwen3-VL-4B and its own generation settings. For Figure 6, start with archived Qwen2.5-VL and Centaur predictions, or the encoded triplets derived from them. Align predictions to triplet IDs and map the two chosen similar objects and the odd choice to the released object index before invoking `choice_rsm`. [Triplet schema](data.md#behavioral-rsm-and-neural-rsa-formats) · [Paper Figure 6 and Appendix H.7](https://arxiv.org/html/2610.41951v1#A8.SS7)

```bash
python -m cogalign.evaluation.choice_rsm \
  --data-path /path/to/qwen25vl-aligned-encoded-triplets.json \
  --output-dir runs/behavioral-rsm --model-tag aligned

bash experiments/behavioral_rsa_48/reproduce.sh \
  runs/behavioral-rsm runs/behavioral-rdm \
  experiments/behavioral_rsa_48/human aligned
```

Repeat the first command for `baseline` and `centaur`, then pass all three tags to the Octave workflow to print their scores and confidence bounds. Python writes whole and split-half 1,854-object RSMs. Octave selects the 48 objects, writes `RDM_<tag>_48object.mat` and `RDM_<tag>_48object_half.mat`, and compares them with the human reference. The current code normalizes by pair opportunities and applies a split-half reliability correction. Confirm these choices against the archived figure inputs before comparing to the reported 73.30%, 80.93% and 65.13% (CoT, CogAlign and Centaur).

## NSD fMRI searchlight RSA

Figures 4–5 use Qwen3-VL-4B-Instruct and participants 1, 2, 5 and 7. The paper extracts `<|vision_end|>` features, compares Pearson-distance RDMs with Spearman correlation, and contrasts aligned minus baseline RSA. Figure 4 uses concatenated last-four-layer features. Figure 5 compares that extraction with the last and penultimate layers for participant 1. [Paper §3.4, Figures 4–5 and Appendices F.1 and H.6](https://arxiv.org/html/2610.41951v1#S3.SS4)

1. **Extract.** Run once per checkpoint on the same ordered stimulus JSON and image directory. The entry point writes `hidden_state_*.pkl` raw feature shards.

   ```bash
   python -m pip install -e '.[neural]' accelerate
   python -m cogalign.neural.extract_hidden_states forward \
     --data-path /path/to/nsd-stimuli.json --image-dir /path/to/nsd-images \
     --model-path /path/to/4b-checkpoint --model-tag baseline \
     --output-dir runs/nsd-extraction --batch-size 1
   ```

2. **Convert and align.** Use the original experiment's tokenizer and layer selection to build the [analysis-input pickle](data.md#behavioral-rsm-and-neural-rsa-formats) for each checkpoint and extraction depth. Concatenating raw shards alone leaves the fields in the wrong format for searchlight. The conversion and preparation of the four stimulus-aligned fMRI volumes are required external steps.
3. **Run searchlight.** Supply the converted features and four volumes for each participant. The command writes score, normalized score and noise-ceiling `.npy` volumes. Keep separate tags and use the same stimulus IDs for both model conditions.

   ```bash
   python -m cogalign.neural.searchlight_rsa searchlight \
     --hidden-state /path/to/converted-features.pkl --subject subj01 \
     --volume /path/to/subj01-aggregate.h5 \
     --volume-trial1 /path/to/subj01-trial-1.h5 \
     --volume-trial2 /path/to/subj01-trial-2.h5 \
     --volume-trial3 /path/to/subj01-trial-3.h5 \
     --output-dir runs/nsd-searchlight --tag baseline
   ```

4. **Visualize and compare.** Use `python -m cogalign.neural.plot_searchlight --help` for arguments. It accepts explicit paths for the human, baseline and aligned volumes and uses the corresponding Pycortex transform and filestore. The plotter updates the user Pycortex config. For exact figures, match the original ROI/mask handling, paired statistics and rendering. The paper specifies a radius-3 sphere and a nonnegative visualization floor. The current code enumerates a radius-3 cube before clipping at volume boundaries and displays signed differences. Use the archived implementation or outputs for the paper figures.

## Rationale quality and generator comparison

Appendix C compares rationales generated by Qwen2.5-72B-Instruct and DeepSeek-V4-Flash, using Kimi-K3 to score dimension faithfulness, factual correctness, logical validity and non-forcedness. For comparison with Table 5, replay the original sampled judgments and verify the selected judge model. A fresh API run can depend on service revisions and fallback behavior. The repository also offers local [rationale metrics](../src/cogalign/evaluation/rationale_metrics.py) for embedding similarity and token counts. [Paper Appendix C](https://arxiv.org/html/2610.41951v1#A3)

```bash
python -m pip install -e '.[judge]'
# Set MOONSHOT_API_KEY in the environment before running the judge.
python -m cogalign.evaluation.kimi_judge \
  --data-path /path/to/judge-sample.json --image-dir /path/to/things-images \
  --output runs/rationale-judgments
```
