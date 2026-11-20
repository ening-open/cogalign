# Data and checkpoints

The sample counts and dataset composition below follow the [public paper](https://arxiv.org/html/2610.41951v1#A7). The checkout includes code and the 48-object reference matrices. Exact training and evaluation splits, model checkpoints, and NSD preprocessing artifacts must be supplied separately.

## Sources and required artifacts

| Source | Role in the paper | What a rerun needs |
| --- | --- | --- |
| [THINGS](https://things-initiative.org/) and SPoSE | Human O1O choices and 66-dimensional object embeddings. The paper uses 80,000 O1O training examples and 453,642 held-out triplets for the main behavioral evaluation. | Exact embedding and object index, train/eval IDs, image selection, assigned dimension, image descriptions, generated rationales, and reference choices as used in the run. |
| [M4-Instruct](https://huggingface.co/datasets/lmms-lab/M4-Instruct-Data) | General-domain supervision. Paper Table 6 lists 229,329 general examples and 309,329 total hybrid examples. | Exact subset manifests for single-image and multi-image tasks, selected upstream images, and the prepared training and validation JSON. |
| [Natural Scenes Dataset](https://naturalscenesdataset.org/) | Independent fMRI alignment analysis for participants `subj01`, `subj02`, `subj05`, and `subj07`. | Ordered stimuli, aggregate and individual-trial HDF5 volumes, precomputed model features, and the Pycortex subject and transform files. |
| [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) and comparison models | Baseline and CogAlign checkpoints for Tables 1–4 and Figures 4–6. | Exact baseline revisions and matching aligned checkpoints with model and processor files. Table 4 and Figure 6 also need their own model-family-specific runs and outputs. |

Record each source revision, split, and access conditions. Consult the upstream terms for raw images, annotations, models, and derived assets.

## Build or obtain the SFT inputs

The paper uses the following sequence. Start from an indexed THINGS triplet and its human choice. Use the fixed SPoSE embedding and the Jackknife probability change to assign the top semantic dimension. Generate a dimension-grounded rationale from the triplet and its descriptions while keeping the human choice fixed. Write the rationale and choice as the assistant target. Combine these examples with the general-domain tasks and keep the original dataset split and sampling order. The paper provides the attribution equations and rationale and training prompts. The current code starts at the prepared JSON files. [Paper §3 and Appendices G and J](https://arxiv.org/html/2610.41951v1#S3)

The `base` loader reads one JSON object per split. Each key is a task name, and each task contains `total` and `data`. Use the actual number of records for `total`, because it contributes to batch allocation. Image paths are relative to the provided image directory. Samples use `conversations` with a user turn and an assistant turn, optionally preceded by a system turn. See the [dataset loader](../src/cogalign/data/dataset.py) for the complete task key list.

```json
{
  "o1o": {
    "total": 1,
    "data": [
      {
        "image": ["o1o/a.jpg", "o1o/b.jpg", "o1o/c.jpg"],
        "conversations": [
          {"role": "user", "text": "Which object is the odd one out?"},
          {"role": "assistant", "text": "A and B are both fruit. Answer: C"}
        ]
      }
    ]
  }
}
```

## O1O prompts, predictions, and IDs

Inference consumes a JSON array. Its prompt field is `conversation` (singular), with `content` in each message. Preserve the original sample `id`, image/option order, and `label_choice` across prompt conditions. The inference module adds `response` to each prediction record. The example illustrates the format. Use the paper prompts and the matching original evaluation split for numerical comparison. [Prompt appendix](https://arxiv.org/html/2610.41951v1#A10)

```json
[
  {
    "id": 0,
    "image": ["o1o/a.jpg", "o1o/b.jpg", "o1o/c.jpg"],
    "conversation": [
      {"role": "user", "content": "Which image is the odd one out? Finish with Answer: A, B, or C."}
    ],
    "option": ["A", "B", "C"],
    "label_choice": "C"
  }
]
```

The Kimi judge reads records with `id`, `triplet_name`, `image`, `dim_name`, and a `response` containing the rationale after the first colon. It sends the images and rationale to Moonshot. Record the original sampled IDs, model identifiers, rubric, and returned judgments for replay.

## Behavioral RSM and neural RSA formats

- **48-object behavior.** Supply ordered triples `[similar_id_1, similar_id_2, odd_id]` using zero-based indices in the 1,854-entry `words.mat` list. The position of the odd ID describes the model choice, so the first two entries must be the chosen similar pair. Reserve `-1` in the third position for unparsed choices. Preserve the input order for the split-half calculation. The workflow subsets to `words48.mat` and compares to the supplied human RDMs under [`experiments/behavioral_rsa_48/human/`](../experiments/behavioral_rsa_48/human/).
- **NSD extraction input.** Supply a JSON array with one relative image name per record and `conversations` in the training schema. The current searchlight loader expects the single-image `info` list to identify `image_{i}.png` at position `i`. Establish that these filenames map to the original ordered NSD stimulus IDs.
- **NSD searchlight input.** For each model and extraction depth, supply an ordered pickle of processed records with `info`, `image_pad`, `vision_end`, and `im_end['user']`. The extractor writes raw `info`, `input_ids`, and `hidden_states`. This requires a token/layer selection and pooling conversion using the original tokenizer and stimulus manifest. Supply one aggregate and three single-trial HDF5 volumes per participant, all with key `data`, shape `(stimulus, x, y, z)`, and the same stimulus row order. Only load trusted pickle and reference files.
