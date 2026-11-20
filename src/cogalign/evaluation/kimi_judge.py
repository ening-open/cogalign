import os
import json
from typing import Union
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
import base64

MODEL = "kimi-k3"
#MODEL = "kimi-k2.6"

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_json(data: Union[list, dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def read_data(path):
    raws = read_json(path)
    datas = []
    for raw in raws:
        idx = raw["id"]
        response = raw["response"]
        text_splits = response.split(":", maxsplit=1)
        assert len(text_splits) == 2
        rationale = text_splits[-1].strip()
        triplet_name = raw["triplet_name"]
        triplet_image = raw["image"]
        dim_name = raw["dim_name"]
        datas.append({
            "id": idx,
            "triplet_name": triplet_name,
            "triplet_image": triplet_image,
            "rationale": rationale,
            "dim_name": dim_name,
        })
    return raws, datas

client = None

def init_worker(api_key, dataset_image_dir):
    global client, image_dir
    image_dir = dataset_image_dir
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.cn/v1",
    )

image_dir = None

def read_image(path):
    with open(path, "rb") as f:
        image_data = f.read()
    return image_data

SYSTEM = """You are a strict evaluator of behavior-grounded visual rationales.

The semantic dimension is fixed and has already been identified from validated
human behavioral data. Do not evaluate whether the dimension itself is correct,
optimal, or preferable to another dimension. Do not replace or reinterpret it.
If a dimension definition is provided, treat that definition as authoritative.

Evaluate only whether the rationale validly verbalizes and applies the GIVEN
dimension to images A, B, and C.

Assess the following criteria independently:

D — Dimension faithfulness
Return 1 if the given dimension is the actual basis of the rationale's comparison
among the objects. The rationale need not repeat the dimension label verbatim,
but merely mentioning it without applying it is insufficient. This criterion
concerns use of the dimension, not whether the object claims are factually true.

F — Factual correctness
Return 1 if every decision-relevant claim about the objects is supported by the
images or stable common knowledge. Return 0 if any claim needed for the conclusion
is false, visually contradicted, or unsupported. Ignore inconsequential wording
or stylistic imprecision.

L — Logical validity
Treat the rationale's supporting claims as premises and assess the inference
separately from their factual truth. Return 1 if the premises are internally
consistent, provide a sufficient comparison, and uniquely support one object as
the odd-one-out under the given dimension. If an explicit final answer is present,
it must agree with the premises; otherwise return 0.

N — Non-forcedness
Return 1 if every dimension-object relation needed for the conclusion is
conventional, characteristic, and discriminative. Return 0 if the conclusion
depends on an ad hoc, remote, hypothetical, accidental, merely symbolic, or
overly permissive association. An indirect relation may pass when it is a
well-established and characteristic relation of the object. A relation being
factually possible is not by itself sufficient.

Infer the odd-one-out from the supporting premises, not from an explicitly stated
answer. The odd-one-out is the object excluded from the uniquely supported
most-similar pair under the given dimension.

Use:
- A for the first image
- B for the second image
- C for the third image
- U if the premises do not uniquely determine an odd-one-out

Judge the four criteria independently. For example, a rationale may use the given
dimension faithfully but contain a factual error, or it may be logically valid
conditional on premises that are factually false.

Output exactly one valid JSON object with these five keys and no additional text:
{"d":0,"f":0,"l":0,"n":0,"o":"U"}

Each of d, f, l, and n must be either 0 or 1.
The value of o must be exactly "A", "B", "C", or "U".
Do not output explanations, confidence scores, markdown, or additional keys."""

USER_TEMPLATE = """Images are provided in the following order:
A = first image
B = second image
C = third image

Given semantic dimension:
{dim_name}

Rationale:
{rationale}

Evaluate the rationale according to the system instructions."""

def build_prompt(triplet_image, dim_name, rationale):
    global image_dir
    image_urls = []
    for image_name in triplet_image:
        image_path = os.path.join(image_dir, image_name)
        image_data = read_image(image_path)
        image_url = f"data:image/{os.path.splitext(image_path)[1].lstrip('.')};base64,{base64.b64encode(image_data).decode('utf-8')}"
        image_urls.append(image_url)

    system = SYSTEM
    user = USER_TEMPLATE.format(dim_name=dim_name, rationale=rationale)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [
            {"type": "image_url", "image_url":{"url": image_urls[0]}},
            {"type": "image_url", "image_url":{"url": image_urls[1]}},
            {"type": "image_url", "image_url":{"url": image_urls[2]}},
            {"type": "text", "text": user},
        ]}
    ]
    return messages

def request(data):
    idx = data["id"]
    triplet_name, triplet_image = data["triplet_name"], data["triplet_image"]
    rationale, dim_name = data["rationale"], data["dim_name"]

    messages = build_prompt(triplet_image, dim_name, rationale)

    try:
        #""" # kimi-k3  select each dim with ten items? because it is too expensive
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=False,
            reasoning_effort="high",
        )
        #"""
        """ # kimi-k2.6
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
            #reasoning_effort="high",
        )
        """
        response = response.model_dump_json()

        return {
            "id": idx,
            "response": response,
            "state": 1,
        }

    except Exception as exc:
        return {
            "id": idx,
            "response": response,
            "state": repr(exc),
        }

def main(args):
    data_path = args.data_path

    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise SystemExit("Set MOONSHOT_API_KEY before running the Kimi judge.")

    raws, datas = read_data(data_path)

    output_dir = args.output
    output_dir = os.path.join(output_dir, MODEL, "parts")
    os.makedirs(output_dir, exist_ok=True)

    max_workers = 32
    save_interval = 64

    total = len(datas)
    outputs = []
    succ, fail = 0, 0

    part_id = 0

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_worker,
        initargs=(api_key, args.image_dir),
    ) as executor:
        results_iterator = executor.map(request, datas, chunksize=1)
        pbar = tqdm(
            results_iterator,
            total=total, desc=f"{MODEL} request",
            unit="sample", dynamic_ncols=True,
        )

        for i, result in enumerate(pbar):
            data_id = result["id"]
            response = result["response"]
            state = result["state"]
            out_item = {"id": data_id, "response": response, "state": state}

            outputs.append(out_item)

            completed = i + 1

            if state == 1:
                succ += 1
            else:
                fail += 1
                tqdm.write(
                    f"[Failed] index={i}, "
                    f"id={id}, "
                    f"error={state}"
                )

            pbar.set_postfix(success=succ, failed=fail)

            if completed % save_interval == 0:
                start_id = completed - len(outputs)
                end_id = completed - 1
                output_path = os.path.join(output_dir, f"{MODEL}_judge_{part_id:05d}_{start_id:08d}_{end_id:08d}.json")
                save_json(outputs, output_path)

                outputs = []
                part_id += 1

    if len(outputs) > 0:
        output_path = os.path.join(output_dir, f"{MODEL}_judge_last.json")
        save_json(outputs, output_path)


    print("Finish All Requests!")
    print(f"Success: {succ} samples, Fail: {fail} samples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--image-dir", type=str)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    main(args)
