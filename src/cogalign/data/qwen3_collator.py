import re
import torch
from typing import Dict, Optional, Sequence, List, Tuple, Any

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

def get_rope_index_3(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    """Different from the original implementation, Qwen3VL use timestamps rather than absolute time position ids."""
    # Since we use timestamps to seperate videos, like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>, the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                # t_index is always 0 because llm_grid_t is always 1 (we use timestamps to encode the temporal information for videos)
                t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return position_ids, mrope_position_deltas

class Qwen3VLCollator:
    def __init__(self, processor, **kwargs):
        self.video_max_total_pixels = getattr(kwargs, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(kwargs, "video_min_total_pixels", 256 * 28 * 28)

        # get_rope_index
        self.get_rope_index = get_rope_index_3
        # update processor
        processor = self.update_processor_pixels(processor, **kwargs)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

    def update_processor_pixels(self, processor, **kwargs):
        # --- Image Processor ---
        ip = processor.image_processor
        if hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
            ip.min_pixels = getattr(kwargs, "min_pixels", ip.min_pixels)
            ip.max_pixels = getattr(kwargs, "max_pixels", ip.max_pixels)

        if hasattr(ip, "size") and isinstance(ip.size, dict):
            ip.size["shortest_edge"] = getattr(kwargs, "min_pixels", ip.size["shortest_edge"])
            ip.size["longest_edge"] = getattr(kwargs, "max_pixels", ip.size["longest_edge"])

        # --- Video Processor ---
        if hasattr(processor, "video_processor") and processor.video_processor is not None:
            vp = processor.video_processor

            if hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
                vp.min_pixels = getattr(kwargs, "video_min_pixels", vp.min_pixels)
                vp.max_pixels = getattr(kwargs, "video_max_pixels", vp.max_pixels)

            if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
                vp.min_frames = getattr(kwargs, "video_min_frames", vp.min_frames)
                vp.max_frames = getattr(kwargs, "video_max_frames", vp.max_frames)

            if hasattr(vp, "fps"):
                vp.fps = getattr(kwargs, "video_fps", vp.fps)

            if hasattr(vp, "size") and isinstance(vp.size, dict):
                vp.size["shortest_edge"] = getattr(kwargs, "video_min_pixels", vp.size["shortest_edge"])
                vp.size["longest_edge"] = getattr(kwargs, "video_max_pixels", vp.size["longest_edge"])
        return processor

    def build_messages(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Extract and normalize images and videos
        images = item.get("image") or []
        if isinstance(images, str):
            images = [images]

        videos = item.get("video") or []
        if isinstance(videos, str):
            videos = [videos]

        # Build media pools with absolute paths
        image_pool = [
            {"type": "image", "image": img} for img in images
        ]
        video_pool = [
            {"type": "video", "video": vid} for vid in videos
        ]

        messages = []
        for turn in item["conversations"]:
            if turn["from"] == "human":
                role = "user"
            elif turn["from"] == "system":
                role = "system"
            else:
                role = "assistant"

            text: str = turn["value"]

            # interleave content
            if role == "user":
                content = []
                # Split text by <image> or <video> placeholders while keeping delimiters
                text_parts = re.split(r"(<image>|<video>)", text)

                for seg in text_parts:
                    if seg == "<image>":
                        if not image_pool:
                            raise ValueError(
                                "Number of <image> placeholders exceeds the number of provided images"
                            )
                        content.append(image_pool.pop(0))
                    elif seg == "<video>":
                        if not video_pool:
                            raise ValueError(
                                "Number of <video> placeholders exceeds the number of provided videos"
                            )
                        content.append(video_pool.pop(0))
                    elif seg.strip():
                        content.append({"type": "text", "text": seg.strip()})

                messages.append({"role": role, "content": content})
            else:
                # Assistant messages contain only text
                messages.append({"role": role, "content": [{"type": "text", "text": text}]})

        # Check for unused media files
        if image_pool:
            raise ValueError(
                f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
            )
        if video_pool:
            raise ValueError(
                f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
            )
        return messages

    def preprocess_qwen_visual(self, sources, processor) -> Dict:
        if len(sources) != 1:
            raise ValueError(f"Expected 1 source, got {len(sources)}")

        source = sources[0]
        messages = self.build_messages(source)
        full_result = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
        input_ids = full_result["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == 77091:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != 151645:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start: ans_end + 2] = input_ids[
                        0, ans_start: ans_end + 2
                    ]
                    pos = ans_end
            pos += 1

        full_result["labels"] = labels
        full_result["input_ids"] = input_ids
        return full_result

    def preprocess(self, sources) -> Dict[str, torch.Tensor]:
        if isinstance(sources, dict):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        data_dict = self.preprocess_qwen_visual(sources, self.processor)
        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [self.processor.video_processor.temporal_patch_size / self.processor.video_processor.fps] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]

        """
        text = self.processor.tokenizer.decode(
            data_dict["input_ids"][0], skip_special_tokens=False
        )
        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        label = self.processor.tokenizer.decode(labels, skip_special_tokens=False)
        """
        return data_dict

    def pad_and_cat(self, tensor_list):
        max_length = max(tensor.shape[2] for tensor in tensor_list)
        padded_tensors = []
        for tensor in tensor_list:
            pad_length = max_length - tensor.shape[2]
            padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
            padded_tensors.append(padded_tensor)

        stacked_tensor = torch.cat(padded_tensors, dim=1)
        return stacked_tensor

    def process_batch(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = self.pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        return batch

    def convert_sample(self, data: dict):
        image = data.get("image", [])
        video = data.get("video", [])
        system = data.get("system", None)
        input = data["input"]
        output = data["output"]

        conversations = []
        if system is not None:
            conversations.append({
                "from": "system",
                "value": system,
            })

        """ attention """
        if len(image) > 0 and "<image>" not in input:
            image_token = "<image>\n" * len(image)
            human_text = f"{image_token}{input}"
        elif len(video) > 0 and "<video>" not in input:
            video_token = "<video>\n" * len(video)
            human_text = f"{video_token}{input}"
        else:
            human_text = input

        conversations.extend([
            {
                "from": "human",
                "value": human_text
            },
            {
                "from": "gpt",
                "value": output
            }
        ])

        new_data = {
            "image": image,
            "video": video,
            "conversations": conversations,
        }
        return new_data

    def __call__(self, samples):
        new_samples = []
        for sample in samples:
            sample = self.convert_sample(sample)
            sample = self.preprocess(sample)
            new_samples.append(sample)
        return self.process_batch(new_samples)
