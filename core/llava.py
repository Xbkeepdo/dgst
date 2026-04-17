from __future__ import annotations

# LLaVA backend 和 vICR 主计算入口。
# 这里负责多模态 forward、token layout 解析和 token/object-level 分数生成。

import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils import is_accelerate_available

from .scoring import (
    jsd,
    jsd_standardized,
    masked_topk,
    mean_or_zero,
    projection_distribution,
    projection_scores,
    renormalize,
    topk_values,
)
from .targets import (
    build_object_targets,
    build_object_targets_from_mentions,
    summarize_targets,
)


@dataclass
class VICTokenLayerScore:
    layer: int
    merged_token_index: int
    token: str
    vicr: float
    context_top_indices: List[int]
    attention_top_mass: float


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _group_contiguous(indices: Sequence[int]) -> List[List[int]]:
    groups: List[List[int]] = []
    for index in indices:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def _resolve_model_source(model_id: str | Path) -> tuple[str, bool]:
    raw = os.path.expandvars(os.path.expanduser(str(model_id)))
    path = Path(raw)
    if path.exists():
        return str(path.resolve()), True
    return raw, False


class LlavaVICRAnalyzer:
    def __init__(
        self,
        model_id: str | Path = "llava-hf/llava-1.5-7b-hf",
        torch_dtype: str = "float16",
        device_map: str = "auto",
        max_memory: Mapping[int | str, str] | None = None,
        local_files_only: bool = False,
        attn_implementation: str | None = None,
        score_backend: str = "paper_multimodal_prompt_core_icr",
    ) -> None:
        dtype = self._resolve_dtype(torch_dtype)
        model_source, inferred_local = _resolve_model_source(model_id)
        if attn_implementation is None:
            attn_implementation = "eager"
        manual_device: str | None = None
        normalized_device_map = device_map
        if isinstance(device_map, str) and device_map.lower() in {"none", "single", "cuda", "cpu"}:
            normalized_device_map = None
            if device_map.lower() == "cpu":
                manual_device = "cpu"
            elif torch.cuda.is_available():
                manual_device = "cuda:0"
            else:
                manual_device = "cpu"
        elif device_map is not None and not is_accelerate_available():
            normalized_device_map = None
            if torch.cuda.is_available():
                manual_device = "cuda:0"
                warnings.warn(
                    "accelerate is not installed; falling back to single-GPU loading on cuda:0.",
                    RuntimeWarning,
                )
            else:
                manual_device = "cpu"
                warnings.warn(
                    "accelerate is not installed and CUDA is unavailable; falling back to CPU loading.",
                    RuntimeWarning,
                )

        load_kwargs: Dict[str, Any] = {
            "dtype": dtype,
        }
        if normalized_device_map is not None:
            load_kwargs["device_map"] = normalized_device_map
            load_kwargs["low_cpu_mem_usage"] = True
        if max_memory:
            load_kwargs["max_memory"] = dict(max_memory)
        if local_files_only or inferred_local:
            load_kwargs["local_files_only"] = True
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_source,
            **load_kwargs,
        )
        if manual_device is not None:
            self.model.to(manual_device)
        self.model.eval()
        processor_kwargs: Dict[str, Any] = {}
        if local_files_only or inferred_local:
            processor_kwargs["local_files_only"] = True
        self.processor = AutoProcessor.from_pretrained(model_source, **processor_kwargs)
        self.tokenizer = self.processor.tokenizer
        self._sync_processor_image_attrs()
        self.score_backend = score_backend

    @staticmethod
    def _resolve_dtype(torch_dtype: str) -> torch.dtype | str:
        name = torch_dtype.lower()
        if name == "auto":
            return "auto"
        if name == "float16":
            return torch.float16
        if name == "bfloat16":
            return torch.bfloat16
        if name == "float32":
            return torch.float32
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")

    def _input_device(self) -> torch.device:
        embeddings = self.model.get_input_embeddings()
        if embeddings is not None and hasattr(embeddings, "weight"):
            return embeddings.weight.device
        return next(self.model.parameters()).device

    def _sync_processor_image_attrs(self) -> None:
        vision_cfg = getattr(self.model.config, "vision_config", None)
        if vision_cfg is None:
            return

        if getattr(self.processor, "patch_size", None) is None and hasattr(vision_cfg, "patch_size"):
            self.processor.patch_size = vision_cfg.patch_size

        if getattr(self.processor, "vision_feature_select_strategy", None) is None:
            strategy = getattr(self.model.config, "vision_feature_select_strategy", None)
            if strategy is not None:
                self.processor.vision_feature_select_strategy = strategy

        if getattr(self.processor, "num_additional_image_tokens", None) is None:
            image_seq_length = getattr(self.model.config, "image_seq_length", None)
            image_size = getattr(vision_cfg, "image_size", None)
            patch_size = getattr(vision_cfg, "patch_size", None)
            if image_seq_length is not None and image_size is not None and patch_size is not None:
                patch_tokens = (image_size // patch_size) ** 2
                self.processor.num_additional_image_tokens = max(image_seq_length - patch_tokens, 0)

    def build_prompt(self, question: str, answer: str | None = None) -> str:
        prefix = f"USER: <image>\n{question.strip()} ASSISTANT:"
        if answer is None:
            return prefix
        clean_answer = answer.strip()
        if not clean_answer:
            return prefix
        return f"{prefix} {clean_answer}"

    def load_image(self, image_path: str | Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def generate_answer(
        self,
        image_path: str | Path,
        question: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        image = self.load_image(image_path)
        prompt = self.build_prompt(question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = _to_device(inputs, self._input_device())

        generate_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "use_cache": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **generate_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        answer_ids = generated[:, prompt_length:]
        answer = self.processor.batch_decode(
            answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return answer.strip()

    def analyze(
        self,
        image_path: str | Path,
        question: str,
        answer: str,
        visual_top_k: int = 10,
        target_mode: str = "all",
        object_phrases: Sequence[str] | None = None,
        object_mentions: Sequence[Dict[str, Any]] | None = None,
        score_backend: str | None = None,
    ) -> Dict[str, Any]:
        image = self.load_image(image_path)
        prefix_text = self.build_prompt(question)
        full_text = self.build_prompt(question, answer)

        prefix_inputs = self.processor(images=image, text=prefix_text, return_tensors="pt")
        full_inputs = self.processor(images=image, text=full_text, return_tensors="pt")

        prefix_ids = prefix_inputs["input_ids"][0]
        full_ids = full_inputs["input_ids"][0]

        full_inputs = _to_device(full_inputs, self._input_device())

        with torch.inference_mode():
            outputs = self.model(
                **full_inputs,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if hidden_states is None or attentions is None:
            raise RuntimeError("Model outputs do not include hidden states and attentions.")

        visual_token_count = self._infer_visual_token_count(outputs)
        image_token_index = self.model.config.image_token_index

        token_layout = self._resolve_token_layout(
            token_ids=full_ids.tolist(),
            prefix_length=prefix_ids.shape[0],
            image_token_index=image_token_index,
            visual_token_count=visual_token_count,
        )

        if len(token_layout["visual_spans"]) != 1:
            raise ValueError("This initial migration only supports a single image span per sample.")

        visual_start, visual_end = token_layout["visual_spans"][0]
        active_backend = score_backend or self.score_backend
        # 论文风格 ICR 的 basis/context 只取 prompt 侧核心区间。
        # 这里故意截断在 prompt 末尾，避免前面已经生成的 answer token
        # 把可能的幻觉上下文继续污染后续 token 的分数。
        prompt_core_start = 0
        prompt_core_end = int(prefix_ids.shape[0])
        token_scores = self._score_answer_tokens(
            hidden_states=hidden_states,
            attentions=attentions,
            answer_token_ids=token_layout["answer_token_ids"],
            answer_token_positions=token_layout["answer_positions_merged"],
            context_start=prompt_core_start,
            context_end=prompt_core_end,
            visual_start=visual_start,
            visual_end=visual_end,
            top_k=visual_top_k,
            score_backend=active_backend,
        )

        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "model_id": self.model.name_or_path,
            "score_backend": active_backend,
            "support_top_k": visual_top_k,
            "target_mode": target_mode,
            "token_layout_mode": token_layout["mode"],
            "visual_span": {"start": visual_start, "end": visual_end, "count": visual_end - visual_start},
            "prompt_core_span": {
                "start": prompt_core_start,
                "end": prompt_core_end,
                "count": prompt_core_end - prompt_core_start,
            },
            "layer_mean_scores": self._aggregate_layer_means(token_scores),
            "token_mean_scores": self._aggregate_token_means(token_scores),
            "sequence_mean_vicr": mean_or_zero([item["mean_vicr"] for item in token_scores]),
            "answer_token_count": len(token_scores),
            "token_scores": token_scores,
        }

        if target_mode == "objects":
            if object_mentions:
                targets, alignment_debug = build_object_targets_from_mentions(
                    tokenizer=self.tokenizer,
                    answer_token_ids=token_layout["answer_token_ids"],
                    answer_token_positions=token_layout["answer_positions_merged"],
                    object_mentions=object_mentions,
                )
                result["object_mentions"] = list(object_mentions)
                result["object_alignment"] = alignment_debug
            else:
                phrases = list(object_phrases or [])
                targets = build_object_targets(
                    tokenizer=self.tokenizer,
                    answer_token_ids=token_layout["answer_token_ids"],
                    answer_token_positions=token_layout["answer_positions_merged"],
                    object_phrases=phrases,
                )
                result["object_phrases"] = phrases
                result["object_alignment"] = []

            object_scores = summarize_targets(token_scores, targets)
            result["object_scores"] = object_scores
            result["object_count"] = len(object_scores)
            result["aligned_object_count"] = len(object_scores)
            result["dropped_unaligned_count"] = max(0, len(result.get("object_alignment", [])) - len(object_scores))
            result["object_mean_vicr"] = mean_or_zero(item["object_global_mean_icr"] for item in object_scores)
            result["object_global_mean_icr"] = result["object_mean_vicr"]

        return result

    def generate_and_analyze(
        self,
        image_path: str | Path,
        question: str,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        visual_top_k: int = 10,
        target_mode: str = "all",
        object_phrases: Sequence[str] | None = None,
        object_mentions: Sequence[Dict[str, Any]] | None = None,
        score_backend: str | None = None,
    ) -> Dict[str, Any]:
        answer = self.generate_answer(
            image_path=image_path,
            question=question,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        result = self.analyze(
            image_path=image_path,
            question=question,
            answer=answer,
            visual_top_k=visual_top_k,
            target_mode=target_mode,
            object_phrases=object_phrases,
            object_mentions=object_mentions,
            score_backend=score_backend,
        )
        result["generation"] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
        return result

    def _infer_visual_token_count(self, outputs: Any) -> int:
        image_hidden_states = getattr(outputs, "image_hidden_states", None)
        if image_hidden_states is not None:
            return int(image_hidden_states.shape[-2])

        image_seq_length = getattr(self.model.config, "image_seq_length", None)
        if image_seq_length is not None:
            return int(image_seq_length)

        vision_cfg = getattr(self.model.config, "vision_config", None)
        if vision_cfg is None:
            raise ValueError("Cannot infer visual token count from the model configuration.")

        image_size = getattr(vision_cfg, "image_size", None)
        patch_size = getattr(vision_cfg, "patch_size", None)
        if image_size is None or patch_size is None:
            raise ValueError("Cannot infer visual token count without image_size and patch_size.")
        return int((image_size // patch_size) ** 2)

    def _resolve_token_layout(
        self,
        token_ids: Sequence[int],
        prefix_length: int,
        image_token_index: int,
        visual_token_count: int,
    ) -> Dict[str, Any]:
        image_positions = [idx for idx, token_id in enumerate(token_ids) if token_id == image_token_index]
        if not image_positions:
            raise ValueError("No <image> token found in the tokenized prompt.")

        answer_token_ids = list(token_ids[prefix_length:])
        answer_positions_tokenized = list(range(prefix_length, len(token_ids)))

        if len(image_positions) == 1:
            image_pos = image_positions[0]
            visual_spans = [(image_pos, image_pos + visual_token_count)]
            answer_positions_merged = [
                pos if pos < image_pos else pos + visual_token_count - 1 for pos in answer_positions_tokenized
            ]
            return {
                "mode": "legacy-single-image-token",
                "visual_spans": visual_spans,
                "answer_token_ids": answer_token_ids,
                "answer_positions_merged": answer_positions_merged,
            }

        if len(image_positions) % visual_token_count != 0:
            raise ValueError(
                "The number of <image> tokens is neither 1 nor a multiple of the visual token count."
            )

        image_groups = _group_contiguous(image_positions)
        if any(len(group) != visual_token_count for group in image_groups):
            raise ValueError("Expanded image tokens are not grouped into contiguous visual spans.")

        return {
            "mode": "expanded-image-placeholders",
            "visual_spans": [(group[0], group[-1] + 1) for group in image_groups],
            "answer_token_ids": answer_token_ids,
            "answer_positions_merged": answer_positions_tokenized,
        }

    def _score_answer_tokens(
        self,
        hidden_states: Sequence[torch.Tensor],
        attentions: Sequence[torch.Tensor],
        answer_token_ids: Sequence[int],
        answer_token_positions: Sequence[int],
        context_start: int,
        context_end: int,
        visual_start: int,
        visual_end: int,
        top_k: int,
        score_backend: str,
    ) -> List[Dict[str, Any]]:
        if score_backend == "paper_multimodal_prompt_core_icr":
            return self._score_answer_tokens_prompt_core(
                hidden_states=hidden_states,
                attentions=attentions,
                answer_token_ids=answer_token_ids,
                answer_token_positions=answer_token_positions,
                context_start=context_start,
                context_end=context_end,
                top_k=top_k,
            )
        if score_backend == "visual_only_icr":
            return self._score_answer_tokens_visual(
                hidden_states=hidden_states,
                attentions=attentions,
                answer_token_ids=answer_token_ids,
                answer_token_positions=answer_token_positions,
                visual_start=visual_start,
                visual_end=visual_end,
                top_k=top_k,
            )
        raise ValueError(f"Unsupported score backend: {score_backend}")

    def _score_answer_tokens_prompt_core(
        self,
        hidden_states: Sequence[torch.Tensor],
        attentions: Sequence[torch.Tensor],
        answer_token_ids: Sequence[int],
        answer_token_positions: Sequence[int],
        context_start: int,
        context_end: int,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        context_slice = slice(context_start, context_end)

        for answer_token_index, (token_id, token_position) in enumerate(zip(answer_token_ids, answer_token_positions)):
            token_text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            layer_scores: List[VICTokenLayerScore] = []
            for layer_index in range(1, len(hidden_states)):
                prev_state = hidden_states[layer_index - 1][0, token_position, :]
                curr_state = hidden_states[layer_index][0, token_position, :]
                update = curr_state - prev_state

                # 论文式迁移做法：先用该 token 的注意力在 prompt-side 里选 support，
                # 再在同一组位置上比较 hidden update 的投影分布。
                basis_states = hidden_states[layer_index - 1][0, context_slice, :]
                proj_values = projection_scores(update, basis_states)

                attn_weights = attentions[layer_index - 1][0, :, token_position, context_slice].mean(dim=0).float()
                attn_top_values, top_indices = topk_values(attn_weights, top_k)
                proj_top_values = proj_values[top_indices]

                score = float(jsd_standardized(proj_top_values, attn_top_values).item())
                attn_dist = renormalize(attn_weights)
                top_mass = float(attn_dist[top_indices].sum().item())
                support_positions = (top_indices + context_start).tolist()

                layer_scores.append(
                    VICTokenLayerScore(
                        layer=layer_index,
                        merged_token_index=token_position,
                        token=token_text,
                        vicr=score,
                        context_top_indices=support_positions,
                        attention_top_mass=top_mass,
                    )
                )

            results.append(
                {
                    "answer_token_index": answer_token_index,
                    "token_id": int(token_id),
                    "token": token_text,
                    "merged_token_index": int(token_position),
                    "mean_vicr": mean_or_zero(score.vicr for score in layer_scores),
                    "max_vicr": max((score.vicr for score in layer_scores), default=0.0),
                    "layer_scores": [asdict(score) for score in layer_scores],
                }
            )

        return results

    def _score_answer_tokens_visual(
        self,
        hidden_states: Sequence[torch.Tensor],
        attentions: Sequence[torch.Tensor],
        answer_token_ids: Sequence[int],
        answer_token_positions: Sequence[int],
        visual_start: int,
        visual_end: int,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        visual_slice = slice(visual_start, visual_end)

        for answer_token_index, (token_id, token_position) in enumerate(zip(answer_token_ids, answer_token_positions)):
            token_text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            layer_scores: List[VICTokenLayerScore] = []
            for layer_index in range(1, len(hidden_states)):
                prev_state = hidden_states[layer_index - 1][0, token_position, :]
                curr_state = hidden_states[layer_index][0, token_position, :]
                update = curr_state - prev_state

                visual_states = hidden_states[layer_index][0, visual_slice, :]
                proj_dist = projection_distribution(update, visual_states)

                attn_weights = attentions[layer_index - 1][0, :, token_position, visual_slice].mean(dim=0)
                attn_dist = renormalize(attn_weights)

                attn_masked, top_indices = masked_topk(attn_dist, top_k)
                proj_masked = torch.zeros_like(proj_dist)
                proj_masked[top_indices] = proj_dist[top_indices]
                proj_masked = renormalize(proj_masked)

                score = float(jsd(proj_masked, attn_masked).item())
                top_mass = float(attn_dist[top_indices].sum().item())
                support_positions = (torch.tensor(top_indices, device=attn_dist.device) + visual_start).tolist()

                layer_scores.append(
                    VICTokenLayerScore(
                        layer=layer_index,
                        merged_token_index=token_position,
                        token=token_text,
                        vicr=score,
                        context_top_indices=support_positions,
                        attention_top_mass=top_mass,
                    )
                )

            results.append(
                {
                    "answer_token_index": answer_token_index,
                    "token_id": int(token_id),
                    "token": token_text,
                    "merged_token_index": int(token_position),
                    "mean_vicr": mean_or_zero(score.vicr for score in layer_scores),
                    "max_vicr": max((score.vicr for score in layer_scores), default=0.0),
                    "layer_scores": [asdict(score) for score in layer_scores],
                }
            )

        return results

    def _aggregate_layer_means(self, token_scores: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        layer_to_scores: Dict[int, List[float]] = {}
        for token in token_scores:
            for layer_score in token["layer_scores"]:
                layer = int(layer_score["layer"])
                layer_to_scores.setdefault(layer, []).append(float(layer_score["vicr"]))

        return [
            {"layer": layer, "mean_vicr": mean_or_zero(scores)}
            for layer, scores in sorted(layer_to_scores.items())
        ]

    def _aggregate_token_means(self, token_scores: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for item in token_scores:
            summary.append(
                {
                    "answer_token_index": item["answer_token_index"],
                    "token": item["token"],
                    "token_id": item["token_id"],
                    "merged_token_index": item["merged_token_index"],
                    "mean_vicr": item["mean_vicr"],
                    "max_vicr": item["max_vicr"],
                }
            )
        return summary

    @staticmethod
    def save_json(result: Dict[str, Any], output_path: str | Path) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
