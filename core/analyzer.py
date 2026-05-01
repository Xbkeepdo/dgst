from __future__ import annotations

# DGST 在 LLaVA 上的对象级分析器。

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

from .llava import LlavaVICRAnalyzer, _to_device
from .scoring import mean_or_zero, renormalize
from .targets import (
    TargetSpan,
    build_object_targets,
    build_object_targets_from_mentions,
    build_sentence_last_token_targets,
)


@dataclass
class DGSTTokenLayerScore:
    layer: int
    risk: float
    source_entropy: float
    target_entropy: float
    distance_mean: float
    source_semantic_penalty_mean: float
    target_semantic_penalty_mean: float
    prompt_last_cosine: float
    prompt_mean_cosine: float
    source_support_size: int
    target_support_size: int
    union_size: int


def _topk_union_indices(source: torch.Tensor, target: torch.Tensor, top_k: int) -> torch.Tensor:
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError("Source and target distributions must be 1D tensors.")
    k_half = min(max(int(top_k // 2), 1), source.numel())
    src_idx = torch.topk(source, k=k_half).indices
    tgt_idx = torch.topk(target, k=k_half).indices
    union = torch.unique(torch.cat([src_idx, tgt_idx], dim=0), sorted=True)
    return union


def _semantic_cost_matrix(
    visual_states: torch.Tensor,
    semantic_scores: torch.Tensor,
    gamma1: float,
    gamma2: float,
) -> torch.Tensor:
    normalized = F.normalize(visual_states.float(), dim=-1)
    cosine = torch.matmul(normalized, normalized.transpose(0, 1))
    distance = (1.0 - cosine).clamp_min(0.0)
    target_penalty = torch.relu(1.0 - semantic_scores).unsqueeze(0)
    source_penalty = torch.relu(1.0 - semantic_scores).unsqueeze(1)
    penalty = 1.0 + float(gamma1) * target_penalty + float(gamma2) * source_penalty
    return distance * penalty


def _distribution_entropy(distribution: torch.Tensor) -> float:
    safe = distribution.clamp_min(1e-12)
    return float((-(safe * torch.log(safe))).sum().item())


def _wasserstein_1_exact(
    source: torch.Tensor,
    target: torch.Tensor,
    cost: torch.Tensor,
    solver: str,
) -> tuple[float, torch.Tensor]:
    solver_name = str(solver).strip().lower()
    if solver_name != "linprog":
        raise ValueError(f"Unsupported OT solver: {solver}")
    try:
        from scipy.optimize import linprog
    except Exception as exc:
        raise ImportError("DGST exact OT requires scipy. Install scipy>=1.11.0.") from exc

    source = source.detach().cpu().double()
    target = target.detach().cpu().double()
    cost = cost.detach().cpu().double()

    source = torch.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    cost = torch.nan_to_num(cost, nan=1e6, posinf=1e6, neginf=1e6).clamp_min(0.0)

    source_sum = float(source.sum().item())
    target_sum = float(target.sum().item())
    if source_sum <= 0.0:
        source = torch.full_like(source, 1.0 / max(source.numel(), 1))
    else:
        source = source / source_sum
    if target_sum <= 0.0:
        target = torch.full_like(target, 1.0 / max(target.numel(), 1))
    else:
        target = target / target_sum

    source = source / source.sum().clamp_min(1e-12)
    target = target / target.sum().clamp_min(1e-12)
    mass_gap = float(source.sum().item() - target.sum().item())
    if abs(mass_gap) > 1e-12:
        target[-1] = max(0.0, float(target[-1].item() + mass_gap))
        target = target / target.sum().clamp_min(1e-12)

    n = int(source.numel())
    c = cost.reshape(-1).numpy()

    row_constraints = []
    row_targets = []
    for row_idx in range(n):
        row = torch.zeros((n, n), dtype=torch.float64)
        row[row_idx, :] = 1.0
        row_constraints.append(row.reshape(-1).numpy())
        row_targets.append(float(source[row_idx].item()))

    col_constraints = []
    col_targets = []
    for col_idx in range(n):
        col = torch.zeros((n, n), dtype=torch.float64)
        col[:, col_idx] = 1.0
        col_constraints.append(col.reshape(-1).numpy())
        col_targets.append(float(target[col_idx].item()))

    def _solve(a_eq_rows, b_eq_values):
        return linprog(
            c=c,
            A_eq=a_eq_rows,
            b_eq=b_eq_values,
            bounds=(0.0, None),
            method="highs",
        )

    result = _solve(row_constraints + col_constraints, row_targets + col_targets)
    if not result.success:
        # The transport polytope only needs 2n-1 independent equality constraints.
        # Dropping the last column-sum equation avoids spurious infeasible reports
        # when source/target masses differ by tiny floating-point noise.
        reduced_constraints = row_constraints + col_constraints[:-1]
        reduced_targets = row_targets + col_targets[:-1]
        result = _solve(reduced_constraints, reduced_targets)
    if not result.success:
        raise RuntimeError(f"Exact OT solver failed: {result.message}")
    plan = torch.tensor(result.x, dtype=torch.float64).reshape(n, n)
    return float(result.fun), plan


def summarize_dgst_targets(
    token_scores: Sequence[Dict[str, Any]],
    targets: Sequence[TargetSpan],
    target_aggregation: str = "mean",
) -> List[Dict[str, Any]]:
    aggregation = str(target_aggregation).strip().lower()
    if aggregation not in {"mean", "last", "max"}:
        raise ValueError(f"Unsupported target aggregation: {target_aggregation!r}")

    def aggregate(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        if aggregation == "max":
            return float(max(values))
        return mean_or_zero(values)

    score_map = {int(item["answer_token_index"]): item for item in token_scores}
    results: List[Dict[str, Any]] = []

    for target in targets:
        selected = [score_map[index] for index in range(target.answer_token_start, target.answer_token_end) if index in score_map]
        if not selected:
            continue
        full_selected = list(selected)
        if aggregation == "last":
            selected = [max(selected, key=lambda item: int(item["answer_token_index"]))]

        layer_to_values: Dict[int, List[float]] = {}
        for token_score in selected:
            for layer_score in token_score["dgst_layer_scores"]:
                layer = int(layer_score["layer"])
                layer_to_values.setdefault(layer, []).append(float(layer_score["risk"]))

        layer_summary = [
            {"layer": layer, "risk": aggregate(values)}
            for layer, values in sorted(layer_to_values.items())
        ]

        probe6_layer_to_values: Dict[int, Dict[str, List[float]]] = {}
        for token_score in selected:
            for layer_score in token_score["dgst_layer_scores"]:
                layer = int(layer_score["layer"])
                slot = probe6_layer_to_values.setdefault(
                    layer,
                    {
                        "risk": [],
                        "source_entropy": [],
                        "target_entropy": [],
                        "distance_mean": [],
                        "source_semantic_penalty_mean": [],
                        "target_semantic_penalty_mean": [],
                    },
                )
                for key in slot:
                    slot[key].append(float(layer_score[key]))

        probe6_layer_summary = []
        probe6_flat: List[float] = []
        probe6_feature_names = [
            "risk",
            "source_entropy",
            "target_entropy",
            "distance_mean",
            "source_semantic_penalty_mean",
            "target_semantic_penalty_mean",
        ]
        for layer, values in sorted(probe6_layer_to_values.items()):
            entry = {"layer": layer}
            for key in probe6_feature_names:
                entry[key] = aggregate(values[key])
                probe6_flat.append(float(entry[key]))
            probe6_layer_summary.append(entry)

        prompt_layer_to_values: Dict[int, Dict[str, List[float]]] = {}
        for token_score in selected:
            for layer_score in token_score["dgst_layer_scores"]:
                layer = int(layer_score["layer"])
                slot = prompt_layer_to_values.setdefault(
                    layer,
                    {
                        "prompt_last_cosine": [],
                        "prompt_mean_cosine": [],
                    },
                )
                for key in slot:
                    slot[key].append(float(layer_score.get(key, 0.0)))

        prompt_layer_summary = []
        prompt_flat: List[float] = []
        prompt_feature_names = ["prompt_last_cosine", "prompt_mean_cosine"]
        for layer, values in sorted(prompt_layer_to_values.items()):
            entry = {"layer": layer}
            for key in prompt_feature_names:
                entry[key] = aggregate(values[key])
                prompt_flat.append(float(entry[key]))
            prompt_layer_summary.append(entry)

        results.append(
            {
                "kind": target.kind,
                "phrase": target.phrase,
                "surface": target.surface,
                "source_surface_word": target.source_surface,
                "canonical_name": target.canonical_name,
                "mention_index": target.mention_index,
                "word_index": target.word_index,
                "hallucinated": int(target.hallucinated or 0),
                "alignment_status": "aligned",
                "alignment_strategy": target.alignment_strategy,
                "answer_token_span": [target.answer_token_start, target.answer_token_end],
                "merged_token_positions": target.merged_token_positions,
                "target_aggregation": aggregation,
                "subtokens": [item["token"] for item in selected],
                "full_subtokens": [item["token"] for item in full_selected],
                "subtoken_scores": selected,
                "full_subtoken_scores": full_selected,
                "layer_ids": [int(item["layer"]) for item in layer_summary],
                "object_layer_dgst_risk": [float(item["risk"]) for item in layer_summary],
                "dgst_layer_scores": layer_summary,
                "dgst_probe6_feature_names": probe6_feature_names,
                "dgst_probe6_layer_stats": probe6_layer_summary,
                "object_layer_dgst_probe6": probe6_flat,
                "dgst_prompt_feature_names": prompt_feature_names,
                "dgst_prompt_layer_stats": prompt_layer_summary,
                "object_layer_prompt_token_cos": prompt_flat,
                "dgst_baseline_mean": aggregate([float(item["dgst_baseline_mean"]) for item in selected]),
                "dgst_baseline_std": aggregate([float(item["dgst_baseline_std"]) for item in selected]),
                "dgst_final_score": aggregate([float(item["dgst_final_score"]) for item in selected]),
                "token_aligned": True,
            }
        )

    return results


class LlavaDGSTAnalyzer(LlavaVICRAnalyzer):
    def _decoder_layers(self):
        if hasattr(self.model, "language_model") and hasattr(self.model.language_model, "model"):
            model = getattr(self.model.language_model, "model")
            if hasattr(model, "layers"):
                return model.layers
        decoder = self.model.get_decoder()
        if hasattr(decoder, "layers"):
            return decoder.layers
        raise ValueError("Unsupported model: cannot resolve decoder layers for DGST.")

    def _output_embedding_layer(self):
        layer = self.model.get_output_embeddings()
        if layer is not None:
            return layer
        if hasattr(self.model, "language_model"):
            layer = self.model.language_model.get_output_embeddings()
            if layer is not None:
                return layer
        raise ValueError("Model does not expose output embeddings.")

    def _forward_with_layer_captures(self, full_inputs: Dict[str, Any]):
        layers = self._decoder_layers()
        captures: list[dict[str, Any]] = [
            {
                "h_prev": None,
                "o_attn": None,
                "attn_weights": None,
                "o_ffn": None,
            }
            for _ in range(len(layers))
        ]
        handles = []

        def _layer_pre_hook(index: int):
            def hook(_module, args):
                captures[index]["h_prev"] = args[0].detach()

            return hook

        def _self_attn_hook(index: int):
            def hook(_module, _args, output):
                if isinstance(output, tuple):
                    captures[index]["o_attn"] = output[0].detach()
                    captures[index]["attn_weights"] = output[1].detach() if len(output) > 1 and output[1] is not None else None
                else:
                    captures[index]["o_attn"] = output.detach()

            return hook

        def _mlp_hook(index: int):
            def hook(_module, _args, output):
                captures[index]["o_ffn"] = output.detach()

            return hook

        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(_layer_pre_hook(index)))
            handles.append(layer.self_attn.register_forward_hook(_self_attn_hook(index)))
            handles.append(layer.mlp.register_forward_hook(_mlp_hook(index)))

        try:
            with torch.inference_mode():
                outputs = self.model(
                    **full_inputs,
                    output_attentions=True,
                    return_dict=True,
                    use_cache=False,
                )
        finally:
            for handle in handles:
                handle.remove()

        for capture in captures:
            if capture["h_prev"] is None or capture["o_attn"] is None or capture["o_ffn"] is None:
                raise RuntimeError("DGST forward hooks did not capture all required intermediate states.")
            if capture["attn_weights"] is None:
                raise RuntimeError("DGST requires attention weights; ensure eager attention is enabled.")
            target_device = capture["o_attn"].device
            if capture["h_prev"].device != target_device:
                capture["h_prev"] = capture["h_prev"].to(target_device)
            if capture["o_ffn"].device != target_device:
                capture["o_ffn"] = capture["o_ffn"].to(target_device)
            if capture["attn_weights"].device != target_device:
                capture["attn_weights"] = capture["attn_weights"].to(target_device)
            capture["h_mid"] = capture["h_prev"] + capture["o_attn"]
        return outputs, captures

    def _build_semantic_probabilities(
        self,
        captures: Sequence[Dict[str, Any]],
        visual_start: int,
        visual_end: int,
        target_token_ids: Sequence[int],
    ) -> List[torch.Tensor]:
        if not target_token_ids:
            return []
        output_layer = self._output_embedding_layer()
        weight = output_layer.weight.float()
        bias = getattr(output_layer, "bias", None)
        bias = bias.float() if bias is not None else None
        token_index = torch.tensor(list(target_token_ids), device=weight.device, dtype=torch.long)
        layer_probs: List[torch.Tensor] = []

        for capture in captures:
            visual_states = capture["h_mid"][0, visual_start:visual_end, :].float().to(weight.device)
            logits = F.linear(visual_states, weight, bias)
            probs = torch.softmax(logits, dim=-1)
            selected = probs.index_select(dim=1, index=token_index).detach().cpu()
            layer_probs.append(selected)
        return layer_probs

    def _resolve_prompt_token_positions(
        self,
        *,
        token_ids: Sequence[int],
        prefix_length: int,
        image_token_index: int,
        visual_token_count: int,
    ) -> List[int]:
        prompt_positions: List[int] = []
        image_positions = {idx for idx, token_id in enumerate(token_ids) if token_id == image_token_index}
        if not image_positions:
            raise ValueError("No <image> token found in the tokenized prompt.")

        if len(image_positions) == 1:
            image_pos = next(iter(image_positions))
            for pos in range(int(prefix_length)):
                if pos == image_pos:
                    continue
                merged_pos = pos if pos < image_pos else pos + int(visual_token_count) - 1
                prompt_positions.append(int(merged_pos))
            return prompt_positions

        for pos in range(int(prefix_length)):
            if pos in image_positions:
                continue
            prompt_positions.append(int(pos))
        return prompt_positions

    def _score_requested_tokens(
        self,
        *,
        captures: Sequence[Dict[str, Any]],
        answer_token_ids: Sequence[int],
        answer_token_positions: Sequence[int],
        requested_indices: Sequence[int],
        prompt_token_positions: Sequence[int],
        visual_start: int,
        visual_end: int,
        semantic_probs_by_layer: Sequence[torch.Tensor],
        tau: float,
        transport_top_k: int,
        gamma1: float,
        gamma2: float,
        baseline_layers: int,
        risk_start_layer: int,
        alpha: float,
        ot_solver: str,
    ) -> List[Dict[str, Any]]:
        if not requested_indices:
            return []
        if not prompt_token_positions:
            raise ValueError("Prompt-aware DGST features require at least one prompt text token.")
        answer_index_to_col = {int(index): offset for offset, index in enumerate(requested_indices)}
        prompt_positions = [int(position) for position in prompt_token_positions]
        prompt_last_position = int(prompt_positions[-1])
        token_scores: List[Dict[str, Any]] = []

        for answer_token_index in requested_indices:
            token_id = int(answer_token_ids[answer_token_index])
            token_position = int(answer_token_positions[answer_token_index])
            token_text = self.tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            layer_scores: List[DGSTTokenLayerScore] = []
            layer_risks: List[float] = []
            for layer_offset, capture in enumerate(captures):
                layer_index = layer_offset + 1
                layer_hidden = (capture["h_mid"] + capture["o_ffn"])[0].float()
                visual_states = capture["h_mid"][0, visual_start:visual_end, :].float()
                update = capture["o_ffn"][0, token_position, :].float().to(visual_states.device)
                semantic_probs = semantic_probs_by_layer[layer_offset][:, answer_index_to_col[answer_token_index]].to(
                    visual_states.device
                ).float()
                token_state = layer_hidden[token_position, :].to(visual_states.device)
                prompt_last_state = layer_hidden[prompt_last_position, :].to(visual_states.device)
                prompt_mean_state = layer_hidden.index_select(
                    0,
                    torch.tensor(prompt_positions, device=layer_hidden.device, dtype=torch.long),
                ).mean(dim=0).to(visual_states.device)
                prompt_last_cosine = float(
                    F.cosine_similarity(token_state.unsqueeze(0), prompt_last_state.unsqueeze(0), dim=-1).item()
                )
                prompt_mean_cosine = float(
                    F.cosine_similarity(token_state.unsqueeze(0), prompt_mean_state.unsqueeze(0), dim=-1).item()
                )

                source_sims = F.cosine_similarity(update.unsqueeze(0), visual_states, dim=-1)
                source_dist = torch.softmax(source_sims / max(float(tau), 1e-6), dim=-1)

                attn_weights = (
                    capture["attn_weights"][0, :, token_position, visual_start:visual_end]
                    .mean(dim=0)
                    .float()
                    .to(visual_states.device)
                )
                attn_dist = renormalize(attn_weights)
                target_dist = renormalize(attn_dist * semantic_probs)

                support = _topk_union_indices(source_dist, target_dist, transport_top_k)
                local_source = renormalize(source_dist.index_select(0, support))
                local_target = renormalize(target_dist.index_select(0, support))
                local_states = visual_states.index_select(0, support)
                local_semantic = semantic_probs.index_select(0, support)
                normalized_states = F.normalize(local_states.float(), dim=-1)
                cosine = torch.matmul(normalized_states, normalized_states.transpose(0, 1))
                distance = (1.0 - cosine).clamp_min(0.0)
                source_semantic_penalty = torch.relu(1.0 - local_semantic)
                target_semantic_penalty = torch.relu(1.0 - local_semantic)
                cost = _semantic_cost_matrix(local_states, local_semantic, gamma1=gamma1, gamma2=gamma2)
                risk, transport_plan = _wasserstein_1_exact(local_source, local_target, cost, solver=ot_solver)
                source_entropy = _distribution_entropy(local_source)
                target_entropy = _distribution_entropy(local_target)
                distance_mean = float((transport_plan.to(distance.device) * distance).sum().item())
                source_penalty_mean = float(
                    (transport_plan.to(source_semantic_penalty.device) * source_semantic_penalty.unsqueeze(1)).sum().item()
                )
                target_penalty_mean = float(
                    (transport_plan.to(target_semantic_penalty.device) * target_semantic_penalty.unsqueeze(0)).sum().item()
                )

                layer_risks.append(float(risk))
                layer_scores.append(
                    DGSTTokenLayerScore(
                        layer=layer_index,
                        risk=float(risk),
                        source_entropy=float(source_entropy),
                        target_entropy=float(target_entropy),
                        distance_mean=float(distance_mean),
                        source_semantic_penalty_mean=float(source_penalty_mean),
                        target_semantic_penalty_mean=float(target_penalty_mean),
                        prompt_last_cosine=float(prompt_last_cosine),
                        prompt_mean_cosine=float(prompt_mean_cosine),
                        source_support_size=min(max(int(transport_top_k // 2), 1), int(source_dist.numel())),
                        target_support_size=min(max(int(transport_top_k // 2), 1), int(target_dist.numel())),
                        union_size=int(support.numel()),
                    )
                )

            base_width = max(1, min(int(baseline_layers), len(layer_risks)))
            baseline_values = layer_risks[:base_width]
            baseline_mean = mean_or_zero(baseline_values)
            baseline_std = math.sqrt(sum((value - baseline_mean) ** 2 for value in baseline_values) / len(baseline_values))
            threshold = baseline_mean + float(alpha) * baseline_std
            final_score = sum(
                max(0.0, value - threshold)
                for layer_number, value in enumerate(layer_risks, start=1)
                if layer_number >= int(risk_start_layer)
            )

            token_scores.append(
                {
                    "answer_token_index": int(answer_token_index),
                    "token_id": token_id,
                    "token": token_text,
                    "merged_token_index": token_position,
                    "dgst_baseline_mean": float(baseline_mean),
                    "dgst_baseline_std": float(baseline_std),
                    "dgst_final_score": float(final_score),
                    "dgst_layer_scores": [asdict(score) for score in layer_scores],
                }
            )

        token_scores.sort(key=lambda item: int(item["answer_token_index"]))
        return token_scores

    def _generation_step_inputs(self, prompt_inputs: Dict[str, Any], token_ids: Sequence[int]) -> Dict[str, Any]:
        input_ids = torch.tensor(
            [list(int(token_id) for token_id in token_ids)],
            dtype=prompt_inputs["input_ids"].dtype,
        )
        step_inputs: Dict[str, Any] = {}
        for key, value in prompt_inputs.items():
            if key == "input_ids":
                step_inputs[key] = input_ids
            elif key == "attention_mask":
                step_inputs[key] = torch.ones_like(input_ids)
            elif isinstance(value, torch.Tensor):
                step_inputs[key] = value.clone()
            else:
                step_inputs[key] = value
        if "attention_mask" not in step_inputs:
            step_inputs["attention_mask"] = torch.ones_like(input_ids)
        return step_inputs

    def _select_next_token_id(
        self,
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> int:
        logits = logits.float()
        if not do_sample or float(temperature) <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())

        scaled = logits / max(float(temperature), 1e-6)
        probs = torch.softmax(scaled, dim=-1)
        if 0.0 < float(top_p) < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove_mask = cumulative > float(top_p)
            remove_mask[0] = False
            sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum().clamp_min(1e-12)
            sampled = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_indices[sampled].item())
        return int(torch.multinomial(probs, num_samples=1).item())

    def _eos_token_ids(self) -> set[int]:
        eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            return set()
        if isinstance(eos, (list, tuple, set)):
            return {int(value) for value in eos if value is not None}
        return {int(eos)}

    def generate_answer_with_ids(
        self,
        image_path: str | Path,
        question: str,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Dict[str, Any]:
        image = self.load_image(image_path)
        prompt = self.build_prompt(question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        prefix_ids = [int(token_id) for token_id in inputs["input_ids"][0].tolist()]
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
        answer_ids_tensor = generated[:, prompt_length:]
        eos_token_ids = self._eos_token_ids()
        answer_token_ids: List[int] = []
        for token_id in answer_ids_tensor[0].detach().cpu().tolist():
            token_id = int(token_id)
            if token_id in eos_token_ids:
                break
            answer_token_ids.append(token_id)
        answer = self.processor.batch_decode(
            torch.tensor([answer_token_ids], dtype=torch.long),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip() if answer_token_ids else ""
        return {
            "answer": answer,
            "answer_token_ids": answer_token_ids,
            "prefix_token_ids": prefix_ids,
            "generation": {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "token_count": len(answer_token_ids),
            },
        }

    def _score_generation_time_requested_tokens(
        self,
        image_path: str | Path,
        question: str,
        *,
        answer_token_ids: Sequence[int],
        requested_indices: Sequence[int],
        tau: float,
        transport_top_k: int,
        gamma1: float,
        gamma2: float,
        baseline_layers: int,
        risk_start_layer: int,
        alpha: float,
        ot_solver: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not requested_indices:
            return [], {}

        image = self.load_image(image_path)
        prefix_text = self.build_prompt(question)
        prompt_inputs = self.processor(images=image, text=prefix_text, return_tensors="pt")
        prefix_ids = [int(token_id) for token_id in prompt_inputs["input_ids"][0].tolist()]
        token_scores: List[Dict[str, Any]] = []
        last_layout: Dict[str, Any] = {}

        for answer_index in sorted({int(index) for index in requested_indices}):
            if answer_index < 0 or answer_index >= len(answer_token_ids):
                continue
            context_token_ids = prefix_ids + [int(token_id) for token_id in answer_token_ids[:answer_index]]
            step_inputs = self._generation_step_inputs(prompt_inputs, context_token_ids)
            step_inputs = _to_device(step_inputs, self._input_device())
            outputs, captures = self._forward_with_layer_captures(step_inputs)

            visual_token_count = self._infer_visual_token_count(outputs)
            image_token_index = self.model.config.image_token_index
            token_layout = self._resolve_token_layout(
                token_ids=context_token_ids,
                prefix_length=len(prefix_ids),
                image_token_index=image_token_index,
                visual_token_count=visual_token_count,
            )
            if len(token_layout["visual_spans"]) != 1:
                raise ValueError("DGST currently supports a single image span per sample.")
            visual_start, visual_end = token_layout["visual_spans"][0]
            prompt_token_positions = self._resolve_prompt_token_positions(
                token_ids=context_token_ids,
                prefix_length=len(prefix_ids),
                image_token_index=image_token_index,
                visual_token_count=visual_token_count,
            )
            if answer_index > 0 and token_layout["answer_positions_merged"]:
                query_position = int(token_layout["answer_positions_merged"][-1])
            else:
                query_position = int(prompt_token_positions[-1])

            target_token_id = int(answer_token_ids[answer_index])
            semantic_probs_by_layer = self._build_semantic_probabilities(
                captures=captures,
                visual_start=int(visual_start),
                visual_end=int(visual_end),
                target_token_ids=[target_token_id],
            )
            step_scores = self._score_requested_tokens(
                captures=captures,
                answer_token_ids=[target_token_id],
                answer_token_positions=[query_position],
                requested_indices=[0],
                prompt_token_positions=prompt_token_positions,
                visual_start=int(visual_start),
                visual_end=int(visual_end),
                semantic_probs_by_layer=semantic_probs_by_layer,
                tau=tau,
                transport_top_k=transport_top_k,
                gamma1=gamma1,
                gamma2=gamma2,
                baseline_layers=baseline_layers,
                risk_start_layer=risk_start_layer,
                alpha=alpha,
                ot_solver=ot_solver,
            )
            if step_scores:
                score_item = dict(step_scores[0])
                score_item["answer_token_index"] = int(answer_index)
                score_item["generation_step"] = int(answer_index)
                score_item["generation_time_scoring"] = "sentence_final_next_token_only"
                score_item["generation_query_merged_token_index"] = int(query_position)
                token_scores.append(score_item)

            last_layout = {
                "token_layout_mode": str(token_layout["mode"]),
                "visual_span": {"start": int(visual_start), "end": int(visual_end), "count": int(visual_end - visual_start)},
                "prompt_text_span": {
                    "positions": prompt_token_positions,
                    "count": len(prompt_token_positions),
                    "last_position": prompt_token_positions[-1] if prompt_token_positions else None,
                },
            }
            del outputs, captures

        token_scores.sort(key=lambda item: int(item["answer_token_index"]))
        return token_scores, last_layout

    def analyze_generation_time_sentence_last(
        self,
        image_path: str | Path,
        question: str,
        answer: str,
        answer_token_ids: Sequence[int],
        sentence_mentions: Sequence[Dict[str, Any]],
        *,
        generation: Dict[str, Any] | None = None,
        tau: float = 0.07,
        transport_top_k: int = 32,
        gamma1: float = 1.0,
        gamma2: float = 1.0,
        baseline_layers: int = 10,
        risk_start_layer: int = 15,
        alpha: float = 2.0,
        ot_solver: str = "linprog",
    ) -> Dict[str, Any]:
        answer_token_ids = [int(token_id) for token_id in answer_token_ids]
        answer_token_positions = list(range(len(answer_token_ids)))
        targets, alignment_debug = build_sentence_last_token_targets(
            tokenizer=self.tokenizer,
            answer_token_ids=answer_token_ids,
            answer_token_positions=answer_token_positions,
            sentence_mentions=list(sentence_mentions or []),
            answer_text=answer,
        )
        requested_indices = sorted({int(target.answer_token_start) for target in targets})
        token_scores, layout = self._score_generation_time_requested_tokens(
            image_path=image_path,
            question=question,
            answer_token_ids=answer_token_ids,
            requested_indices=requested_indices,
            tau=tau,
            transport_top_k=transport_top_k,
            gamma1=gamma1,
            gamma2=gamma2,
            baseline_layers=baseline_layers,
            risk_start_layer=risk_start_layer,
            alpha=alpha,
            ot_solver=ot_solver,
        )

        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "model_id": self.model.name_or_path,
            "target_mode": "sentence_last",
            "feature_timing": "generation_time",
            "generation_time_scoring": "sentence_final_next_token_only",
            "token_layout_mode": layout.get("token_layout_mode"),
            "visual_span": layout.get("visual_span", {"start": 0, "end": 0, "count": 0}),
            "prompt_text_span": layout.get("prompt_text_span", {"positions": [], "count": 0, "last_position": None}),
            "answer_token_ids": answer_token_ids,
            "answer_token_positions": answer_token_positions,
            "generation": generation or {},
            "dgst_config": {
                "tau": float(tau),
                "transport_top_k": int(transport_top_k),
                "gamma1": float(gamma1),
                "gamma2": float(gamma2),
                "baseline_layers": int(baseline_layers),
                "risk_start_layer": int(risk_start_layer),
                "alpha": float(alpha),
                "ot_solver": ot_solver,
                "target_aggregation": "sentence_last_generation_time",
            },
            "sentence_mentions": list(sentence_mentions or []),
            "sentence_alignment": alignment_debug,
            "token_scores": token_scores,
        }

        sentence_scores = summarize_dgst_targets(token_scores, targets, target_aggregation="last")
        sentence_entries = {
            int(entry.get("sentence_index", entry.get("mention_index", index))): entry
            for index, entry in enumerate(sentence_mentions or [])
        }
        for item in sentence_scores:
            sentence_index = int(item.get("mention_index", 0))
            entry = sentence_entries.get(sentence_index, {})
            item["sentence_index"] = sentence_index
            item["feature_timing"] = "generation_time"
            item["generation_time_scoring"] = "sentence_final_next_token_only"
            for key, value in entry.items():
                if str(key).startswith("sentence_"):
                    item[key] = value

        result["dgst_sentence_scores"] = sentence_scores
        result["sentence_count"] = len(sentence_scores)
        result["aligned_sentence_count"] = len(sentence_scores)
        result["dropped_unaligned_sentence_count"] = max(0, len(alignment_debug) - len(sentence_scores))
        result["dgst_final_score_mean"] = mean_or_zero(item["dgst_final_score"] for item in sentence_scores)
        return result

    def generate_with_generation_time_dgst(
        self,
        image_path: str | Path,
        question: str,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tau: float = 0.07,
        transport_top_k: int = 32,
        gamma1: float = 1.0,
        gamma2: float = 1.0,
        baseline_layers: int = 10,
        risk_start_layer: int = 15,
        alpha: float = 2.0,
        ot_solver: str = "linprog",
    ) -> Dict[str, Any]:
        image = self.load_image(image_path)
        prefix_text = self.build_prompt(question)
        prompt_inputs = self.processor(images=image, text=prefix_text, return_tensors="pt")
        prefix_ids = [int(token_id) for token_id in prompt_inputs["input_ids"][0].tolist()]
        generated_ids: List[int] = []
        answer_positions_for_scores: List[int] = []
        token_scores: List[Dict[str, Any]] = []
        eos_token_ids = self._eos_token_ids()

        visual_span: tuple[int, int] | None = None
        prompt_token_positions: List[int] = []
        token_layout_mode: str | None = None

        for generation_step in range(int(max_new_tokens)):
            current_token_ids = prefix_ids + generated_ids
            step_inputs = self._generation_step_inputs(prompt_inputs, current_token_ids)
            step_inputs = _to_device(step_inputs, self._input_device())
            outputs, captures = self._forward_with_layer_captures(step_inputs)

            visual_token_count = self._infer_visual_token_count(outputs)
            image_token_index = self.model.config.image_token_index
            token_layout = self._resolve_token_layout(
                token_ids=current_token_ids,
                prefix_length=len(prefix_ids),
                image_token_index=image_token_index,
                visual_token_count=visual_token_count,
            )
            if len(token_layout["visual_spans"]) != 1:
                raise ValueError("DGST currently supports a single image span per sample.")
            visual_start, visual_end = token_layout["visual_spans"][0]
            visual_span = (int(visual_start), int(visual_end))
            token_layout_mode = str(token_layout["mode"])
            prompt_token_positions = self._resolve_prompt_token_positions(
                token_ids=current_token_ids,
                prefix_length=len(prefix_ids),
                image_token_index=image_token_index,
                visual_token_count=visual_token_count,
            )
            if generated_ids:
                query_position = int(token_layout["answer_positions_merged"][-1])
            else:
                query_position = int(prompt_token_positions[-1])

            next_token_id = self._select_next_token_id(
                outputs.logits[0, -1, :],
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )
            if next_token_id in eos_token_ids:
                break

            semantic_probs_by_layer = self._build_semantic_probabilities(
                captures=captures,
                visual_start=int(visual_start),
                visual_end=int(visual_end),
                target_token_ids=[int(next_token_id)],
            )
            step_scores = self._score_requested_tokens(
                captures=captures,
                answer_token_ids=[int(next_token_id)],
                answer_token_positions=[int(query_position)],
                requested_indices=[0],
                prompt_token_positions=prompt_token_positions,
                visual_start=int(visual_start),
                visual_end=int(visual_end),
                semantic_probs_by_layer=semantic_probs_by_layer,
                tau=tau,
                transport_top_k=transport_top_k,
                gamma1=gamma1,
                gamma2=gamma2,
                baseline_layers=baseline_layers,
                risk_start_layer=risk_start_layer,
                alpha=alpha,
                ot_solver=ot_solver,
            )
            if step_scores:
                score_item = dict(step_scores[0])
                score_item["answer_token_index"] = int(generation_step)
                score_item["generation_step"] = int(generation_step)
                score_item["generation_time_scoring"] = "next_token_from_current_context"
                score_item["generation_query_merged_token_index"] = int(query_position)
                token_scores.append(score_item)

            generated_ids.append(int(next_token_id))
            answer_positions_for_scores.append(int(query_position))
            del outputs, captures

        if generated_ids:
            answer = self.processor.batch_decode(
                torch.tensor([generated_ids], dtype=torch.long),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        else:
            answer = ""

        visual_start, visual_end = visual_span if visual_span is not None else (0, 0)
        return {
            "question": question,
            "answer": answer,
            "model_id": self.model.name_or_path,
            "target_mode": "generated_tokens",
            "feature_timing": "generation_time",
            "token_layout_mode": token_layout_mode,
            "visual_span": {"start": visual_start, "end": visual_end, "count": visual_end - visual_start},
            "prompt_text_span": {
                "positions": prompt_token_positions,
                "count": len(prompt_token_positions),
                "last_position": prompt_token_positions[-1] if prompt_token_positions else None,
            },
            "answer_token_ids": [int(token_id) for token_id in generated_ids],
            "answer_token_positions": [int(position) for position in answer_positions_for_scores],
            "generation": {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(do_sample),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "token_count": len(generated_ids),
            },
            "dgst_config": {
                "tau": float(tau),
                "transport_top_k": int(transport_top_k),
                "gamma1": float(gamma1),
                "gamma2": float(gamma2),
                "baseline_layers": int(baseline_layers),
                "risk_start_layer": int(risk_start_layer),
                "alpha": float(alpha),
                "ot_solver": ot_solver,
                "target_aggregation": "generation_time_token",
            },
            "token_scores": token_scores,
        }

    def summarize_generation_time_sentence_last(
        self,
        generation_result: Dict[str, Any],
        sentence_mentions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        answer_token_ids = [int(token_id) for token_id in generation_result.get("answer_token_ids", [])]
        answer_token_positions = [
            int(position) for position in generation_result.get("answer_token_positions", [])
        ]
        targets, alignment_debug = build_sentence_last_token_targets(
            tokenizer=self.tokenizer,
            answer_token_ids=answer_token_ids,
            answer_token_positions=answer_token_positions,
            sentence_mentions=list(sentence_mentions or []),
            answer_text=str(generation_result.get("answer", "")),
        )
        result = dict(generation_result)
        result["target_mode"] = "sentence_last"
        result["sentence_mentions"] = list(sentence_mentions or [])
        result["sentence_alignment"] = alignment_debug

        sentence_scores = summarize_dgst_targets(
            result.get("token_scores", []),
            targets,
            target_aggregation="last",
        )
        sentence_entries = {
            int(entry.get("sentence_index", entry.get("mention_index", index))): entry
            for index, entry in enumerate(sentence_mentions or [])
        }
        for item in sentence_scores:
            sentence_index = int(item.get("mention_index", 0))
            entry = sentence_entries.get(sentence_index, {})
            item["sentence_index"] = sentence_index
            item["feature_timing"] = "generation_time"
            for key, value in entry.items():
                if str(key).startswith("sentence_"):
                    item[key] = value

        result["dgst_sentence_scores"] = sentence_scores
        result["sentence_count"] = len(sentence_scores)
        result["aligned_sentence_count"] = len(sentence_scores)
        result["dropped_unaligned_sentence_count"] = max(
            0,
            len(result.get("sentence_alignment", [])) - len(sentence_scores),
        )
        result["dgst_final_score_mean"] = mean_or_zero(item["dgst_final_score"] for item in sentence_scores)
        return result

    def analyze(
        self,
        image_path: str | Path,
        question: str,
        answer: str,
        *,
        target_mode: str = "objects",
        object_phrases: Sequence[str] | None = None,
        object_mentions: Sequence[Dict[str, Any]] | None = None,
        sentence_mentions: Sequence[Dict[str, Any]] | None = None,
        tau: float = 0.07,
        transport_top_k: int = 32,
        gamma1: float = 1.0,
        gamma2: float = 1.0,
        baseline_layers: int = 10,
        risk_start_layer: int = 15,
        alpha: float = 2.0,
        ot_solver: str = "linprog",
        target_aggregation: str = "mean",
    ) -> Dict[str, Any]:
        image = self.load_image(image_path)
        prefix_text = self.build_prompt(question)
        full_text = self.build_prompt(question, answer)

        prefix_inputs = self.processor(images=image, text=prefix_text, return_tensors="pt")
        full_inputs = self.processor(images=image, text=full_text, return_tensors="pt")
        prefix_ids = prefix_inputs["input_ids"][0]
        full_ids = full_inputs["input_ids"][0]

        full_inputs = _to_device(full_inputs, self._input_device())
        outputs, captures = self._forward_with_layer_captures(full_inputs)

        visual_token_count = self._infer_visual_token_count(outputs)
        image_token_index = self.model.config.image_token_index
        token_layout = self._resolve_token_layout(
            token_ids=full_ids.tolist(),
            prefix_length=prefix_ids.shape[0],
            image_token_index=image_token_index,
            visual_token_count=visual_token_count,
        )
        if len(token_layout["visual_spans"]) != 1:
            raise ValueError("DGST currently supports a single image span per sample.")

        visual_start, visual_end = token_layout["visual_spans"][0]
        answer_token_ids = token_layout["answer_token_ids"]
        answer_token_positions = token_layout["answer_positions_merged"]
        prompt_token_positions = self._resolve_prompt_token_positions(
            token_ids=full_ids.tolist(),
            prefix_length=prefix_ids.shape[0],
            image_token_index=image_token_index,
            visual_token_count=visual_token_count,
        )

        result: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "model_id": self.model.name_or_path,
            "target_mode": target_mode,
            "token_layout_mode": token_layout["mode"],
            "visual_span": {"start": visual_start, "end": visual_end, "count": visual_end - visual_start},
            "prompt_text_span": {
                "positions": prompt_token_positions,
                "count": len(prompt_token_positions),
                "last_position": prompt_token_positions[-1] if prompt_token_positions else None,
            },
            "dgst_config": {
                "tau": float(tau),
                "transport_top_k": int(transport_top_k),
                "gamma1": float(gamma1),
                "gamma2": float(gamma2),
                "baseline_layers": int(baseline_layers),
                "risk_start_layer": int(risk_start_layer),
                "alpha": float(alpha),
                "ot_solver": ot_solver,
                "target_aggregation": target_aggregation,
            },
            "token_scores": [],
        }

        targets: List[TargetSpan] = []
        if target_mode == "objects":
            if object_mentions:
                targets, alignment_debug = build_object_targets_from_mentions(
                    tokenizer=self.tokenizer,
                    answer_token_ids=answer_token_ids,
                    answer_token_positions=answer_token_positions,
                    object_mentions=object_mentions,
                )
                result["object_mentions"] = list(object_mentions)
                result["object_alignment"] = alignment_debug
            else:
                phrases = list(object_phrases or [])
                targets = build_object_targets(
                    tokenizer=self.tokenizer,
                    answer_token_ids=answer_token_ids,
                    answer_token_positions=answer_token_positions,
                    object_phrases=phrases,
                )
                result["object_phrases"] = phrases
                result["object_alignment"] = []
            requested_indices = sorted(
                {
                    int(index)
                    for target in targets
                    for index in range(target.answer_token_start, target.answer_token_end)
                }
            )
        elif target_mode in {"sentences", "sentence_last"}:
            targets, alignment_debug = build_sentence_last_token_targets(
                tokenizer=self.tokenizer,
                answer_token_ids=answer_token_ids,
                answer_token_positions=answer_token_positions,
                sentence_mentions=list(sentence_mentions or []),
                answer_text=answer,
            )
            result["sentence_mentions"] = list(sentence_mentions or [])
            result["sentence_alignment"] = alignment_debug
            requested_indices = sorted(
                {
                    int(index)
                    for target in targets
                    for index in range(target.answer_token_start, target.answer_token_end)
                }
            )
        else:
            requested_indices = list(range(len(answer_token_ids)))

        if requested_indices:
            semantic_probs_by_layer = self._build_semantic_probabilities(
                captures=captures,
                visual_start=visual_start,
                visual_end=visual_end,
                target_token_ids=[int(answer_token_ids[index]) for index in requested_indices],
            )
            token_scores = self._score_requested_tokens(
                captures=captures,
                answer_token_ids=answer_token_ids,
                answer_token_positions=answer_token_positions,
                requested_indices=requested_indices,
                prompt_token_positions=prompt_token_positions,
                visual_start=visual_start,
                visual_end=visual_end,
                semantic_probs_by_layer=semantic_probs_by_layer,
                tau=tau,
                transport_top_k=transport_top_k,
                gamma1=gamma1,
                gamma2=gamma2,
                baseline_layers=baseline_layers,
                risk_start_layer=risk_start_layer,
                alpha=alpha,
                ot_solver=ot_solver,
            )
            result["token_scores"] = token_scores

        if target_mode == "objects":
            object_scores = summarize_dgst_targets(
                result["token_scores"],
                targets,
                target_aggregation=target_aggregation,
            )
            result["dgst_object_scores"] = object_scores
            result["object_count"] = len(object_scores)
            result["aligned_object_count"] = len(object_scores)
            result["dropped_unaligned_count"] = max(0, len(result.get("object_alignment", [])) - len(object_scores))
            result["dgst_final_score_mean"] = mean_or_zero(item["dgst_final_score"] for item in object_scores)
        elif target_mode in {"sentences", "sentence_last"}:
            sentence_scores = summarize_dgst_targets(
                result["token_scores"],
                targets,
                target_aggregation="last",
            )
            sentence_entries = {
                int(entry.get("sentence_index", entry.get("mention_index", index))): entry
                for index, entry in enumerate(sentence_mentions or [])
            }
            for item in sentence_scores:
                sentence_index = int(item.get("mention_index", 0))
                entry = sentence_entries.get(sentence_index, {})
                item["sentence_index"] = sentence_index
                for key, value in entry.items():
                    if str(key).startswith("sentence_"):
                        item[key] = value
            result["dgst_sentence_scores"] = sentence_scores
            result["sentence_count"] = len(sentence_scores)
            result["aligned_sentence_count"] = len(sentence_scores)
            result["dropped_unaligned_sentence_count"] = max(
                0,
                len(result.get("sentence_alignment", [])) - len(sentence_scores),
            )
            result["dgst_final_score_mean"] = mean_or_zero(item["dgst_final_score"] for item in sentence_scores)
        else:
            result["dgst_final_score_mean"] = mean_or_zero(item["dgst_final_score"] for item in result["token_scores"])

        return result

    def generate_and_analyze(
        self,
        image_path: str | Path,
        question: str,
        *,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        target_mode: str = "objects",
        object_phrases: Sequence[str] | None = None,
        object_mentions: Sequence[Dict[str, Any]] | None = None,
        sentence_mentions: Sequence[Dict[str, Any]] | None = None,
        tau: float = 0.07,
        transport_top_k: int = 32,
        gamma1: float = 1.0,
        gamma2: float = 1.0,
        baseline_layers: int = 10,
        risk_start_layer: int = 15,
        alpha: float = 2.0,
        ot_solver: str = "linprog",
        target_aggregation: str = "mean",
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
            target_mode=target_mode,
            object_phrases=object_phrases,
            object_mentions=object_mentions,
            sentence_mentions=sentence_mentions,
            tau=tau,
            transport_top_k=transport_top_k,
            gamma1=gamma1,
            gamma2=gamma2,
            baseline_layers=baseline_layers,
            risk_start_layer=risk_start_layer,
            alpha=alpha,
            ot_solver=ot_solver,
            target_aggregation=target_aggregation,
        )
        result["generation"] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        }
        return result
