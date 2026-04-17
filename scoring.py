from __future__ import annotations

# vICR 底层张量打分函数：JSD、投影分布和 top-k support 裁剪都集中放在这里。

from typing import Iterable, List

import torch


EPS = 1e-12


def renormalize(values: torch.Tensor) -> torch.Tensor:
    # 把一维向量重新归一化成概率分布。
    # 如果总和太小，就退化成均匀分布，避免后面出现除零。
    values = values.float().clamp_min(0.0)
    total = values.sum()
    if total <= EPS:
        return torch.full_like(values, 1.0 / max(values.numel(), 1))
    return values / total


def masked_topk(values: torch.Tensor, top_k: int) -> tuple[torch.Tensor, List[int]]:
    # 只保留 top-k 位置，其余位置清零后再归一化。
    # 这个函数适合“先挑最重要的位置，再把它们当成概率分布”。
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got shape={tuple(values.shape)}")
    if values.numel() == 0:
        raise ValueError("Cannot apply top-k masking to an empty tensor.")

    k = min(max(top_k, 1), values.numel())
    top_values, top_indices = torch.topk(values, k=k)
    masked = torch.zeros_like(values)
    masked[top_indices] = top_values
    return renormalize(masked), top_indices.tolist()


def topk_values(values: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # 直接返回 top-k 的数值和索引，不做归一化。
    # 论文式 ICR 会先取 support 位置，再对取出的向量做标准化 softmax。
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D tensor, got shape={tuple(values.shape)}")
    if values.numel() == 0:
        raise ValueError("Cannot apply top-k selection to an empty tensor.")

    k = min(max(top_k, 1), values.numel())
    top_values, top_indices = torch.topk(values, k=k)
    return top_values, top_indices


def jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # 对两个一维概率分布计算 Jensen-Shannon Divergence。
    p = renormalize(p)
    q = renormalize(q)
    m = 0.5 * (p + q)
    log_base = torch.log(torch.tensor(2.0, device=p.device))

    def _kl(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        safe_a = a.clamp_min(EPS)
        safe_b = b.clamp_min(EPS)
        return torch.sum(safe_a * (torch.log(safe_a) - torch.log(safe_b))) / log_base

    return 0.5 * (_kl(p, m) + _kl(q, m))


def standardized_softmax(values: torch.Tensor) -> torch.Tensor:
    # 先做 z-score 标准化，再做 softmax。
    # 这是为了贴近原论文里 ICR 分数的处理方式。
    values = values.float()
    centered = values - values.mean()
    std = values.std(unbiased=False).clamp_min(EPS)
    normalized = centered / std
    return torch.softmax(normalized, dim=-1)


def jsd_standardized(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # 原论文风格的 JSD：不是直接把原始向量当概率，
    # 而是先标准化、再 softmax、最后再算 JSD。
    return jsd(standardized_softmax(p), standardized_softmax(q))


def projection_distribution(update: torch.Tensor, basis_states: torch.Tensor) -> torch.Tensor:
    # 计算 hidden update 在一组 basis states 上的投影分布。
    # 输出是 softmax 后的概率形式，适合直接和 attention 分布比较。
    if update.ndim != 1:
        raise ValueError(f"Expected update vector with shape [hidden], got {tuple(update.shape)}")
    if basis_states.ndim != 2:
        raise ValueError(
            f"Expected basis states with shape [tokens, hidden], got {tuple(basis_states.shape)}"
        )
    norms = basis_states.norm(dim=-1).clamp_min(EPS)
    scores = torch.matmul(basis_states, update) / norms
    return torch.softmax(scores.float(), dim=-1)


def projection_scores(update: torch.Tensor, basis_states: torch.Tensor) -> torch.Tensor:
    # 和 projection_distribution 类似，但这里只返回原始投影分数。
    # 后续如果要按论文流程做“先选 top-k，再统一标准化”，会更方便。
    if update.ndim != 1:
        raise ValueError(f"Expected update vector with shape [hidden], got {tuple(update.shape)}")
    if basis_states.ndim != 2:
        raise ValueError(
            f"Expected basis states with shape [tokens, hidden], got {tuple(basis_states.shape)}"
        )
    norms = basis_states.norm(dim=-1).clamp_min(EPS)
    return torch.matmul(basis_states, update) / norms


def mean_or_zero(values: Iterable[float]) -> float:
    # 空列表时返回 0，避免到处额外写空值判断。
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))
