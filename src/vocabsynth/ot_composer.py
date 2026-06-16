"""不均衡最適輸送に基づく出力重み合成。

非 weight tying モデルにおいて、入力埋め込み空間 E と
出力重み空間 W の幾何学的対応を局所的な不均衡最適輸送
（UOT）で推定し、仮想トークンの出力重みを合成する。

従来手法（W の構成トークン行を直接平均する）では E→W 間の
非自明な幾何構造が無視される。UOT の重心射影を用いることで、
構成トークン近傍の E→W 対応関係を反映した出力重みが得られる。
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import ot as pot
import torch

from vocabsynth.composer import ComposeMethod


class OTMethod(Enum):
    """最適輸送の方式。"""

    BALANCED = "balanced"
    UNBALANCED = "unbalanced"


class OTOutputComposer:
    """UOT 重心射影による出力重み合成器。

    構成トークンごとに E 空間の k 近傍を取得し、同一トークン群の
    W 空間上の配置との間で局所的な UOT 計画を解く。
    得られた輸送計画の重心射影で各構成トークンの出力重みを求め、
    それらを入力側と同じ重みで合成する。

    Args:
        input_embeddings: 入力埋め込み行列 E (形状: ``[V, d]``)。
        output_weights: 出力重み行列 W (形状: ``[V, d]``)。
        k_neighbors: 近傍トークン数。
        epsilon: エントロピー正則化係数。小さいほど決定的な輸送になる。
        tau: 周辺制約の KL ペナルティ係数。
            小さいほど不均衡性を許容し、大きいほど均衡に近づく。
            :attr:`OTMethod.BALANCED` では無視される。
    """

    def __init__(
        self,
        input_embeddings: torch.Tensor,
        output_weights: torch.Tensor,
        k_neighbors: int = 200,
        epsilon: float = 0.05,
        tau: float = 1.0,
    ) -> None:
        self._E = input_embeddings.detach().float()
        self._W = output_weights.detach().float()
        self._k = k_neighbors
        self._eps = epsilon
        self._tau = tau
        self._V, self._d = self._E.shape

        self._E_norm = torch.nn.functional.normalize(self._E, dim=1)

        self._bary_cache: dict[tuple[int, str, float], torch.Tensor] = {}

    def compose(
        self,
        token_ids: list[int],
        compose_method: ComposeMethod = ComposeMethod.MEAN,
        ot_method: OTMethod = OTMethod.UNBALANCED,
        tokenizer=None,
    ) -> torch.Tensor:
        """構成トークンID列から UOT 重心射影で出力重みを合成する。

        Args:
            token_ids: 構成トークンのID列。
            compose_method: 構成トークン間の重み配分方式。
            ot_method: 最適輸送の方式。
            tokenizer: ``ComposeMethod.LENGTH_WEIGHTED`` で使用。

        Returns:
            合成出力重みベクトル (形状: ``[d]``)。
        """
        weights = self._compute_alpha(
            token_ids, compose_method, tokenizer
        )

        mapped = []
        for tid in token_ids:
            bary = self._barycentric_projection(tid, ot_method)
            mapped.append(bary)

        mapped_stack = torch.stack(mapped)
        W_z = (weights.unsqueeze(1) * mapped_stack).sum(dim=0)

        target_norm = self._W[token_ids].norm(dim=1).mean()
        if W_z.norm() > 0:
            W_z = W_z * (target_norm / W_z.norm())

        return W_z

    def compose_direct(
        self,
        token_ids: list[int],
        compose_method: ComposeMethod = ComposeMethod.MEAN,
        tokenizer=None,
    ) -> torch.Tensor:
        """W の行を直接合成する（ベースライン）。

        OT を使わず、構成トークンの W 行を重み付き平均する。

        Args:
            token_ids: 構成トークンのID列。
            compose_method: 重み配分方式。
            tokenizer: ``ComposeMethod.LENGTH_WEIGHTED`` で使用。

        Returns:
            合成出力重みベクトル (形状: ``[d]``)。
        """
        weights = self._compute_alpha(
            token_ids, compose_method, tokenizer
        )
        W_local = self._W[token_ids]
        W_z = (weights.unsqueeze(1) * W_local).sum(dim=0)

        target_norm = W_local.norm(dim=1).mean()
        if W_z.norm() > 0:
            W_z = W_z * (target_norm / W_z.norm())

        return W_z

    def _barycentric_projection(
        self, token_id: int, ot_method: OTMethod
    ) -> torch.Tensor:
        """トークンの UOT 重心射影を計算する。

        1. E 空間で k 近傍を探索する。
        2. 近傍トークン群の E 行と W 行の間で局所 OT を解く。
        3. 対象トークン行の輸送計画から W 空間への重心射影を返す。
        """
        cache_key = (token_id, ot_method.value, self._tau)
        if cache_key in self._bary_cache:
            return self._bary_cache[cache_key]

        neighbor_ids = self._find_neighbors(token_id)

        E_local = self._E[neighbor_ids]
        W_local = self._W[neighbor_ids]

        C = torch.cdist(E_local, W_local).pow(2)
        C_np = C.cpu().numpy().astype(np.float64)

        C_median = np.median(C_np[C_np > 0])
        if C_median > 0:
            C_normalized = C_np / C_median
        else:
            C_normalized = C_np

        k = len(neighbor_ids)
        a = np.ones(k, dtype=np.float64) / k
        b = np.ones(k, dtype=np.float64) / k

        if ot_method == OTMethod.BALANCED:
            pi = pot.bregman.sinkhorn(
                a, b, C_normalized,
                reg=self._eps,
                numItermax=1000,
                warn=False,
            )
        else:
            pi = pot.unbalanced.sinkhorn_unbalanced(
                a, b, C_normalized,
                reg=self._eps,
                reg_m=self._tau,
                numItermax=1000,
                warn=False,
            )

        query_mask = (neighbor_ids == token_id)
        query_pos = query_mask.nonzero(as_tuple=True)[0].item()

        row = pi[query_pos]
        row_sum = row.sum()

        if row_sum > 1e-10:
            bary_weights = torch.tensor(
                row / row_sum,
                dtype=self._W.dtype,
                device=self._W.device,
            )
            result = (bary_weights.unsqueeze(1) * W_local).sum(dim=0)
        else:
            result = self._W[token_id].clone()

        self._bary_cache[cache_key] = result
        return result

    def _find_neighbors(self, token_id: int) -> torch.Tensor:
        """E 空間での k 近傍トークンIDを返す。"""
        query = self._E_norm[token_id]
        sims = torch.mv(self._E_norm, query)
        _, top_ids = sims.topk(self._k)
        return top_ids

    def _compute_alpha(
        self,
        token_ids: list[int],
        method: ComposeMethod,
        tokenizer=None,
    ) -> torch.Tensor:
        k = len(token_ids)
        if k == 1:
            return torch.ones(1, device=self._W.device)

        if method == ComposeMethod.MEAN:
            return torch.ones(k, device=self._W.device) / k

        if method == ComposeMethod.HEAD_WEIGHTED:
            w = torch.full((k,), 0.4 / (k - 1), device=self._W.device)
            w[-1] = 0.6
            return w

        if method == ComposeMethod.LENGTH_WEIGHTED:
            if tokenizer is None:
                return torch.ones(k, device=self._W.device) / k
            lengths = torch.tensor([
                max(len(tokenizer.decode([tid]).strip()), 1)
                for tid in token_ids
            ], dtype=torch.float, device=self._W.device)
            return lengths / lengths.sum()

        return torch.ones(k, device=self._W.device) / k

    def diagnose_local_geometry(
        self, token_id: int
    ) -> dict:
        """構成トークン近傍の E→W 対応関係を診断する。

        Args:
            token_id: 対象トークンID。

        Returns:
            近傍の E-W 整合度、ノルム分布などの診断情報。
        """
        neighbor_ids = self._find_neighbors(token_id)
        E_local = self._E[neighbor_ids]
        W_local = self._W[neighbor_ids]

        paired_cos = torch.nn.functional.cosine_similarity(
            E_local, W_local, dim=1
        )

        C = torch.cdist(E_local, W_local).pow(2)
        diag_costs = C.diag()
        off_diag_mask = ~torch.eye(len(neighbor_ids), dtype=torch.bool)
        off_diag_costs = C[off_diag_mask]

        return {
            "token_id": token_id,
            "k": len(neighbor_ids),
            "paired_cosine_mean": paired_cos.mean().item(),
            "paired_cosine_std": paired_cos.std().item(),
            "diagonal_cost_mean": diag_costs.mean().item(),
            "off_diagonal_cost_mean": off_diag_costs.mean().item(),
            "E_norm_mean": E_local.norm(dim=1).mean().item(),
            "W_norm_mean": W_local.norm(dim=1).mean().item(),
        }
