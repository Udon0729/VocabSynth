"""最適輸送に基づく出力重み合成。

非 weight tying モデルにおいて、入力埋め込み空間 E と
出力重み空間 W の幾何学的対応を局所的な最適輸送で推定し、
仮想トークンの出力重みを合成する。

二つの方式を提供する。

**交差空間 OT（均衡 / 不均衡）**:
  E 点と W 点の間のユークリッド距離をコストとして
  Sinkhorn ソルバで輸送計画を解く。
  二空間が整列していない場合、コスト行列の情報量が低下する。

**Gromov-Wasserstein（GW / 半緩和 GW / 部分 GW）**:
  各空間内の距離構造 C_E, C_W を比較し、構造的対応を求める。
  二空間の絶対的な配置が無関係でも、内部の相対的な幾何が
  類似していれば有効に機能する。
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
    GW = "gw"
    SEMIRELAXED_GW = "srgw"
    PARTIAL_GW = "pgw"


class OTOutputComposer:
    """最適輸送の重心射影による出力重み合成器。

    構成トークンごとに E 空間の k 近傍を取得し、局所的な
    輸送計画を解いて W 空間への重心射影で出力重みを求める。

    交差空間 OT（``BALANCED``, ``UNBALANCED``）は E-W 間の
    ユークリッド距離をコストに使い、Gromov-Wasserstein 系
    （``GW``, ``SEMIRELAXED_GW``, ``PARTIAL_GW``）は各空間
    内部の距離構造を比較する。

    Args:
        input_embeddings: 入力埋め込み行列 E (形状: ``[V, d]``)。
        output_weights: 出力重み行列 W (形状: ``[V, d]``)。
        k_neighbors: 近傍トークン数。
        epsilon: エントロピー正則化係数。
        tau: UOT の周辺制約 KL ペナルティ。
        partial_mass: 部分 GW で輸送する質量の割合 (0, 1]。
    """

    def __init__(
        self,
        input_embeddings: torch.Tensor,
        output_weights: torch.Tensor,
        k_neighbors: int = 200,
        epsilon: float = 0.05,
        tau: float = 1.0,
        partial_mass: float = 0.5,
    ) -> None:
        self._E = input_embeddings.detach().float()
        self._W = output_weights.detach().float()
        self._k = k_neighbors
        self._eps = epsilon
        self._tau = tau
        self._partial_mass = partial_mass
        self._V, self._d = self._E.shape

        self._E_norm = torch.nn.functional.normalize(self._E, dim=1)

        self._bary_cache: dict[tuple, torch.Tensor] = {}

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
        """トークンの重心射影を計算する。

        交差空間 OT は E-W 間距離をコストに使い、GW 系は
        各空間内の距離行列 C_E, C_W の構造的対応を求める。
        """
        cache_key = (
            token_id, ot_method.value, self._tau, self._partial_mass,
        )
        if cache_key in self._bary_cache:
            return self._bary_cache[cache_key]

        neighbor_ids = self._find_neighbors(token_id)
        E_local = self._E[neighbor_ids]
        W_local = self._W[neighbor_ids]
        k = len(neighbor_ids)

        if ot_method in (OTMethod.BALANCED, OTMethod.UNBALANCED):
            pi = self._solve_cross_space(E_local, W_local, k, ot_method)
        else:
            pi = self._solve_gromov_wasserstein(
                E_local, W_local, k, ot_method,
            )

        query_pos = (neighbor_ids == token_id).nonzero(as_tuple=True)[0].item()
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

    def _solve_cross_space(
        self,
        E_local: torch.Tensor,
        W_local: torch.Tensor,
        k: int,
        ot_method: OTMethod,
    ) -> np.ndarray:
        """E-W 間ユークリッド距離をコストとする交差空間 OT。"""
        C = torch.cdist(E_local, W_local).pow(2)
        C_np = C.cpu().numpy().astype(np.float64)

        C_median = np.median(C_np[C_np > 0])
        if C_median > 0:
            C_np = C_np / C_median

        a = np.ones(k, dtype=np.float64) / k
        b = np.ones(k, dtype=np.float64) / k

        if ot_method == OTMethod.BALANCED:
            return pot.bregman.sinkhorn(
                a, b, C_np,
                reg=self._eps,
                numItermax=1000,
                warn=False,
            )
        return pot.unbalanced.sinkhorn_unbalanced(
            a, b, C_np,
            reg=self._eps,
            reg_m=self._tau,
            numItermax=1000,
            warn=False,
        )

    def _solve_gromov_wasserstein(
        self,
        E_local: torch.Tensor,
        W_local: torch.Tensor,
        k: int,
        ot_method: OTMethod,
    ) -> np.ndarray:
        """各空間内の距離構造を比較する Gromov-Wasserstein。"""
        C_E = torch.cdist(E_local, E_local).pow(2)
        C_W = torch.cdist(W_local, W_local).pow(2)

        C_E_np = C_E.cpu().numpy().astype(np.float64)
        C_W_np = C_W.cpu().numpy().astype(np.float64)

        C_E_med = np.median(C_E_np[C_E_np > 0])
        C_W_med = np.median(C_W_np[C_W_np > 0])
        if C_E_med > 0:
            C_E_np = C_E_np / C_E_med
        if C_W_med > 0:
            C_W_np = C_W_np / C_W_med

        p = np.ones(k, dtype=np.float64) / k
        q = np.ones(k, dtype=np.float64) / k

        if ot_method == OTMethod.GW:
            return pot.gromov.entropic_gromov_wasserstein(
                C_E_np, C_W_np, p, q,
                loss_fun="square_loss",
                epsilon=self._eps,
                max_iter=1000,
                verbose=False,
                log=False,
            )
        if ot_method == OTMethod.SEMIRELAXED_GW:
            return pot.gromov.entropic_semirelaxed_gromov_wasserstein(
                C_E_np, C_W_np, p,
                loss_fun="square_loss",
                epsilon=self._eps,
                max_iter=1000,
                verbose=False,
                log=False,
            )
        m = self._partial_mass * min(p.sum(), q.sum())
        return pot.gromov.entropic_partial_gromov_wasserstein(
            C_E_np, C_W_np, p, q,
            reg=self._eps,
            m=m,
            loss_fun="square_loss",
            numItermax=1000,
            verbose=False,
            log=False,
        )

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

        交差空間（E-W 直接比較）と各空間内部の距離構造の両方を報告する。
        GW が有効かどうかの事前判断に使える。

        Args:
            token_id: 対象トークンID。

        Returns:
            近傍の幾何情報を含む辞書。
        """
        neighbor_ids = self._find_neighbors(token_id)
        E_local = self._E[neighbor_ids]
        W_local = self._W[neighbor_ids]
        k = len(neighbor_ids)

        paired_cos = torch.nn.functional.cosine_similarity(
            E_local, W_local, dim=1
        )

        C_cross = torch.cdist(E_local, W_local).pow(2)
        diag_costs = C_cross.diag()
        off_diag_mask = ~torch.eye(k, dtype=torch.bool)
        off_diag_costs = C_cross[off_diag_mask]

        C_E = torch.cdist(E_local, E_local).pow(2)
        C_W = torch.cdist(W_local, W_local).pow(2)
        C_E_flat = C_E[off_diag_mask]
        C_W_flat = C_W[off_diag_mask]

        rank_corr = _spearman_corr(C_E_flat, C_W_flat)

        return {
            "token_id": token_id,
            "k": k,
            "paired_cosine_mean": paired_cos.mean().item(),
            "paired_cosine_std": paired_cos.std().item(),
            "diagonal_cost_mean": diag_costs.mean().item(),
            "off_diagonal_cost_mean": off_diag_costs.mean().item(),
            "E_norm_mean": E_local.norm(dim=1).mean().item(),
            "W_norm_mean": W_local.norm(dim=1).mean().item(),
            "intra_distance_rank_corr": rank_corr,
        }


def _spearman_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    """二つのベクトルの Spearman 順位相関を計算する。"""
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.norm() * ry.norm()
    if denom < 1e-12:
        return 0.0
    return (rx @ ry / denom).item()
