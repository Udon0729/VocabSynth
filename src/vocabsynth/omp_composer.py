"""直交マッチング追跡（OMP）による疎な線形結合での埋め込み合成。

既存語彙の埋め込み（または出力重み）行列を辞書として使い、
ターゲットベクトルを辞書原子の疎な線形結合として表現する。

平均合成が全新トークンを同一点に縮退させる問題（CW2V, GTI が指摘）
に対して、OMP は各仮想語彙に固有の疎表現を与え、表現力を高める。
訓練は不要。

重要な設計判断:
  ターゲット（構成サブワードの平均）は辞書行列の行の線形結合であるため、
  構成サブワード自体を辞書に含めると OMP が自明に完全再構成する（残差 0）。
  そこで ``exclude_ids`` を使い、構成サブワード自体を辞書から除外する。
  これにより OMP は「構成サブワード以外の意味的に関連するトークン」の
  疎な組み合わせでターゲットを近似する必要があり、非自明な合成が実現する。

  さらに、語レベルの個別合成ではなく仮想語彙レベルで 1 本の出力重みを
  構築し、VirtualLogitHead と同じ h @ w_z 方式で logit を計算する。

参考: Goddard+ 2025, mergekit-tokensurgeon
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase

from vocabsynth.logit_head import VirtualLogitHead
from vocabsynth.registry import VocabularyRegistry


def omp_compose(
    target: torch.Tensor,
    dictionary: torch.Tensor,
    sparsity: int = 32,
    exclude_ids: list[int] | None = None,
) -> tuple[torch.Tensor, list[int], torch.Tensor, float]:
    """直交マッチング追跡で辞書原子の疎な線形結合を求める。

    アルゴリズム:
      1. 残差 r = target を初期化
      2. sparsity 回繰り返す:
         a. 辞書の全原子との内積を計算し、絶対値最大の原子を選択
         b. 選択済み原子集合で最小二乗法により係数を再計算
         c. 残差を更新: r = target - dictionary[選択] @ coefficients
      3. dictionary[選択] @ coefficients が最終的な合成ベクトル

    Args:
        target: ターゲットベクトル (形状: ``[d]``)。
        dictionary: 辞書行列 (形状: ``[V, d]``)。各行が一つの原子。
        sparsity: 非零要素数（選択する原子の数）。
        exclude_ids: 辞書から除外するインデックス。構成サブワード自体を
            除外して非自明な解を強制するために使う。

    Returns:
        (合成ベクトル, 選択された辞書インデックス, 係数, 最終残差ノルム)
        のタプル。
    """
    V = dictionary.shape[0]
    device = target.device

    # 除外マスクの構築
    exclude_set = set(exclude_ids) if exclude_ids else set()

    # 残差を初期化
    residual = target.clone()

    # 選択済み原子のインデックス
    selected_indices: list[int] = []
    coefficients = torch.zeros(0, device=device, dtype=target.dtype)

    for _ in range(min(sparsity, V - len(exclude_set))):
        # 辞書全原子との内積をバッチで計算
        correlations = dictionary @ residual

        # 除外インデックスと選択済み原子をマスク
        mask_ids = list(exclude_set) + selected_indices
        if mask_ids:
            mask_tensor = torch.tensor(mask_ids, device=device, dtype=torch.long)
            correlations[mask_tensor] = 0.0

        # 絶対値が最大の原子を選択
        best_idx = correlations.abs().argmax().item()

        # 全相関が 0 の場合（選択可能な原子がない）は終了
        if correlations.abs().max().item() < 1e-12:
            break

        selected_indices.append(best_idx)

        # 選択済み原子の部分行列で最小二乗法
        D_selected = dictionary[selected_indices]  # [k, d]
        coefficients = torch.linalg.lstsq(
            D_selected.T, target.unsqueeze(1)
        ).solution.squeeze(1)

        # 残差を更新
        residual = target - D_selected.T @ coefficients

    # 最終的な合成ベクトル
    if selected_indices:
        D_selected = dictionary[selected_indices]
        composed = D_selected.T @ coefficients
    else:
        composed = target.clone()

    final_residual_norm = residual.norm().item()

    return composed, selected_indices, coefficients, final_residual_norm


def build_omp_virtual_logit_head(
    tokenizer: PreTrainedTokenizerBase,
    registry: VocabularyRegistry,
    output_weight: torch.Tensor,
    sparsity: int = 32,
    norm_correction: bool = True,
    exclude_components: bool = True,
) -> tuple[VirtualLogitHead, dict]:
    """OMP 合成で仮想語彙の出力重みを構築し VirtualLogitHead を返す。

    各仮想語彙について:
      1. 全構成サブワードの出力重み平均をターゲットとする
      2. OMP で出力重み行列を辞書として疎な線形結合を求める
      3. ノルム補正を適用する

    Args:
        tokenizer: トークナイザ。
        registry: 仮想トークンのレジストリ。
        output_weight: 出力重み行列 (形状: ``[V, d]``)。
        sparsity: 非零要素数。
        norm_correction: ノルム補正の適用有無。
        exclude_components: 構成サブワードを辞書から除外するかどうか。

    Returns:
        (VirtualLogitHead, 診断情報辞書) のタプル。
    """
    W = output_weight.detach().float()
    dict_norms = W.norm(dim=1)
    median_norm = dict_norms.median()

    weight_rows: list[torch.Tensor] = []
    surface_names: list[str] = []
    diagnostics: dict[str, dict] = {}

    for vtoken in registry:
        # 全構成サブワードの ID を収集
        all_component_ids: list[int] = []
        component_detail: list[dict] = []
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            all_component_ids.extend(ids)
            tokens_str = [tokenizer.decode([t]) for t in ids]
            component_detail.append({
                "word": word,
                "token_ids": ids,
                "tokens": tokens_str,
            })

        # 全構成サブワードの出力重み平均をターゲットとする
        target = W[all_component_ids].mean(dim=0)

        # 除外リストの構築
        exclude_ids = all_component_ids if exclude_components else None

        # OMP で疎な線形結合を求める
        composed, selected, coeffs, residual_norm = omp_compose(
            target, W, sparsity, exclude_ids=exclude_ids,
        )

        # ノルム補正
        if norm_correction:
            current_norm = composed.norm()
            if current_norm > 0:
                composed = composed * (median_norm / current_norm)

        weight_rows.append(composed)
        surface_names.append(vtoken.surface)

        # 診断情報の記録
        cos_sim = F.cosine_similarity(
            target.unsqueeze(0), composed.unsqueeze(0),
        ).item()

        # 選択された原子のトークン文字列
        selected_tokens = [
            (idx, tokenizer.decode([idx]).strip(), coeffs[i].item())
            for i, idx in enumerate(selected)
        ] if selected else []
        # 係数の絶対値で降順ソート
        selected_tokens.sort(key=lambda x: abs(x[2]), reverse=True)

        diagnostics[vtoken.surface] = {
            "components": component_detail,
            "target_norm": target.norm().item(),
            "composed_norm": composed.norm().item(),
            "residual_norm": residual_norm,
            "cosine_similarity": cos_sim,
            "num_selected": len(selected),
            "top_atoms": selected_tokens[:10],
        }

    output_weights = torch.stack(weight_rows)
    head = VirtualLogitHead(output_weights, surface_names)
    return head, diagnostics


def build_omp_component_logit_head(
    tokenizer: PreTrainedTokenizerBase,
    registry: VocabularyRegistry,
    output_weight: torch.Tensor,
    sparsity: int = 32,
    norm_correction: bool = True,
    exclude_components: bool = True,
) -> tuple[VirtualLogitHead, dict]:
    """語レベルの OMP 合成 + word_mean 集約による VirtualLogitHead を構築。

    :func:`build_omp_virtual_logit_head` が仮想語彙レベルで 1 本の
    出力重みを構築するのに対し、本関数は語レベルで独立に OMP を適用し、
    その平均を仮想語彙の出力重みとする。

    Args:
        tokenizer: トークナイザ。
        registry: 仮想トークンのレジストリ。
        output_weight: 出力重み行列 (形状: ``[V, d]``)。
        sparsity: 非零要素数。
        norm_correction: ノルム補正の適用有無。
        exclude_components: 構成サブワードを辞書から除外するかどうか。

    Returns:
        (VirtualLogitHead, 診断情報辞書) のタプル。
    """
    W = output_weight.detach().float()
    dict_norms = W.norm(dim=1)
    median_norm = dict_norms.median()

    weight_rows: list[torch.Tensor] = []
    surface_names: list[str] = []
    diagnostics: dict[str, dict] = {}

    for vtoken in registry:
        word_vectors: list[torch.Tensor] = []
        word_diags: list[dict] = []

        # 全構成サブワードの ID を収集（除外用）
        all_component_ids: list[int] = []
        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            all_component_ids.extend(ids)

        for word in vtoken.components:
            ids = tokenizer.encode(word, add_special_tokens=False)
            target = W[ids].mean(dim=0)

            exclude_ids = all_component_ids if exclude_components else None
            composed, selected, coeffs, residual_norm = omp_compose(
                target, W, sparsity, exclude_ids=exclude_ids,
            )

            cos_sim = F.cosine_similarity(
                target.unsqueeze(0), composed.unsqueeze(0),
            ).item()
            word_vectors.append(composed)

            selected_tokens = [
                (idx, tokenizer.decode([idx]).strip(), coeffs[i].item())
                for i, idx in enumerate(selected)
            ] if selected else []
            selected_tokens.sort(key=lambda x: abs(x[2]), reverse=True)

            word_diags.append({
                "word": word,
                "residual_norm": residual_norm,
                "cosine_similarity": cos_sim,
                "top_atoms": selected_tokens[:5],
            })

        # 語レベルの平均（word_mean）
        composed_mean = torch.stack(word_vectors).mean(dim=0)

        # ノルム補正
        if norm_correction:
            current_norm = composed_mean.norm()
            if current_norm > 0:
                composed_mean = composed_mean * (median_norm / current_norm)

        weight_rows.append(composed_mean)
        surface_names.append(vtoken.surface)
        diagnostics[vtoken.surface] = {"words": word_diags}

    output_weights = torch.stack(weight_rows)
    head = VirtualLogitHead(output_weights, surface_names)
    return head, diagnostics
