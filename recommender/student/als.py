"""实验三·Spark ALS Provider（business 模式）.

原理：LFM 把用户/电影映射到 rank 维隐向量，以内积预测评分；
ALS 通过交替最小二乘在 Spark MLlib 中求解这些隐向量（见 scripts/train_als_spark.py）。
本 Provider 只加载训练好的因子工件，对请求用户做内积打分：
  score(u, i) = <p_u, q_i>
排除已评分电影后取 Top-N；同分按 movie_id 升序。
模型不覆盖的账号（如新注册 app 账号）返回合法空结果，不伪造。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ..business import read_history
from ..contracts import Algorithm, RecommendationItem, RecommendationRequest, RecommendationResult
from ..data import FrozenData


STUDENT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = STUDENT_DIR / "als-factors-business.json"
MANIFEST_PATH = STUDENT_DIR / "als-manifest.json"
MODEL_VERSION = "als-v1"
MIN_SUPPORT_MOVIES = 25  # 候选过滤：显式ALS按预测评分排Top-N易被低支持度电影霸榜


@lru_cache(maxsize=1)
def _load_factors() -> dict:
    raw = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    return raw


class AlsProvider:
    """实验三 Provider：Spark MLlib ALS 隐向量内积打分."""

    model_manifest_path = str(MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        factors = _load_factors()
        catalog = FrozenData().catalog()
        history = read_history(request)
        rated = {fb.movie_id for fb in history.ratings}

        user_vec = factors["users"].get(str(request.subject_id))
        support = factors.get("support", {})
        scored: list[tuple[str, float]] = []
        if user_vec is not None:
            for mid, item_vec in factors["movies"].items():
                if mid not in catalog or mid in rated or support.get(mid, 0) < MIN_SUPPORT_MOVIES:
                    continue
                scored.append((mid, sum(a * b for a, b in zip(user_vec, item_vec))))
        scored.sort(key=lambda x: (-x[1], int(x[0])))

        items = [
            RecommendationItem(movie_id=mid, rank=i + 1, algorithm=Algorithm.als,
                               model_version=MODEL_VERSION, score=round(score, 4))
            for i, (mid, score) in enumerate(scored[: request.limit])
        ]
        return RecommendationResult(
            algorithm=Algorithm.als,
            model_version=MODEL_VERSION,
            dataset_version=request.dataset_version,
            split_version=request.split_version,
            runtime_mode=request.runtime_mode,
            generated_at=datetime.now(timezone.utc),
            items=items,
        )
