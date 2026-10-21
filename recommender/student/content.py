"""实验一·基于内容的推荐 Provider.

原理：
  movies.csv 里的 genres 字段用 | 分隔（如 "Adventure|Children|Fantasy"），
  用 genres 构造 Multi-hot 特征；用户画像 = 正反馈（评分 ≥ 4.0）电影
  特征的加权平均；对候选电影算余弦相似度取 Top-10，排除已评分电影。
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ..business import read_history
from ..contracts import Algorithm, RecommendationItem, RecommendationRequest, RecommendationResult
from ..data import FrozenData


STUDENT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = STUDENT_DIR / "content-genres.json"
MANIFEST_PATH = STUDENT_DIR / "content-manifest.json"
MODEL_VERSION = "content-v1"
POSITIVE_THRESHOLD = 4.0


def _ensure_artifact() -> None:
    """首次加载时把 movies.csv 的 genres 映射存为本地 artifact 并校验。"""
    if ARTIFACT_PATH.is_file():
        return
    vectors: dict[str, list[str]] = {}
    for row in FrozenData().rows("movies.csv"):
        mid = row["movie_id"]
        raw = (row.get("genres") or "").strip()
        if not raw or raw == "no genres listed":
            continue
        vectors[mid] = sorted(g for g in raw.split("|") if g.strip())
    ARTIFACT_PATH.write_text(json.dumps(vectors, ensure_ascii=False), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_vectors() -> dict[str, dict[str, float]]:
    """返回 {movie_id: {genre: 1.0, ...}} 稀疏向量字典。"""
    _ensure_artifact()
    raw = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    return {mid: {g: 1.0 for g in genres} for mid, genres in raw.items()}


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(a[k] * b[k] for k in set(a) & set(b))
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class ContentProvider:
    """实验一 Provider：基于 genres 的余弦相似度推荐."""

    model_manifest_path = str(MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        vectors = _load_vectors()
        catalog = FrozenData().catalog()

        # 1) 读取用户历史（business 模式自动从 account_ratings 或 history_ratings 取）
        history = read_history(request)
        positive: dict[str, float] = {}  # movie_id -> 评分（≥4.0）
        for fb in history.ratings:
            try:
                rating = float(fb.rating)
            except (TypeError, ValueError):
                continue
            if rating >= POSITIVE_THRESHOLD:
                positive[fb.movie_id] = rating

        # 2) 聚合用户画像（正反馈特征加权平均）
        profile: dict[str, float] = {}
        for mid, rating in positive.items():
            vec = vectors.get(mid)
            if not vec:
                continue
            for genre, weight in vec.items():
                profile[genre] = profile.get(genre, 0.0) + weight * rating
        if positive:
            scale = 1.0 / max(len(positive), 1)
            profile = {g: v * scale for g, v in profile.items()}

        # 3) 排序候选：排除所有已评分电影 + 无 genres 的电影
        all_rated = {fb.movie_id for fb in history.ratings}
        scored: list[tuple[str, float]] = []
        for mid in catalog:
            if mid in all_rated:
                continue
            vec = vectors.get(mid)
            if not vec:
                continue
            score = _cosine(profile, vec)
            if score > 0:
                scored.append((mid, score))

        # 4) 同分电影按 movie_id 升序（contracts.check_result 强制要求）
        scored.sort(key=lambda x: (-x[1], int(x[0])))

        top = scored[: request.limit]
        items = [
            RecommendationItem(
                movie_id=mid,
                rank=i + 1,
                algorithm=Algorithm.content,
                model_version=MODEL_VERSION,
                score=round(score, 4),
            )
            for i, (mid, score) in enumerate(top)
        ]

        return RecommendationResult(
            algorithm=Algorithm.content,
            model_version=MODEL_VERSION,
            dataset_version=request.dataset_version,
            split_version=request.split_version,
            runtime_mode=request.runtime_mode,
            generated_at=datetime.now(timezone.utc),
            items=items,
        )
