"""实验二·ItemCF Provider.

原理：基于物品的协同过滤（Item-Based Collaborative Filtering）
  1. 从 history_ratings.csv 构建用户-电影评分矩阵
  2. 计算电影之间的余弦相似度（基于用户评分向量）
  3. 对目标用户：找到其高评分电影，聚合相似电影得分
  4. 排除已评分电影，输出 Top-N
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ..business import read_history
from ..contracts import Algorithm, RecommendationItem, RecommendationRequest, RecommendationResult
from ..data import FrozenData


STUDENT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = STUDENT_DIR / "itemcf-similarities.json"
MANIFEST_PATH = STUDENT_DIR / "itemcf-manifest.json"
MODEL_VERSION = "itemcf-v1"
MIN_SUPPORT = 5  # 最小共现支持度


def _build_artifact() -> None:
    """构建电影相似度矩阵（只保留每部电影的 top-50 相似邻居）.

    优化：用倒排索引只计算有共现的电影对，避免 8459² 全量遍历。
    """
    data = FrozenData()
    path = data.package / "business" / "history_ratings.csv"

    # 1. 构建用户-电影评分映射
    user_ratings: dict[str, dict[str, float]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["userId"]
            mid = row["movieId"]
            rating = float(row["rating"])
            user_ratings[uid][mid] = rating

    # 2. 转置：电影 -> {用户: 评分}
    movie_users: dict[str, dict[str, float]] = defaultdict(dict)
    for uid, mids in user_ratings.items():
        for mid, r in mids.items():
            movie_users[mid][uid] = r

    # 3. 计算每部电影的 L2 范数
    movie_norms: dict[str, float] = {}
    for mid, users in movie_users.items():
        movie_norms[mid] = math.sqrt(sum(r * r for r in users.values()))

    # 4. 倒排索引：用户 -> 其评分的电影列表（用于快速找共现）
    user_movies: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for uid, mids in user_ratings.items():
        user_movies[uid] = list(mids.items())

    # 5. 只计算有共现的电影对（通过用户倒排）
    pair_dots: dict[tuple[str, str], float] = defaultdict(float)
    pair_support: dict[tuple[str, str], int] = defaultdict(int)

    for uid, movies in user_movies.items():
        for i, (mid_a, r_a) in enumerate(movies):
            for mid_b, r_b in movies[i + 1:]:
                key = (mid_a, mid_b) if mid_a < mid_b else (mid_b, mid_a)
                pair_dots[key] += r_a * r_b
                pair_support[key] += 1

    # 6. 计算相似度，只保留共现 >= MIN_SUPPORT 的
    similarities: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (mid_a, mid_b), dot in pair_dots.items():
        support = pair_support[(mid_a, mid_b)]
        if support < MIN_SUPPORT:
            continue
        norm_a = movie_norms.get(mid_a, 0)
        norm_b = movie_norms.get(mid_b, 0)
        if norm_a == 0 or norm_b == 0:
            continue
        sim = dot / (norm_a * norm_b)
        if sim > 0:
            similarities[mid_a].append((mid_b, sim))
            similarities[mid_b].append((mid_a, sim))

    # 7. 每部电影只保留 top-50 邻居
    output = {}
    for mid, nbs in similarities.items():
        nbs.sort(key=lambda x: -x[1])
        output[mid] = [{"movie_id": s, "score": round(sc, 4)} for s, sc in nbs[:50]]

    ARTIFACT_PATH.write_text(json.dumps(output, ensure_ascii=False), encoding="utf-8")


@lru_cache(maxsize=1)
def _load_similarities() -> dict[str, list[tuple[str, float]]]:
    if not ARTIFACT_PATH.is_file():
        _build_artifact()
    raw = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    return {mid: [(n["movie_id"], n["score"]) for n in nbs] for mid, nbs in raw.items()}


class ItemCFProvider:
    """实验二·ItemCF：基于物品协同过滤."""

    model_manifest_path = str(MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        sims = _load_similarities()
        catalog = FrozenData().catalog()

        history = read_history(request)
        positive: dict[str, float] = {}
        for fb in history.ratings:
            try:
                r = float(fb.rating)
                if r >= 4.0:
                    positive[fb.movie_id] = r
            except (TypeError, ValueError):
                continue

        # 聚合候选得分：用户喜欢的每部电影 -> 其相似电影加权求和
        candidate_scores: dict[str, float] = defaultdict(float)
        for mid, rating in positive.items():
            for sim_mid, sim_score in sims.get(mid, []):
                candidate_scores[sim_mid] += rating * sim_score

        # 排除已评分
        all_rated = {fb.movie_id for fb in history.ratings}
        candidates = [
            (mid, score) for mid, score in candidate_scores.items()
            if mid not in all_rated and mid in catalog
        ]

        # 同分按 movie_id 升序
        candidates.sort(key=lambda x: (-x[1], int(x[0])))

        items = [
            RecommendationItem(
                movie_id=mid,
                rank=i + 1,
                algorithm=Algorithm.itemcf,
                model_version=MODEL_VERSION,
                score=round(score, 4),
            )
            for i, (mid, score) in enumerate(candidates[:request.limit])
        ]

        return RecommendationResult(
            algorithm=Algorithm.itemcf,
            model_version=MODEL_VERSION,
            dataset_version=request.dataset_version,
            split_version=request.split_version,
            runtime_mode=request.runtime_mode,
            generated_at=datetime.now(timezone.utc),
            items=items,
        )
