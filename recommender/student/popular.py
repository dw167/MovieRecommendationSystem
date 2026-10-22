"""实验二·热门推荐 Provider.

策略：加权评分（Weighted Rating）避免只按极少量高分排序。
  WR = (v/(v+m)) * R + (m/(v+m)) * C
  v = 该电影评分人数, m = 最小阈值(50), R = 该电影平均分, C = 全数据集平均分
非个性化：对所有用户返回相同榜单。
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from ..contracts import Algorithm, RecommendationItem, RecommendationRequest, RecommendationResult
from ..data import FrozenData


STUDENT_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = STUDENT_DIR / "popular-rankings.json"
MANIFEST_PATH = STUDENT_DIR / "popular-manifest.json"
MODEL_VERSION = "popular-v1"
MIN_VOTES = 50  # 最小评分人数阈值


@lru_cache(maxsize=1)
def _load_rankings() -> list[tuple[str, float]]:
    """返回 [(movie_id, weighted_score), ...] 按 WR 降序."""
    if not ARTIFACT_PATH.is_file():
        _build_artifact()
    raw = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    return [(item["movie_id"], item["score"]) for item in raw]


def _build_artifact() -> None:
    """从 history_ratings.csv 计算加权评分并保存."""
    data = FrozenData()
    path = data.package / "business" / "history_ratings.csv"
    votes: dict[str, list[float]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mid = row["movieId"]
            rating = float(row["rating"])
            votes.setdefault(mid, []).append(rating)

    # 全数据集平均分
    all_ratings = [r for rs in votes.values() for r in rs]
    C = sum(all_ratings) / len(all_ratings) if all_ratings else 3.0

    scored: list[dict] = []
    for mid, rs in votes.items():
        v = len(rs)
        R = sum(rs) / v
        wr = (v / (v + MIN_VOTES)) * R + (MIN_VOTES / (v + MIN_VOTES)) * C
        scored.append({"movie_id": mid, "score": round(wr, 4), "votes": v, "avg": round(R, 2)})

    scored.sort(key=lambda x: (-x["score"], int(x["movie_id"])))
    ARTIFACT_PATH.write_text(json.dumps(scored, ensure_ascii=False), encoding="utf-8")


class PopularProvider:
    """实验二·热门推荐：加权评分榜单，非个性化."""

    model_manifest_path = str(MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        rankings = _load_rankings()
        catalog = FrozenData().catalog()

        items = []
        for mid, score in rankings:
            if len(items) >= request.limit:
                break
            if mid in catalog:
                items.append(RecommendationItem(
                    movie_id=mid,
                    rank=len(items) + 1,
                    algorithm=Algorithm.popular,
                    model_version=MODEL_VERSION,
                    score=round(score, 4),
                ))

        return RecommendationResult(
            algorithm=Algorithm.popular,
            model_version=MODEL_VERSION,
            dataset_version=request.dataset_version,
            split_version=request.split_version,
            runtime_mode=request.runtime_mode,
            generated_at=datetime.now(timezone.utc),
            items=items,
        )
