"""实验二·新用户冷启动 Provider.

策略：
  - 无评分/极少评分（< 3 条）→ 热门榜单兜底，明确标注非个性化
  - 有评分 → 委托 content 算法（基于内容）产生候选
  关键：界面必须标明推荐策略，不能把热门榜单伪称为个性化结果。
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..business import read_history
from ..contracts import Algorithm, RecommendationRequest, RecommendationResult
from .content import ContentProvider
from .popular import PopularProvider


STUDENT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = STUDENT_DIR / "cold-start-manifest.json"
MODEL_VERSION = "cold-start-v1"
MIN_RATINGS_FOR_PERSONALIZATION = 3


class ColdStartProvider:
    """实验二·冷启动：无评分用热门，有评分用内容推荐."""

    model_manifest_path = str(MANIFEST_PATH)

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        history = read_history(request)
        ratings_count = len(history.ratings)

        if ratings_count < MIN_RATINGS_FOR_PERSONALIZATION:
            # 无足够画像：热门榜单兜底（非个性化），recall_sources 明确标注策略来源
            popular = PopularProvider()
            result = popular.recommend(request)
            strategy = "popular-fallback"
        else:
            # 有足够评分：委托内容推荐（个性化）
            content = ContentProvider()
            result = content.recommend(request)
            strategy = "content-personalized"

        for item in result.items:
            item.algorithm = Algorithm.cold_start
            item.model_version = MODEL_VERSION
            item.recall_sources = [strategy]
        result.algorithm = Algorithm.cold_start
        result.model_version = MODEL_VERSION
        return result
