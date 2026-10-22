"""实验三·Spark MLlib ALS 显式评分训练（用 spark-submit 运行，不在 HTTP 进程内训练）。

用法（apache/spark 容器内）:
  spark-submit --master local[4] scripts/train_als_spark.py \
    --business data/packages/<ds>/business/history_ratings.csv \
    --train data/packages/<ds>/evaluation/train.csv \
    --out recommender/student

产出:
  als-factors-business.json  business 模型因子 {rank, users, movies}
  als-factors-eval.json      evaluation 模型因子（只拟合冻结 train.csv）
  als-eval-top60.json        evaluation 模型对每个训练用户的 top60（未做排除过滤）
  als-train-summary.json     行数/用户数/电影数/Spark 版本等摘要
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

RANK = 16
MAX_ITER = 12
REG_PARAM = 0.1
SEED = 42


def train(spark: SparkSession, csv_path: str):
    df = (spark.read.option("header", "true").csv(csv_path)
          .selectExpr("CAST(userId AS INT) AS user", "CAST(movieId AS INT) AS item",
                      "CAST(rating AS FLOAT) AS rating"))
    rows = df.count()
    users = df.select("user").distinct().count()
    items = df.select("item").distinct().count()
    als = ALS(rank=RANK, maxIter=MAX_ITER, regParam=REG_PARAM, seed=SEED,
              userCol="user", itemCol="item", ratingCol="rating",
              implicitPrefs=False, coldStartStrategy="drop")
    started = time.time()
    model = als.fit(df)
    seconds = round(time.time() - started, 1)
    support = {str(r[0]): int(r[1]) for r in df.groupBy("item").count().collect()}
    return model, {"rows": rows, "users": users, "items": items, "fit_seconds": seconds}, support


def dump_factors(model, path: Path, support: dict) -> None:
    users = {str(r.id): [round(float(v), 6) for v in r.features] for r in model.userFactors.collect()}
    movies = {str(r.id): [round(float(v), 6) for v in r.features] for r in model.itemFactors.collect()}
    path.write_text(json.dumps({"rank": RANK, "users": users, "movies": movies, "support": support},
                               ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--business", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spark = (SparkSession.builder.appName("cinelab-als-lab3")
             .config("spark.sql.shuffle.partitions", "8").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")

    biz_model, biz_stats, biz_support = train(spark, args.business)
    dump_factors(biz_model, out / "als-factors-business.json", biz_support)

    eval_model, eval_stats, eval_support = train(spark, args.train)
    dump_factors(eval_model, out / "als-factors-eval.json", eval_support)

    top60 = {}
    for row in eval_model.recommendForAllUsers(60).collect():
        top60[str(row.user)] = [[int(rec[0]), round(float(rec[1]), 6)] for rec in row.recommendations]
    (out / "als-eval-top60.json").write_text(json.dumps(top60, ensure_ascii=False), encoding="utf-8")

    summary = {"framework": "spark-mllib-als", "spark_version": spark.sparkContext.version,
               "parameters": {"rank": RANK, "maxIter": MAX_ITER, "regParam": REG_PARAM,
                              "seed": SEED, "implicitPrefs": False, "coldStartStrategy": "drop"},
               "business": biz_stats, "evaluation": eval_stats,
               "eval_top60_users": len(top60)}
    (out / "als-train-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ALS-TRAIN-OK", json.dumps(summary))
    spark.stop()


if __name__ == "__main__":
    main()
