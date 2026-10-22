"""实验三·统一离线评价管线（独立进程，无数据库连接）。

流程：
  1. 只用冻结 evaluation/train.csv 拟合四个算法的 evaluation 工件/清单
     （popular 加权榜、content 类型画像、itemcf 共现相似、als 读 Spark 因子工件）；
  2. 对 validation 分区固定 4209 名评价用户逐人生成 Top-10 预测（candidates = catalog - train 已评分），
     无结果用户提交 items=[]，不悄悄剔除；
  3. 用 recommender.evaluation.check_predictions 校验全部预测的契约与候选合法性；
  4. 以 validation.csv 中评分 >=4.0 为正例，宏观平均计算 Precision@10 / Recall@10；
  5. 生成 EvaluationResult 并用 check_evaluation_result 校验，落到 student/eval-results/。

用法（项目根目录）: python scripts/lab3_evaluate.py
"""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import evaluation as ev  # noqa: E402
from recommender.contracts import Algorithm, RecommendationItem, RecommendationResult  # noqa: E402
from recommender.data import DATASET, SCHEMA, SPLIT, MANIFEST_SHA, FrozenData, sha256  # noqa: E402

STUDENT = ROOT / "recommender" / "student"
EVAL_OUT = ROOT / "eval-out"
RESULTS = STUDENT / "eval-results"
PARTITION = "validation"
POSITIVE = 4.0
MIN_VOTES = 50
MIN_SUPPORT = 5
MIN_SUPPORT_MOVIES = 25  # ALS Top-N 候选过滤，与 business Provider 同一定义
TOP_NEIGHBORS = 50
PYTHON_VER = platform.python_version()


def load_train():
    ratings = defaultdict(list)
    for row in FrozenData().training():
        ratings[row["userId"]].append((row["movieId"], float(row["rating"])))
    return ratings


def build_popular_eval(train):
    votes = defaultdict(list)
    for rows in train.values():
        for mid, r in rows:
            votes[mid].append(r)
    all_r = [r for rs in votes.values() for r in rs]
    c = sum(all_r) / len(all_r)
    scored = []
    for mid, rs in votes.items():
        v = len(rs)
        wr = (v / (v + MIN_VOTES)) * (sum(rs) / v) + (MIN_VOTES / (v + MIN_VOTES)) * c
        scored.append({"movie_id": mid, "score": round(wr, 4), "votes": v})
    scored.sort(key=lambda x: (-x["score"], int(x["movie_id"])))
    path = STUDENT / "popular-rankings-eval.json"
    path.write_text(json.dumps(scored, ensure_ascii=False), encoding="utf-8")
    return [(it["movie_id"], it["score"]) for it in scored]


def build_itemcf_eval(train):
    movie_users = defaultdict(dict)
    for uid, rows in train.items():
        for mid, r in rows:
            movie_users[mid][uid] = r
    norms = {mid: math.sqrt(sum(r * r for r in users.values())) for mid, users in movie_users.items()}
    pair_dot, pair_sup = defaultdict(float), defaultdict(int)
    for rows in train.values():
        rmap = dict(rows)
        mids = [m for m, _ in rows]
        for i, a in enumerate(mids):
            for b in mids[i + 1:]:
                key = (a, b) if a < b else (b, a)
                pair_dot[key] += rmap[a] * rmap[b]
                pair_sup[key] += 1
    sims = defaultdict(list)
    for (a, b), dot in pair_dot.items():
        if pair_sup[(a, b)] < MIN_SUPPORT or norms[a] == 0 or norms[b] == 0:
            continue
        sim = dot / (norms[a] * norms[b])
        if sim > 0:
            sims[a].append((b, sim))
            sims[b].append((a, sim))
    out = {}
    for mid, nbs in sims.items():
        nbs.sort(key=lambda x: (-x[1], int(x[0])))
        out[mid] = [{"movie_id": s, "score": round(sc, 4)} for s, sc in nbs[:TOP_NEIGHBORS]]
    path = STUDENT / "itemcf-similarities-eval.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return {mid: [(n["movie_id"], n["score"]) for n in nbs] for mid, nbs in out.items()}


def load_content_index():
    genres, order = {}, []
    for row in FrozenData().rows("movies.csv"):
        raw = (row.get("genres") or "").strip()
        if not raw or raw == "no genres listed":
            continue
        genres[row["movie_id"]] = sorted(g for g in raw.split("|") if g.strip())
        order.append(row["movie_id"])
    idx = defaultdict(list)
    for i, mid in enumerate(order):
        for g in genres[mid]:
            idx[g].append(i)
    norms = [math.sqrt(len(genres[mid])) for mid in order]
    return order, genres, idx, norms


def write_manifest(name, algorithm, version, mode, training_file, artifacts, parameters, tool_versions, feature_files=None):
    data = FrozenData()
    manifest = {
        "algorithm": algorithm, "model_version": version, "runtime_mode": mode,
        "dataset_version": DATASET, "split_version": SPLIT, "schema_version": SCHEMA,
        "manifest_sha256": MANIFEST_SHA,
        "training_files": {training_file: data.manifest["files"][training_file]["sha256"]},
        "feature_files": feature_files or {},
        "artifacts": {n: sha256(STUDENT / n) for n in artifacts},
        "parameters": parameters,
        "tool_versions": tool_versions,
    }
    path = STUDENT / name
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def top10(scored, rated, limit=10):
    scored = [(round(s, 4), mid) for mid, s in scored if mid not in rated]
    scored.sort(key=lambda x: (-x[0], int(x[1])))
    return [(mid, s) for s, mid in scored[:limit]]


def main():
    EVAL_OUT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    data = FrozenData()
    train = load_train()
    movie_counts = defaultdict(int)
    for rows in train.values():
        for mid, _ in rows:
            movie_counts[mid] += 1
    catalog = data.catalog()
    eligible = ev.eligible_users(data, PARTITION)
    task_sha = ev.task(data, PARTITION)["task_sha256"]

    print("[1/5] 构建 evaluation 工件 …")
    popular_rank = build_popular_eval(train)
    itemcf_sims = build_itemcf_eval(train)
    order, genres, genre_idx, genre_norm = load_content_index()
    pos_index = {mid: i for i, mid in enumerate(order)}
    als_factors = json.loads((STUDENT / "als-factors-eval.json").read_text(encoding="utf-8"))
    als_top60 = json.loads((STUDENT / "als-eval-top60.json").read_text(encoding="utf-8"))
    summary = json.loads((STUDENT / "als-train-summary.json").read_text(encoding="utf-8"))
    spark_tools = {"spark": summary["spark_version"], "python": PYTHON_VER}
    py_tools = {"python": PYTHON_VER}

    print("[2/5] 写 evaluation / business 清单 …")
    manifests = {
        "popular": write_manifest("popular-eval-manifest.json", "popular", "popular-eval-v1", "evaluation",
                                  "evaluation/train.csv", ["popular-rankings-eval.json"],
                                  {"min_votes": MIN_VOTES, "weighting": "WR"}, py_tools),
        "content": write_manifest("content-eval-manifest.json", "content", "content-eval-v1", "evaluation",
                                  "evaluation/train.csv", ["content-genres.json"],
                                  {"feature": "genres-multi-hot", "similarity": "cosine", "positive_threshold": POSITIVE},
                                  py_tools, {"movies.csv": data.manifest["files"]["movies.csv"]["sha256"]}),
        "itemcf": write_manifest("itemcf-eval-manifest.json", "itemcf", "itemcf-eval-v1", "evaluation",
                                 "evaluation/train.csv", ["itemcf-similarities-eval.json"],
                                 {"min_support": MIN_SUPPORT, "top_neighbors": TOP_NEIGHBORS,
                                  "positive_threshold": POSITIVE}, py_tools),
        "als": write_manifest("als-eval-manifest.json", "als", "als-eval-v1", "evaluation",
                              "evaluation/train.csv", ["als-factors-eval.json", "als-eval-top60.json"],
                              {**summary["parameters"], "min_support_movies": MIN_SUPPORT_MOVIES}, spark_tools),
    }
    write_manifest("als-manifest.json", "als", "als-v1", "business",
                   "business/history_ratings.csv", ["als-factors-business.json"],
                   {**summary["parameters"], "min_support_movies": MIN_SUPPORT_MOVIES}, spark_tools)

    positives = defaultdict(set)
    for row in data.rows("evaluation/validation.csv", evaluation=True):
        if float(row["rating"]) >= POSITIVE:
            positives[row["userId"]].add(row["movieId"])

    def predict(alg, uid):
        rated = {m for m, _ in train.get(uid, [])}
        if alg == "popular":
            picked = []
            for mid, score in popular_rank:
                if mid in catalog and mid not in rated:
                    picked.append((mid, score))
                    if len(picked) >= 10:
                        break
            return picked
        if alg == "content":
            profile = defaultdict(float)
            for mid, r in train.get(uid, []):
                if r >= POSITIVE and mid in pos_index:
                    for g in genres[mid]:
                        profile[g] += r
            if not profile:
                return []
            dot = defaultdict(float)
            for g, w in profile.items():
                for i in genre_idx[g]:
                    mid = order[i]
                    if mid not in rated:
                        dot[i] += w
            pn = math.sqrt(sum(w * w for w in profile.values()))
            scored = [(order[i], d / (pn * genre_norm[i])) for i, d in dot.items()]
            return top10(scored, rated)
        if alg == "itemcf":
            scores = defaultdict(float)
            for mid, r in train.get(uid, []):
                if r < POSITIVE:
                    continue
                for nb, sim in itemcf_sims.get(mid, []):
                    if nb not in rated:
                        scores[nb] += r * sim
            return top10(list(scores.items()), rated)
        scored = [(str(mid), s) for mid, s in als_top60.get(uid, [])
                  if str(mid) in catalog and movie_counts.get(str(mid), 0) >= MIN_SUPPORT_MOVIES]
        return top10(scored, rated)

    print("[3/5] 生成 4209 用户 × 4 算法预测并校验契约 …")
    metrics = {}
    for alg in ["popular", "content", "itemcf", "als"]:
        path = EVAL_OUT / f"{alg}-predictions-{PARTITION}.jsonl"
        hits_sum = p_sum = r_sum = 0.0
        with path.open("w", encoding="utf-8") as stream:
            for uid in sorted(eligible, key=int):
                picked = predict(alg, uid)
                dto = RecommendationResult(
                    algorithm=Algorithm(alg), model_version=f"{alg}-eval-v1",
                    dataset_version=DATASET, split_version=SPLIT, runtime_mode="evaluation",
                    generated_at=datetime.now(timezone.utc),
                    items=[RecommendationItem(movie_id=mid, rank=i + 1, algorithm=Algorithm(alg),
                                              model_version=f"{alg}-eval-v1", score=round(s, 4))
                           for i, (mid, s) in enumerate(picked)])
                stream.write(json.dumps({"ml_user_id": uid, "result": json.loads(dto.model_dump_json())},
                                        ensure_ascii=False) + "\n")
                pos = positives.get(uid, set())
                hits = len({mid for mid, _ in picked} & pos)
                p_sum += hits / 10
                r_sum += hits / len(pos) if pos else 0.0
                hits_sum += hits
        ev.check_predictions(path, manifests[alg], PARTITION, data)
        n = len(eligible)
        metrics[alg] = {"precision_at_10": round(p_sum / n, 6), "recall_at_10": round(r_sum / n, 6),
                        "users": n, "hits": int(hits_sum)}
        print(f"  {alg}: P@10={metrics[alg]['precision_at_10']:.4f} R@10={metrics[alg]['recall_at_10']:.4f}")

    print("[4/5] 写 EvaluationResult 并校验 …")
    for alg, manifest in manifests.items():
        result = ev.EvaluationResult(
            status="completed", runtime_mode="evaluation", algorithm=alg,
            model_version=f"{alg}-eval-v1", task_sha256=task_sha,
            model_manifest_sha256=sha256(manifest), dataset_version=DATASET,
            split_version=SPLIT, schema_version=SCHEMA, partition=PARTITION,
            user_count=len(eligible), precision_at_10=metrics[alg]["precision_at_10"],
            recall_at_10=metrics[alg]["recall_at_10"])
        out = RESULTS / f"{alg}-{PARTITION}.json"
        out.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        ev.check_evaluation_result(out, manifest, data)

    print("[5/5] 完成。评价表：")
    (EVAL_OUT / "evaluation-table.json").write_text(
        json.dumps({"partition": PARTITION, "user_count": len(eligible), **metrics}, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
