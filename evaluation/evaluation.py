"""공통 평가 코드 (팀원 요청 7절, T12).

검색 모델마다 낸 결과 파일을 같은 기준으로 채점한다. 표준 라이브러리만 쓴다.

채점 정의 (unit_id 기준)
------------------------
- 질문 하나의 정답 = reference_unit_ids (집합으로 본다).
- 결과 = retrieved_unit_ids 배열. 배열 순서가 검색 순위 (1위부터).
- top-k = 결과 배열의 앞 k 칸. k 기본 3.
- 같은 unit_id 가 결과에 여러 번 나오면 처음 것만 센다.
  뒤에 나온 중복은 순위 칸을 차지한 채 0점이다 (중복을 지우고 뒤를 당겨 오지 않는다).
  예: 정답 {A,B}, 결과 [A,A,B], k=3 이면 A 는 1위, B 는 3위.
- Recall@k = (top-k 안의 정답 단위 수) / (정답 단위 수).
  분모는 정답 수 그대로다. 정답이 k 보다 많으면 Recall@k 는 1 이 될 수 없다.
- NDCG@k: 정답 관련도 1, 오답 0.
  DCG  = Σᵢ₌₁ᵏ relᵢ / log₂(i+1)   (i 는 1부터)
  IDCG = 정답 min(정답 수, k) 개가 맨 위에 있을 때의 DCG
  NDCG = DCG / IDCG
- 전체 점수 = 채점 대상 질문마다 낸 점수의 평균.
  채점 대상 = 질문 파일의 질문 중 needs_review (기본 제외) 와 정답이 빈 질문을 뺀 것.
  결과 파일에 없는 질문은 0점으로 평균에 넣는다. query_type 별 평균도 같은 규칙.

사용
----
python evaluation/evaluation.py --questions data/processed/test.jsonl --predictions run.jsonl \
    [--k 3] [--corpus data/processed/corpus.jsonl] [--per-question out.csv] [--include-needs-review]
python evaluation/evaluation.py --self-test
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict


# ── 질문 하나 채점 ──────────────────────────────────────────

def _top_k_hits(retrieved, relevant, k):
    """top-k 칸마다 관련도 (1/0) 목록. 중복은 처음 것만 1, 뒤는 0 (칸은 차지)."""
    seen = set()
    rels = []
    for uid in list(retrieved)[:k]:
        if uid in seen:
            rels.append(0)
            continue
        seen.add(uid)
        rels.append(1 if uid in relevant else 0)
    return rels


def recall_at_k(retrieved, relevant, k=3):
    """Recall@k = top-k 안의 정답 수 / 정답 수. 정답이 비면 ValueError."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("정답 단위가 비어 있어 Recall 을 계산할 수 없다")
    if k < 1:
        raise ValueError("k 는 1 이상이어야 한다")
    return sum(_top_k_hits(retrieved, relevant, k)) / len(relevant)


def ndcg_at_k(retrieved, relevant, k=3):
    """NDCG@k (이진 관련도). 정답이 비면 ValueError."""
    relevant = set(relevant)
    if not relevant:
        raise ValueError("정답 단위가 비어 있어 NDCG 를 계산할 수 없다")
    if k < 1:
        raise ValueError("k 는 1 이상이어야 한다")
    rels = _top_k_hits(retrieved, relevant, k)
    dcg = sum(r / math.log2(i + 1) for i, r in enumerate(rels, start=1))
    idcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / idcg


# ── 파일 읽기 ──────────────────────────────────────────────

def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # 빈 줄은 건너뜀
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} JSON 이 아니다: {e}") from e
    return rows


def load_questions(path):
    return _read_jsonl(path)


def load_predictions(path):
    """{question_id: retrieved_unit_ids}. 같은 question_id 가 또 나오면 첫 줄을 쓴다."""
    preds = {}
    dup_qids = []
    for row in _read_jsonl(path):
        qid = row["question_id"]
        if qid in preds:
            dup_qids.append(qid)
            continue
        preds[qid] = list(row.get("retrieved_unit_ids") or [])
    return preds, dup_qids


def load_corpus_ids(path):
    return {row["unit_id"] for row in _read_jsonl(path)}


# ── 전체 채점 ──────────────────────────────────────────────

def evaluate(questions, predictions, k=3, corpus_ids=None,
             include_needs_review=False, duplicate_prediction_qids=None):
    """질문 목록과 {question_id: retrieved_unit_ids} 를 받아 결과 dict 를 돌려준다.

    questions: 질문 dict 목록 (question_id, reference_unit_ids, mapping_status, query_type)
    predictions: {question_id: [unit_id, ...]}
    corpus_ids: corpus 의 unit_id 집합 (주면 없는 unit_id 를 센다)
    """
    if k < 1:
        raise ValueError("k 는 1 이상이어야 한다")

    question_ids = set()
    per_question = []
    skipped_needs_review = 0
    skipped_empty_reference = []
    missing_predictions = 0
    duplicate_units_in_topk = 0  # top-k 안에서 중복으로 0점 처리된 칸 수

    for q in questions:
        qid = q["question_id"]
        question_ids.add(qid)
        if q.get("mapping_status") == "needs_review" and not include_needs_review:
            skipped_needs_review += 1
            continue
        relevant = set(q.get("reference_unit_ids") or [])
        if not relevant:
            skipped_empty_reference.append(qid)
            continue

        if qid in predictions:
            retrieved = predictions[qid]
            in_predictions = True
        else:
            retrieved = []
            in_predictions = False
            missing_predictions += 1

        top = retrieved[:k]
        duplicate_units_in_topk += len(top) - len(set(top))
        per_question.append({
            "question_id": qid,
            "query_type": q.get("query_type", ""),
            "n_reference": len(relevant),
            "in_predictions": in_predictions,
            "recall": recall_at_k(retrieved, relevant, k),
            "ndcg": ndcg_at_k(retrieved, relevant, k),
        })

    # 질문 파일에 없는 question_id
    unknown_qids = sorted(set(predictions) - question_ids)

    # corpus 에 없는 unit_id (결과 전체 기준, 서로 다른 id 수와 등장 횟수)
    unknown_units = set()
    unknown_unit_occurrences = 0
    if corpus_ids is not None:
        for uids in predictions.values():
            for uid in uids:
                if uid not in corpus_ids:
                    unknown_units.add(uid)
                    unknown_unit_occurrences += 1

    n = len(per_question)
    by_type = defaultdict(list)
    for r in per_question:
        by_type[r["query_type"]].append(r)

    def _mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else float("nan")

    return {
        "k": k,
        "n_questions_in_file": len(questions),
        "n_scored": n,
        "mean_recall": _mean(per_question, "recall"),
        "mean_ndcg": _mean(per_question, "ndcg"),
        "by_query_type": {
            t: {"n": len(rows), "mean_recall": _mean(rows, "recall"),
                "mean_ndcg": _mean(rows, "ndcg")}
            for t, rows in sorted(by_type.items())
        },
        "skipped_needs_review": skipped_needs_review,
        "skipped_empty_reference": skipped_empty_reference,
        "missing_predictions": missing_predictions,
        "unknown_question_ids": unknown_qids,
        "unknown_unit_ids": sorted(unknown_units),
        "unknown_unit_occurrences": unknown_unit_occurrences,
        "duplicate_units_in_topk": duplicate_units_in_topk,
        "duplicate_prediction_qids": list(duplicate_prediction_qids or []),
        "per_question": per_question,
    }


# ── 출력 ──────────────────────────────────────────────────

def _warn(msg):
    print(f"[경고] {msg}", file=sys.stderr)


def report(res, include_needs_review=False):
    k = res["k"]
    # 경고는 stderr
    if res["missing_predictions"]:
        _warn(f"결과 파일에 없는 질문 {res['missing_predictions']}개 → 0점으로 평균에 넣음")
    if res["unknown_question_ids"]:
        ex = ", ".join(res["unknown_question_ids"][:5])
        _warn(f"질문 파일에 없는 question_id {len(res['unknown_question_ids'])}개 (채점 안 함): {ex}")
    if res["unknown_unit_ids"]:
        ex = ", ".join(res["unknown_unit_ids"][:5])
        _warn(f"corpus 에 없는 unit_id {len(res['unknown_unit_ids'])}종 "
              f"(등장 {res['unknown_unit_occurrences']}번): {ex}")
    if res["duplicate_prediction_qids"]:
        ex = ", ".join(res["duplicate_prediction_qids"][:5])
        _warn(f"결과 파일에서 같은 question_id 가 또 나온 줄 {len(res['duplicate_prediction_qids'])}개 "
              f"(첫 줄만 씀): {ex}")
    if res["duplicate_units_in_topk"]:
        _warn(f"top-{k} 안의 중복 unit_id {res['duplicate_units_in_topk']}칸 → 처음 것만 셈")
    if res["skipped_empty_reference"]:
        _warn(f"정답이 빈 질문 {len(res['skipped_empty_reference'])}개 → 채점에서 뺌")

    # 결과는 stdout
    print(f"질문 파일 질문 수: {res['n_questions_in_file']}")
    if include_needs_review:
        print(f"needs_review 질문: 포함 (옵션)")
    else:
        print(f"needs_review 로 뺀 질문 수: {res['skipped_needs_review']}")
    print(f"채점한 질문 수: {res['n_scored']} (결과 없음 0점 {res['missing_predictions']}개 포함)")
    print(f"평균 Recall@{k}: {res['mean_recall']:.4f}")
    print(f"평균 NDCG@{k}:   {res['mean_ndcg']:.4f}")
    print("query_type 별:")
    for t, v in res["by_query_type"].items():
        print(f"  {t} (n={v['n']}): Recall@{k} {v['mean_recall']:.4f} · NDCG@{k} {v['mean_ndcg']:.4f}")


def write_per_question_csv(res, path):
    k = res["k"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "query_type", "n_reference", "in_predictions",
                    f"recall@{k}", f"ndcg@{k}"])
        for r in res["per_question"]:
            w.writerow([r["question_id"], r["query_type"], r["n_reference"],
                        int(r["in_predictions"]), f"{r['recall']:.6f}", f"{r['ndcg']:.6f}"])


# ── 손계산 시험 ────────────────────────────────────────────

def self_test():
    close = lambda a, b: math.isclose(a, b, abs_tol=1e-9)
    L = math.log2
    cases = [
        # (설명, 결과, 정답, k, 기대 Recall, 기대 NDCG)
        ("2위가 오답", ["A", "X", "B"], ["A", "B"], 3,
         1.0, (1 + 1 / L(4)) / (1 + 1 / L(3))),                # ≈ 0.9197
        ("정답 4개 중 3개 맞힘 (분모 = 정답 수)", ["A", "B", "C"], ["A", "B", "C", "D"], 3,
         0.75, 1.0),                                           # IDCG 는 min(4,3)=3개
        ("전부 오답", ["X", "Y", "Z"], ["A"], 3, 0.0, 0.0),
        ("빈 결과", [], ["A", "B"], 3, 0.0, 0.0),
        ("중복은 자리를 차지한 채 0점", ["A", "A", "B"], ["A", "B"], 3,
         1.0, (1 + 1 / L(4)) / (1 + 1 / L(3))),                # 당겨 오면 1.0 이 됨
        ("중복 때문에 정답이 top-k 밖으로", ["A", "A", "B"], ["A", "B"], 2,
         0.5, 1 / (1 + 1 / L(3))),                             # ≈ 0.6131
        ("정답 1개가 3위", ["X", "Y", "A"], ["A"], 3, 1.0, 1 / L(4)),   # 0.5
    ]
    ok = True
    for name, ret, rel, k, er, en in cases:
        r, n = recall_at_k(ret, rel, k), ndcg_at_k(ret, rel, k)
        good = close(r, er) and close(n, en)
        ok &= good
        print(f"[{'통과' if good else '실패'}] {name}: Recall@{k}={r:.4f} (기대 {er:.4f}), "
              f"NDCG@{k}={n:.4f} (기대 {en:.4f})")

    # evaluate() 전체 흐름: 결과 없음 0점 · needs_review 제외 · 모르는 id 경고
    qs = [
        {"question_id": "Q1", "reference_unit_ids": ["A", "B"], "mapping_status": "matched", "query_type": "t1"},
        {"question_id": "Q2", "reference_unit_ids": ["C"], "mapping_status": "matched", "query_type": "t2"},
        {"question_id": "Q3", "reference_unit_ids": ["D"], "mapping_status": "needs_review", "query_type": "t2"},
    ]
    preds = {"Q1": ["A", "B", "Z"], "Q9": ["A"], "Q3": ["D"]}  # Q2 없음, Q9 모름, Z 는 corpus 밖
    res = evaluate(qs, preds, k=3, corpus_ids={"A", "B", "C", "D"})
    checks = [
        ("채점 질문 수 2 (needs_review 1개 뺌)", res["n_scored"] == 2 and res["skipped_needs_review"] == 1),
        ("결과 없는 질문 1개, 평균 Recall 0.5", res["missing_predictions"] == 1 and close(res["mean_recall"], 0.5)),
        ("모르는 question_id 1개", res["unknown_question_ids"] == ["Q9"]),
        ("corpus 에 없는 unit_id 1개", res["unknown_unit_ids"] == ["Z"]),
        ("t1 평균 NDCG 1.0, t2 평균 0.0",
         close(res["by_query_type"]["t1"]["mean_ndcg"], 1.0) and close(res["by_query_type"]["t2"]["mean_ndcg"], 0.0)),
    ]
    res2 = evaluate(qs, preds, k=3, include_needs_review=True)
    checks.append(("--include-needs-review 면 3개 채점, 평균 Recall 2/3",
                   res2["n_scored"] == 3 and close(res2["mean_recall"], 2 / 3)))
    for name, good in checks:
        ok &= good
        print(f"[{'통과' if good else '실패'}] evaluate: {name}")

    print("self-test", "전부 통과" if ok else "실패 있음")
    return ok


# ── 명령줄 ────────────────────────────────────────────────

def main(argv=None):
    p = argparse.ArgumentParser(description="검색 결과를 unit_id 기준 Recall@k · NDCG@k 로 채점한다.")
    p.add_argument("--questions", help="질문 JSONL (validation.jsonl / test.jsonl)")
    p.add_argument("--predictions", help="모델 결과 JSONL (question_id, retrieved_unit_ids)")
    p.add_argument("--k", type=int, default=3, help="top-k (기본 3)")
    p.add_argument("--corpus", help="corpus.jsonl (주면 corpus 에 없는 unit_id 를 센다)")
    p.add_argument("--per-question", help="질문별 점수 CSV 저장 경로")
    p.add_argument("--include-needs-review", action="store_true",
                   help="mapping_status 가 needs_review 인 질문도 채점")
    p.add_argument("--self-test", action="store_true", help="손계산 예로 시험")
    a = p.parse_args(argv)

    if a.self_test:
        return 0 if self_test() else 1
    if not a.questions or not a.predictions:
        p.error("--questions 와 --predictions 가 필요하다 (또는 --self-test)")
    if a.k < 1:
        p.error("--k 는 1 이상이어야 한다")

    questions = load_questions(a.questions)
    predictions, dup_qids = load_predictions(a.predictions)
    corpus_ids = load_corpus_ids(a.corpus) if a.corpus else None
    res = evaluate(questions, predictions, k=a.k, corpus_ids=corpus_ids,
                   include_needs_review=a.include_needs_review,
                   duplicate_prediction_qids=dup_qids)
    report(res, include_needs_review=a.include_needs_review)
    if a.per_question:
        write_per_question_csv(res, a.per_question)
        print(f"질문별 점수 CSV: {a.per_question}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
