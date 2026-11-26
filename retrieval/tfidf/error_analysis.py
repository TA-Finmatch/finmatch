"""TF-IDF + N-gram 오류 분석 (validation 만 사용).

test 의 실패 사례를 보고 설정을 고치면 test 점수가 더 이상 정직하지 않으므로,
분석은 validation 결과 (results/tfidf/tfidf-ngram-validation.jsonl) 로만 한다.

사용 (저장소 루트에서)
    python retrieval/tfidf/error_analysis.py
출력: results/tfidf/error_analysis_validation.md
"""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "v3"
RES = ROOT / "results" / "tfidf"
K = 3


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    corpus = {d["unit_id"]: d for d in read_jsonl(DATA / "corpus.jsonl")}
    # Notion 규칙: 검색 결과의 unit_id 로 products.jsonl 의 상품 (이름 · 가입 구분 · 클래스) 을 찾아 설명한다
    prods = defaultdict(list)
    for p in read_jsonl(DATA / "products.jsonl"):
        prods[p["unit_id"]].append(p)

    def describe(u):
        d = corpus[u]
        ps = prods.get(u, [])
        classes = ", ".join(sorted({p["share_class"] for p in ps if p.get("share_class")})[:4])
        return (f"{u} {d['fund_name']} [{d['investment_type']}, {d['account_type']}, "
                f"상품 {len(ps)}개, 클래스: {classes or '-'}]")
    questions = {q["question_id"]: q for q in read_jsonl(DATA / "validation.jsonl")}
    preds = {r["question_id"]: r["retrieved_unit_ids"]
             for r in read_jsonl(RES / "tfidf-ngram-validation.jsonl")}
    with open(RES / "tfidf-ngram-validation-scores.csv", newline="", encoding="utf-8-sig") as f:
        scores = {r["question_id"]: r for r in csv.DictReader(f)}

    # 검색 문서가 완전히 같은 쌍 (구별이 불가능한 문서)
    by_text = defaultdict(list)
    for uid, d in corpus.items():
        by_text[d["search_text"]].append(uid)
    twin = {uid: len(g) - 1 for g in by_text.values() for uid in g}

    rows = []
    for qid, q in questions.items():
        s = scores[qid]
        refs = set(q["reference_unit_ids"])
        s = {"recall": s["recall@3"]}
        top = preds[qid][:K]
        ref_funds = {corpus[u]["fund_id"] for u in refs}
        top_funds = {corpus[u]["fund_id"] for u in top}
        acc = {corpus[u]["account_type"] for u in refs}
        rows.append(dict(
            qid=qid, qtype=q["query_type"], n_ref=len(refs), recall=float(s["recall"]),
            hit=float(s["recall"]) > 0,
            same_fund_other_unit=(float(s["recall"]) == 0 and bool(ref_funds & top_funds)),
            acc="mixed" if len(acc) > 1 else next(iter(acc)),
            has_twin=any(twin[u] > 0 for u in refs),
            mentions_pension=("연금" in q["question"] or "퇴직" in q["question"]),
            refs=refs, top=top, question=q["question"]))

    out = []
    w = out.append
    n = len(rows)
    w("# 오류 분석: TF-IDF + N-gram (validation, 6,400 질문)\n")
    w(f"전체 Recall@3 = {mean(r['recall'] for r in rows):.4f}; "
      f"정답을 하나도 못 찾은 질문 = {sum(not r['hit'] for r in rows)} ({sum(not r['hit'] for r in rows) / n:.1%})\n")

    def table(title, key, order=None):
        w(f"\n## {title}\n")
        w("| 구분 | 질문 수 | Recall@3 | 하나도 못 찾음 |")
        w("|---|---|---|---|")
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for k in (order or sorted(groups, key=str)):
            g = groups[k]
            w(f"| {k} | {len(g)} | {mean(r['recall'] for r in g):.4f} | {mean(not r['hit'] for r in g):.1%} |")

    table("질문 유형별", "qtype")
    table("정답 단위 수별", "n_ref")
    table("정답 단위의 가입 구분별 (account_type)", "acc")
    table("질문에 '연금' 또는 '퇴직' 이 들어 있는가", "mentions_pension")
    table("정답 문서와 search_text 가 똑같은 다른 문서가 있는가", "has_twin")

    fail = [r for r in rows if not r["hit"]]
    sf = sum(r["same_fund_other_unit"] for r in fail)
    w("\n## 같은 펀드의 다른 단위를 가져온 경우\n")
    w(f"못 찾은 질문 {len(fail)}개 중 {sf}개 ({sf / len(fail):.1%}) 는 top-3 안에 정답과 **같은 펀드 (fund_id)** 의 "
      "다른 단위 (연금 ↔ 일반) 가 있었다. 단위 기준으로는 오답이지만 펀드 기준으로는 맞은 경우다.\n")

    w("\n## 투자유형 (investment_type): 정답 문서 vs 가져온 문서\n")
    w("질문의 '과거 투자 경험' 때문에 엉뚱한 유형을 가져오는지 보기 위해, 정답 문서와 top-3 문서의 투자유형 분포를 비교한다.\n")
    for qt in sorted({r["qtype"] for r in rows}):
        sel = [r for r in rows if r["qtype"] == qt]
        ref_c = Counter(corpus[u]["investment_type"] for r in sel for u in r["refs"])
        top_c = Counter(corpus[u]["investment_type"] for r in sel for u in r["top"])
        nr, nt = sum(ref_c.values()), sum(top_c.values())
        w(f"\n**{qt}** (질문 {len(sel)}개)\n")
        w("| 투자유형 | 정답 문서 비중 | 가져온 문서 비중 |")
        w("|---|---|---|")
        for t, _ in (ref_c + top_c).most_common(6):
            w(f"| {t} | {ref_c[t] / nr:.1%} | {top_c[t] / nt:.1%} |")

    rng = random.Random(0)
    w("\n## 실패 사례 (질문 유형 '상품 비교 추천형', 무작위 3개, seed 0)\n")
    cands = [r for r in fail if r["qtype"] == "상품 비교 추천형"]
    for r in rng.sample(cands, min(3, len(cands))):
        w(f"\n**{r['qid']}** — {r['question'][:260]}...\n")
        w("- 정답:\n" + "\n".join(f"  - {describe(u)}" for u in sorted(r["refs"])))
        w("- 가져온 top-3:\n" + "\n".join(f"  - {describe(u)}" for u in r["top"]))

    path =RES / "error_analysis_validation.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n쓴 파일: {path}")


if __name__ == "__main__":
    main()
