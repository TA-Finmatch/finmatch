"""TF-IDF 와 TF-IDF + N-gram 검색 (Notion "TF-IDF, TF-IDF + N-gram" 규칙).

규칙 요약
- 검색 후보: corpus.jsonl 3,930개 전체, 상품 필터 없음.
- 입력: 질문의 `question`. `reference_unit_ids` 는 채점에서만 쓴다 (검색 함수는 읽지도 않는다).
- 토큰화: 팀 공통 모듈 retrieval/common/kiwi_tokenizer.py 를 그대로 쓴다 (규칙은 거기서만 바꾼다).
  문서 (`search_text`) 와 질문 (`question`) 을 같은 함수로 자른다.
- TF-IDF 는 corpus 의 `search_text` 에만 fit 하고, 질문은 같은 벡터화기로 변환만 한다.
- 출력: 질문마다 서로 다른 unit_id 상위 3개. 한 줄에 {"question_id", "retrieved_unit_ids"}.
- 설정은 validation 의 NDCG@3 으로만 고른다. test 는 `final` 로 설정을 고정한 뒤 한 번만 돌린다.
- 두 모델: "TF-IDF" (n-gram 1 ~ 1) 와 "TF-IDF + N-gram" (n-gram 1 ~ 2, 3, 4).

사용 (저장소 루트에서)
    python retrieval/tfidf/tfidf.py grid     # validation 에서 설정 탐색 → results/tfidf/
    python retrieval/tfidf/tfidf.py final    # best_configs.json 의 설정으로 test 결과 JSONL 생성 (한 번만)
    python retrieval/tfidf/tfidf.py summary  # 저장된 예측 JSONL 을 채점해 평균 점수 표 (summary.csv · summary.md) 를 쓴다
"""

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "retrieval" / "common"))
import evaluation as ev  # noqa: E402  공통 평가 코드 (팀 공용)
import kiwi_tokenizer as kt  # noqa: E402  공통 토큰화 (팀 공용)

DATA = ROOT / "data" / "processed" / "v3"
OUT = ROOT / "results" / "tfidf"
CACHE_DIR = ROOT / "scratch" / "cache" / "tokens"  # scratch/ 는 .gitignore 대상
K = 3
SLUG = {"TF-IDF": "tfidf", "TF-IDF + N-gram": "tfidf-ngram"}  # Notion 이 요구하는 두 모델 → 파일 이름 앞부분


# ── 입출력 ──────────────────────────────────────────────────
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all():
    corpus = read_jsonl(DATA / "corpus.jsonl")
    val = read_jsonl(DATA / "validation.jsonl")
    test = read_jsonl(DATA / "test.jsonl")
    return corpus, val, test


def write_environment():
    """실행 환경과 토큰화 규칙의 지문을 results/tfidf/environment.txt 에 남긴다."""
    import platform
    from importlib.metadata import version

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"python {platform.python_version()}"]
    lines += [f"{p} {version(p)}" for p in ("numpy", "scipy", "scikit-learn", "kiwipiepy", "kiwipiepy_model")]
    lines.append(f"tokenizer retrieval/common/kiwi_tokenizer.py config_id={kt.config_id()}")
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── 토큰화 (팀 공통 모듈) ────────────────────────────────────
def get_tokens(corpus, val, test):
    """문서와 질문을 공통 토크나이저로 잘라 낱말 목록 (글마다 list[str]) 으로 돌려준다.

    공통 토크나이저는 Kiwi 사전에 있는 여러 낱말짜리 이름 ("스탠더드 앤드 푸어스") 을 토큰 하나로 낼 때가 있다
    (corpus 에서 20번). 토큰을 공백으로 이었다가 다시 나누면 이런 토큰이 쪼개져 공통 토큰화의 결과가 바뀐다.
    그래서 목록을 그대로 벡터화기에 넣고, n-gram 만 아래 ngram_analyzer 가 만든다.
    """
    def cut(name, texts):
        return kt.tokenize_all(texts, cache_dir=CACHE_DIR, name=name)

    return {
        "corpus": cut("corpus", [d["search_text"] for d in corpus]),
        "validation": cut("validation", [q["question"] for q in val]),
        "test": cut("test", [q["question"] for q in test]),
    }


# ── 설정 ────────────────────────────────────────────────────
def ngram_analyzer(lo, hi):
    """이미 자른 토큰 목록에서 lo ~ hi 개씩 이어진 낱말 묶음을 만든다 (scikit-learn 의 ngram_range 와 같은 방식:
    묶음 안 낱말은 공백으로 잇는다). analyzer 가 함수이면 ngram_range 는 무시되므로 직접 만든다."""
    def analyze(tokens):
        out = []
        for n in range(lo, hi + 1):
            out.extend(" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
        return out
    return analyze


def make_vectorizer(cfg):
    # 토큰은 공통 토크나이저가 이미 정했으므로 그대로 쓴다 (소문자 · 불용어 처리 등을 더 하지 않는다)
    return TfidfVectorizer(analyzer=ngram_analyzer(*cfg["ngram"]),
                           sublinear_tf=cfg["sublinear"], max_df=cfg["max_df"], min_df=cfg["min_df"])


def family(cfg):
    """n-gram 최대 길이가 1 이면 'TF-IDF', 1 보다 크면 'TF-IDF + N-gram' (Notion 의 두 모델)."""
    return "TF-IDF" if cfg["ngram"][1] == 1 else "TF-IDF + N-gram"


def describe(cfg):
    return (f"shared Kiwi tokenizer, ngram={tuple(cfg['ngram'])}, sublinear_tf={cfg['sublinear']}, "
            f"max_df={cfg['max_df']}, min_df={cfg['min_df']}")


# ── 검색 ────────────────────────────────────────────────────
def rank_top_k(vec, doc_texts, query_texts, unit_ids, k=K, chunk=500):
    """corpus 에만 fit → 질문 변환 → 코사인 유사도(L2 정규화라 내적) 상위 k. 동점은 corpus 순서."""
    X = vec.fit_transform(doc_texts)
    Q = vec.transform(query_texts)
    XT = X.T.tocsr()
    result = []
    for s in range(0, Q.shape[0], chunk):
        S = (Q[s:s + chunk] @ XT).toarray()
        order = np.argsort(-S, axis=1, kind="stable")[:, :k]
        result.extend([[unit_ids[j] for j in row] for row in order])
    return result


def predict(cfg, split, data):
    """{question_id: [상위 3 unit_id]}. 질문 글 (question) 만 쓰고 정답 필드는 읽지 않는다."""
    corpus, val, test, texts = data
    unit_ids = [d["unit_id"] for d in corpus]
    qs = val if split == "validation" else test
    ranked = rank_top_k(make_vectorizer(cfg), texts["corpus"], texts[split], unit_ids)
    return {q["question_id"]: r for q, r in zip(qs, ranked)}


def score(preds, questions, corpus_ids):
    return ev.evaluate(questions, preds, k=K, corpus_ids=corpus_ids)


def write_preds(path, preds, questions):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            row = {"question_id": q["question_id"], "retrieved_unit_ids": preds[q["question_id"]]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── 설정 탐색 (validation 만) ───────────────────────────────
def grid_configs():
    """n-gram 범위 (1-1 이 'TF-IDF', 나머지가 'TF-IDF + N-gram') x sublinear_tf x max_df x min_df = 64개."""
    return [dict(ngram=list(ng), sublinear=sub, max_df=mdf, min_df=mn)
            for ng, sub, mdf, mn in itertools.product(
                [(1, 1), (1, 2), (1, 3), (1, 4)], [False, True], [1.0, 0.5, 0.3, 0.1], [1, 2])]


def cmd_grid(_args):
    write_environment()
    corpus, val, test = load_all()
    data = (corpus, val, test, get_tokens(corpus, val, test))
    corpus_ids = {d["unit_id"] for d in corpus}

    rows = []
    cfgs = grid_configs()
    for i, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        res = score(predict(cfg, "validation", data), val, corpus_ids)
        row = dict(family=family(cfg), ngram=f"{cfg['ngram'][0]}-{cfg['ngram'][1]}",
                   sublinear_tf=cfg["sublinear"], max_df=cfg["max_df"], min_df=cfg["min_df"],
                   precision3=res["mean_precision"], recall3=res["mean_recall"], ndcg3=res["mean_ndcg"])
        rows.append((row, cfg))
        print(f"[{i:2d}/{len(cfgs)}] {row['family']:<16} {describe(cfg):<95} "
              f"P@3={row['precision3']:.4f} R@3={row['recall3']:.4f} NDCG@3={row['ndcg3']:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "grid_validation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0][0].keys()))
        w.writeheader()
        w.writerows(r for r, _ in rows)

    best = {}
    for fam in SLUG:
        r, c = max(((r, c) for r, c in rows if r["family"] == fam), key=lambda rc: rc[0]["ndcg3"])  # 선택 기준: validation NDCG@3
        best[fam] = dict(config=c, validation=dict(precision3=r["precision3"], recall3=r["recall3"],
                                                   ndcg3=r["ndcg3"]))
    with open(OUT / "best_configs.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\n== validation 최고 설정 (NDCG@3 기준) ==")
    print("Model | Key settings | Precision@3 | Recall@3 | NDCG@3")
    for fam, b in best.items():
        v = b["validation"]
        write_preds(OUT / f"{SLUG[fam]}-validation.jsonl", predict(b["config"], "validation", data), val)
        print(f"{fam} | {describe(b['config'])} | {v['precision3']:.4f} | {v['recall3']:.4f} | {v['ndcg3']:.4f}")


# ── test (설정 고정 후 한 번) ───────────────────────────────
def cmd_final(_args):
    path = OUT / "best_configs.json"
    if not path.exists():
        sys.exit("best_configs.json 이 없다. 먼저 grid 를 돌려라.")
    with open(path, encoding="utf-8") as f:
        best = json.load(f)
    write_environment()
    corpus, val, test = load_all()
    data = (corpus, val, test, get_tokens(corpus, val, test))
    for fam, b in best.items():
        write_preds(OUT / f"{SLUG[fam]}-test.jsonl", predict(b["config"], "test", data), test)
        print(f"쓴 파일: {OUT / (SLUG[fam] + '-test.jsonl')}")


# ── 평균 점수 표 ────────────────────────────────────────────
def cmd_summary(_args):
    """저장된 예측 JSONL 을 공통 평가 코드로 채점해 평균 점수를 summary.csv · summary.md 에 쓴다.

    새로 검색하지 않는다 (test 를 다시 돌리지 않는다). 값은 evaluation.py 가 내는 평균과 같다.
    """
    with open(OUT / "best_configs.json", encoding="utf-8") as f:
        best = json.load(f)
    corpus, val, test = load_all()
    corpus_ids = {d["unit_id"] for d in corpus}
    rows = []
    for fam, slug in SLUG.items():
        for split, qs in (("validation", val), ("test", test)):
            preds, _dups = ev.load_predictions(OUT / f"{slug}-{split}.jsonl")
            res = ev.evaluate(qs, preds, k=K, corpus_ids=corpus_ids)
            base = dict(model=fam, split=split, key_settings=describe(best[fam]["config"]))
            rows.append(dict(base, query_type="ALL", n_questions=res["n_scored"], precision3=res["mean_precision"],
                             recall3=res["mean_recall"], ndcg3=res["mean_ndcg"]))
            for t, v in res["by_query_type"].items():
                rows.append(dict(base, query_type=t, n_questions=v["n"], precision3=v["mean_precision"],
                                 recall3=v["mean_recall"], ndcg3=v["mean_ndcg"]))

    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    env = (OUT / "environment.txt").read_text(encoding="utf-8").strip().replace("\n", "; ")
    md = ["# TF-IDF and TF-IDF + N-gram: mean scores", "",
          "Generated by `python retrieval/tfidf/tfidf.py summary` from the saved prediction JSONLs, scored with "
          "`evaluation/evaluation.py` (k=3). Each score is the mean over all questions of the split. "
          "Settings were chosen on validation by NDCG@3 (`grid_validation.csv`, `best_configs.json`); "
          "test was evaluated with those fixed settings.", "", f"Environment: {env}", ""]
    for split, n in (("validation", len(val)), ("test", len(test))):
        md += [f"## {split} ({n:,} questions)", "",
               "| Model | Key settings | Precision@3 | Recall@3 | NDCG@3 |", "|---|---|---|---|---|"]
        md += [f"| {r['model']} | {r['key_settings']} | {r['precision3']:.4f} | {r['recall3']:.4f} | {r['ndcg3']:.4f} |"
               for r in rows if r["split"] == split and r["query_type"] == "ALL"]
        md += ["", f"By query_type ({split}):", "",
               "| Model | query_type | Questions | Precision@3 | Recall@3 | NDCG@3 |", "|---|---|---|---|---|---|"]
        md += [f"| {r['model']} | {r['query_type']} | {r['n_questions']} | {r['precision3']:.4f} | "
               f"{r['recall3']:.4f} | {r['ndcg3']:.4f} |"
               for r in rows if r["split"] == split and r["query_type"] != "ALL"]
        md.append("")
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("grid", help="validation 에서 설정 탐색").set_defaults(fn=cmd_grid)
    sub.add_parser("final", help="고정한 설정으로 test 결과 JSONL 생성").set_defaults(fn=cmd_final)
    sub.add_parser("summary", help="저장된 예측을 채점해 평균 점수 표 (summary.csv · summary.md) 생성").set_defaults(
        fn=cmd_summary)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
