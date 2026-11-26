"""TF-IDF 와 TF-IDF + N-gram 검색 (Notion "TF-IDF, TF-IDF + N-gram" 규칙).

규칙 요약
- 검색 후보: corpus.jsonl 3,930개 전체, 상품 필터 없음.
- 입력: 질문의 `question`. `reference_unit_ids` 는 채점에서만 쓴다 (이 파일은 읽지도 않는다).
- TF-IDF 는 corpus 의 `search_text` 에만 fit 하고, 질문은 같은 토크나이저 · 같은 벡터화기로 변환한다.
- 출력: 질문마다 서로 다른 unit_id 상위 3개. 한 줄에 {"question_id", "retrieved_unit_ids"}.
- 설정은 validation 의 NDCG@3 으로만 고른다. test 는 `final` 로 설정을 고정한 뒤 한 번만 돌린다.

사용 (저장소 루트에서)
    python retrieval/tfidf/tfidf.py grid     # validation 에서 설정 탐색 → results/tfidf/
    python retrieval/tfidf/tfidf.py final    # best_configs.json 의 설정으로 test 결과 JSONL 생성 (한 번만)
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
import evaluation as ev  # noqa: E402  공통 평가 코드 (팀 공용)

DATA = ROOT / "data" / "processed" / "v3"
OUT = ROOT / "results" / "tfidf"
CACHE = ROOT / "scratch" / "tfidf_tokens.json"  # scratch/ 는 .gitignore 대상
K = 3

# 품사 필터 (Kiwi 태그). 'all' 은 문장부호 · 기호만 뺀다.
PUNCT = {"SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW"}
NOUN = {"NNG", "NNP", "SL", "SN"}
CONTENT = NOUN | {"VV", "VA", "XR"}


# ── 입출력 ──────────────────────────────────────────────────
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_all():
    corpus = read_jsonl(DATA / "corpus.jsonl")
    val = read_jsonl(DATA / "validation.jsonl")
    test = read_jsonl(DATA / "test.jsonl")
    return corpus, val, test


# ── 토큰화 (Kiwi, 한 번만 돌려 캐시) ─────────────────────────
def kiwi_tokens(texts):
    from kiwipiepy import Kiwi

    kiwi = Kiwi()
    out = []
    for toks in kiwi.tokenize(texts):
        out.append([[t.form, t.tag] for t in toks])
    return out


def kiwipiepy_version():
    from importlib.metadata import version

    return version("kiwipiepy")


def write_environment():
    """실행 환경 (Python · 라이브러리 버전) 을 results/tfidf/environment.txt 에 남긴다."""
    import platform
    from importlib.metadata import version

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"python {platform.python_version()}"]
    lines += [f"{p} {version(p)}" for p in ("numpy", "scipy", "scikit-learn", "kiwipiepy", "kiwipiepy_model")]
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def get_tokenized(corpus, val, test):
    """{'corpus': [...], 'validation': [...], 'test': [...]} 각 문서는 [[form, tag], ...].

    캐시는 kiwipiepy 버전이 같을 때만 쓴다 (버전이 다르면 토큰이 달라질 수 있다).
    """
    if CACHE.exists():
        with open(CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        if (cached.get("kiwipiepy") == kiwipiepy_version()
                and len(cached["corpus"]) == len(corpus) and len(cached["validation"]) == len(val)
                and len(cached["test"]) == len(test)):
            return cached
    t0 = time.time()
    res = {
        "kiwipiepy": kiwipiepy_version(),
        "corpus": kiwi_tokens([d["search_text"] for d in corpus]),
        "validation": kiwi_tokens([q["question"] for q in val]),
        "test": kiwi_tokens([q["question"] for q in test]),
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    print(f"Kiwi 토큰화 {time.time() - t0:.0f}s → {CACHE}")
    return res


def keep(tokens, pos):
    if pos == "all":
        return " ".join(f for f, t in tokens if t not in PUNCT)
    allowed = NOUN if pos == "noun" else CONTENT
    return " ".join(f for f, t in tokens if t in allowed)


# ── 설정 ────────────────────────────────────────────────────
def make_vectorizer(cfg):
    common = dict(sublinear_tf=cfg["sublinear"], max_df=cfg["max_df"], min_df=cfg["min_df"],
                  ngram_range=tuple(cfg["ngram"]))
    if cfg["analyzer"] == "word":  # 이미 토큰이 공백으로 이어진 글
        return TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=True, **common)
    return TfidfVectorizer(analyzer="char_wb", lowercase=True, **common)


def family(cfg):
    if cfg["analyzer"] == "char_wb":
        return "TF-IDF + char N-gram"
    return "TF-IDF" if cfg["ngram"][1] == 1 else "TF-IDF + N-gram"


def describe(cfg):
    if cfg["analyzer"] == "char_wb":
        tok = "char_wb"
    else:
        tok = f"Kiwi({cfg['pos']})"
    return (f"{tok}, ngram={tuple(cfg['ngram'])}, sublinear_tf={cfg['sublinear']}, "
            f"max_df={cfg['max_df']}, min_df={cfg['min_df']}")


def texts_for(cfg, tokenized, corpus, val, test):
    """(문서 글, {'validation': 질문 글, 'test': 질문 글}) — 문서와 질문을 같은 방식으로 만든다."""
    if cfg["analyzer"] == "char_wb":
        return ([d["search_text"] for d in corpus],
                {"validation": [q["question"] for q in val], "test": [q["question"] for q in test]})
    p = cfg["pos"]
    return ([keep(t, p) for t in tokenized["corpus"]],
            {"validation": [keep(t, p) for t in tokenized["validation"]],
             "test": [keep(t, p) for t in tokenized["test"]]})


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
    corpus, val, test, tokenized = data
    docs, queries = texts_for(cfg, tokenized, corpus, val, test)
    unit_ids = [d["unit_id"] for d in corpus]
    qs = val if split == "validation" else test
    ranked = rank_top_k(make_vectorizer(cfg), docs, queries[split], unit_ids)
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
    cfgs = []
    for pos, ng, sub, mdf in itertools.product(
            ["all", "content", "noun"], [(1, 1), (1, 2), (1, 3)], [False, True], [1.0, 0.5]):
        cfgs.append(dict(analyzer="word", pos=pos, ngram=list(ng), sublinear=sub, max_df=mdf, min_df=1))
    for ng, sub in itertools.product([(2, 4), (3, 5)], [False, True]):  # README 참고 기준 (char_wb 2~4)
        cfgs.append(dict(analyzer="char_wb", pos=None, ngram=list(ng), sublinear=sub, max_df=1.0, min_df=1))
    return cfgs


def refine_configs():
    """2차 탐색: 1차 최고 설정이 격자 끝 (max_df 0.5, ngram 1-3) 이라 그 너머를 더 본다."""
    cfgs = []
    for pos, ng, sub, mdf, mn in itertools.product(
            ["noun", "content"], [(1, 1), (1, 2), (1, 3), (1, 4)], [False, True], [0.5, 0.3, 0.1], [1, 2]):
        cfgs.append(dict(analyzer="word", pos=pos, ngram=list(ng), sublinear=sub, max_df=mdf, min_df=mn))
    return cfgs


def cfg_key(c):
    return (c["analyzer"], c["pos"], tuple(c["ngram"]), c["sublinear"], c["max_df"], c["min_df"])


def load_prior():
    """이미 저장된 grid_validation.csv 의 결과를 (행, 설정) 으로 되살린다 (stage 칸이 없으면 1차)."""
    path = OUT / "grid_validation.csv"
    if not path.exists():
        return []
    prior = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            lo, hi = r["ngram"].split("-")
            cfg = dict(analyzer=r["analyzer"], pos=r["pos"] or None, ngram=[int(lo), int(hi)],
                       sublinear=r["sublinear_tf"] == "True", max_df=float(r["max_df"]),
                       min_df=int(r["min_df"]))
            row = dict(family=r["family"], analyzer=r["analyzer"], pos=r["pos"], ngram=r["ngram"],
                       sublinear_tf=cfg["sublinear"], max_df=cfg["max_df"], min_df=cfg["min_df"],
                       precision3=float(r["precision3"]), recall3=float(r["recall3"]),
                       ndcg3=float(r["ndcg3"]), stage=int(r.get("stage") or 1))
            prior.append((row, cfg))
    return prior


def cmd_grid(args):
    write_environment()
    corpus, val, test = load_all()
    tokenized = get_tokenized(corpus, val, test)
    data = (corpus, val, test, tokenized)
    corpus_ids = {d["unit_id"] for d in corpus}

    rows = load_prior() if args.stage == 2 else []
    done = {cfg_key(c) for _, c in rows}
    cfgs = [c for c in (grid_configs() if args.stage == 1 else refine_configs()) if cfg_key(c) not in done]
    for i, cfg in enumerate(cfgs, 1):
        t0 = time.time()
        res = score(predict(cfg, "validation", data), val, corpus_ids)
        row = dict(family=family(cfg), analyzer=cfg["analyzer"], pos=cfg["pos"] or "",
                   ngram=f"{cfg['ngram'][0]}-{cfg['ngram'][1]}", sublinear_tf=cfg["sublinear"],
                   max_df=cfg["max_df"], min_df=cfg["min_df"],
                   precision3=res["mean_precision"], recall3=res["mean_recall"], ndcg3=res["mean_ndcg"],
                   stage=args.stage)
        rows.append((row, cfg))
        print(f"[{i:2d}/{len(cfgs)}] {row['family']:<22} {describe(cfg):<70} "
              f"P@3={row['precision3']:.4f} R@3={row['recall3']:.4f} NDCG@3={row['ndcg3']:.4f} "
              f"({time.time() - t0:.1f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "grid_validation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0][0].keys()))
        w.writeheader()
        w.writerows(r for r, _ in rows)

    best = {}
    for fam in ("TF-IDF", "TF-IDF + N-gram", "TF-IDF + char N-gram"):
        cands = [(r, c) for r, c in rows if r["family"] == fam]
        r, c = max(cands, key=lambda rc: rc[0]["ndcg3"])  # 선택 기준: validation NDCG@3
        best[fam] = dict(config=c, validation=dict(precision3=r["precision3"], recall3=r["recall3"],
                                                   ndcg3=r["ndcg3"]))
    with open(OUT / "best_configs.json", "w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print("\n== validation 최고 설정 (NDCG@3 기준) ==")
    print("Model | Key settings | Precision@3 | Recall@3 | NDCG@3")
    for fam, b in best.items():
        v = b["validation"]
        slug = {"TF-IDF": "tfidf", "TF-IDF + N-gram": "tfidf-ngram", "TF-IDF + char N-gram": "tfidf-char"}[fam]
        write_preds(OUT / f"{slug}-validation.jsonl", predict(b["config"], "validation", data), val)
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
    data = (corpus, val, test, get_tokenized(corpus, val, test))
    for fam, b in best.items():
        slug = {"TF-IDF": "tfidf", "TF-IDF + N-gram": "tfidf-ngram", "TF-IDF + char N-gram": "tfidf-char"}[fam]
        write_preds(OUT / f"{slug}-test.jsonl", predict(b["config"], "test", data), test)
        print(f"쓴 파일: {OUT / (slug + '-test.jsonl')}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid", help="validation 에서 설정 탐색 (--stage 2 는 1차 결과에 이어 세부 탐색)")
    g.add_argument("--stage", type=int, choices=[1, 2], default=1)
    g.set_defaults(fn=cmd_grid)
    sub.add_parser("final", help="고정한 설정으로 test 결과 JSONL 생성").set_defaults(fn=cmd_final)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
