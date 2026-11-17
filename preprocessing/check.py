"""팀원 요청 8절 점검 목록을 JSONL 4개에 돌린다 (T14, 2026-11-17).
실행 (저장소 맨 위에서): python preprocessing/check.py  → 결과를 화면과 data/processed/점검_결과.md 에 쓴다.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pipeline.py · class_codes.py 가 같은 폴더에 있다
import pipeline as pl  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "data", "processed")
ATTR = ["product_type", "protected_type", "maturity_type", "tax_type", "payment_type", "loss_rate", "liquidity_conditions"]


def read(name):
    with open(os.path.join(OUT, name), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    P, C, V, T = (read(f"{n}.jsonl") for n in ("products", "corpus", "validation", "test"))
    pid = {p["product_id"]: p for p in P}
    uid = {u["unit_id"]: u for u in C}
    res = []

    def ok(name, cond, detail=""):
        res.append((name, "통과" if cond else "실패", detail))

    raw = pl.load_products()
    ok("원천 8,000개 = 남은 상품 + 지운 기록 34", len(raw) == len(P) + 34, f"원천 {len(raw)}, 남은 {len(P)}")
    ok("상품 ID · 단위 ID · 질문 ID 가 겹치지 않음",
       len(pid) == len(P) and len(uid) == len(C) and len({q['question_id'] for q in V + T}) == len(V) + len(T))
    linked = [i for u in C for i in u["source_product_ids"]]
    ok("모든 상품이 단위 하나에만 들어감", sorted(linked) == sorted(pid), f"단위가 가리키는 상품 {len(linked)}")
    ok("source_product_ids 개수 = n_products", all(len(u["source_product_ids"]) == u["n_products"] for u in C))
    ok("상품의 unit_id 와 단위의 source_product_ids 가 서로 맞음",
       all(pid[i]["unit_id"] == u["unit_id"] for u in C for i in u["source_product_ids"]))
    bad_acct = [u["unit_id"] for u in C if any(pid[i]["account_type"] != u["account_type"] for i in u["source_product_ids"])]
    ok("상품과 단위의 가입 구분이 같음", not bad_acct, f"다른 단위 {len(bad_acct)}")
    conflict = [u["unit_id"] for u in C for c in ATTR
                if any(pid[i]["source"][c] != u[c] for i in u["source_product_ids"])]
    grade = [u["unit_id"] for u in C if any(int(re.search(r"\d", pid[i]["source"]["risk_grade"]).group()) != u["risk_grade"]
                                            for i in u["source_product_ids"])]
    ok("단위 안 상품의 원천 속성이 단위 값과 같음 (속성 충돌 없음)", not conflict and not grade,
       f"속성 다른 단위 {len(set(conflict))}, 위험등급 다른 단위 {len(grade)}")
    miss = [u["unit_id"] for u in C if not all(str(u[c]) in u["search_text"] for c in ATTR) or u["fund_name"] not in u["search_text"]
            or (u["strategy"] and u["strategy"].strip() not in u["search_text"])]  # 원천 글 끝 공백은 search_text 에서 잘린다
    ok("search_text 에 펀드 이름 · 속성 · 투자전략이 그대로 들어감", not miss, f"어긋난 단위 {len(miss)}")
    refs = [x for q in V + T for x in q["reference_unit_ids"]]
    ok("모든 정답 unit_id 가 corpus 에 있음", all(x in uid for x in refs), f"정답 단위 참조 {len(refs)}")
    ok("모든 정답 product_id 가 products 에 있음", all(x in pid for q in V + T for x in q["reference_product_ids"]))
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    dup = {norm(q["question"]) for q in V} & {norm(q["question"]) for q in T}
    ok("validation · test 사이에 같은 질문 없음", not dup, f"같은 질문 {len(dup)}")
    counts = {
        "투자전략 빈 단위": sum(not u["strategy"] for u in C),
        "가입 구분 unknown (상품)": sum(p["account_type"] == "unknown" for p in P),
        "정답 못 이은 질문 (needs_review)": sum(q["mapping_status"] != "matched" for q in V + T),
        "못 이은 정답 이름": sum(len(q["unresolved_product_names"]) for q in V + T),
        "investment_type null (상품)": sum(p["investment_type"] is None for p in P),
        "클래스 없음 (상품)": sum(p["class_status"] == "클래스 없음" for p in P),
    }
    lines = ["# 점검 결과 (팀원 요청 8절)", "", "| 점검 | 결과 | 비고 |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in res]
    lines += ["", "| 건수 | 값 |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in counts.items()]
    text = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "점검_결과.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
