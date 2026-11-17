"""팀원 요청 8절 점검 목록 + 최종 수정 요청 (11-17) 점검을 JSONL 4개에 돌린다.
실행 (저장소 맨 위에서): python preprocessing/check.py  → 결과를 화면과 preprocessing/records/점검_결과.md 에 쓴다.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pipeline.py · class_codes.py 가 같은 폴더에 있다
import pipeline as pl  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "data", "processed")
REC = os.path.join(HERE, "records")
ATTR = ["protected_type", "maturity_type", "tax_type", "payment_type", "liquidity_conditions"]
KO = {"general": "일반", "pension": "연금", "restricted": "가입 제한", "unknown": "미확인"}


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
    ok("원천 8,000개 = 남은 상품 + 지운 기록 34 + 뺀 상품 3", len(raw) == len(P) + 34 + 3, f"원천 {len(raw)}, 남은 {len(P)}")
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
    inv = [u["unit_id"] for u in C if any(pid[i]["investment_type"] != u["investment_type"] for i in u["source_product_ids"])]
    ok("단위 안 상품의 원천 속성이 단위 값과 같음 (속성 충돌 없음)", not conflict and not grade and not inv,
       f"속성 다른 단위 {len(set(conflict))}, 위험등급 다른 단위 {len(grade)}, 투자유형 다른 단위 {len(inv)}")
    miss = [u["unit_id"] for u in C if not all(str(u[c]).strip() in u["search_text"] for c in ATTR) or u["fund_name"] not in u["search_text"]
            or (u["strategy"] and u["strategy"].strip() not in u["search_text"])]  # 원천 글 끝 공백은 search_text 에서 잘린다
    ok("search_text 에 펀드 이름 · 속성 · 투자전략이 그대로 들어감", not miss, f"어긋난 단위 {len(miss)}")
    acc = [u["unit_id"] for u in C if f"가입 구분: {KO[u['account_type']]}" not in u["search_text"]
           or (u["restriction_kind"] and f"({u['restriction_kind']})" not in u["search_text"])]
    ok("search_text 에 가입 구분이 들어감", not acc, f"빠진 단위 {len(acc)}")
    USE = ("available", "generated")
    leak = [u["unit_id"] for u in C if u["strategy_status"] not in USE and (u["strategy"] or "투자전략:" in u["search_text"])]
    ok("available · generated 가 아닌 투자전략은 strategy · search_text 에 없음", not leak, f"새어 나간 단위 {len(leak)}")
    mix = [u["unit_id"] for u in C if any(pid[i]["strategy_status"] != u["strategy_status"] for i in u["source_product_ids"])]
    ok("단위와 그 상품들의 strategy_status 가 같음", not mix, f"섞인 단위 {len(mix)}")
    refs = [x for q in V + T for x in q["reference_unit_ids"]]
    ok("모든 정답 unit_id 가 corpus 에 있음", all(x in uid for x in refs), f"정답 단위 참조 {len(refs)}")
    ok("모든 정답 product_id 가 products 에 있음", all(x in pid for q in V + T for x in q["reference_product_ids"]))
    norm = lambda s: re.sub(r"\s+", " ", s).strip()  # noqa: E731
    dup = {norm(q["question"]) for q in V} & {norm(q["question"]) for q in T}
    ok("validation · test 사이에 같은 질문 없음", not dup, f"같은 질문 {len(dup)}")
    counts = {
        "투자전략 available (단위)": sum(u["strategy_status"] == "available" for u in C),
        "투자전략 needs_review (단위)": sum(u["strategy_status"] == "needs_review" for u in C),
        "투자전략 missing (단위)": sum(u["strategy_status"] == "missing" for u in C),
        "투자전략 generated (단위)": sum(u["strategy_status"] == "generated" for u in C),
        "투자전략 available 아님 (상품)": sum(p["strategy_status"] != "available" for p in P),
        "search_text 가 완전히 같은 단위 묶음": sum(1 for n in __import__("collections").Counter(u["search_text"] for u in C).values() if n > 1),
        "가입 구분 unknown (상품)": sum(p["account_type"] == "unknown" for p in P),
        "정답 못 이은 질문 (needs_review)": sum(q["mapping_status"] != "matched" for q in V + T),
        "못 이은 정답 이름": sum(len(q["unresolved_product_names"]) for q in V + T),
        "investment_type null (상품)": sum(p["investment_type"] is None for p in P),
        "클래스 없음 (상품)": sum(p["class_status"] == "클래스 없음" for p in P),
    }
    from collections import Counter
    for name, qs in (("validation", V), ("test", T)):
        for k, v in sorted(Counter(len(q["reference_unit_ids"]) for q in qs).items()):
            counts[f"{name} 질문 중 정답 단위 {k}개"] = v
    lines = ["# 점검 결과 (팀원 요청 8절 + 최종 수정 요청)", "", "| 점검 | 결과 | 비고 |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in res]
    lines += ["", "| 건수 | 값 |", "|---|---|"] + [f"| {k} | {v:,} |" for k, v in counts.items()]
    text = "\n".join(lines) + "\n"
    os.makedirs(REC, exist_ok=True)
    open(os.path.join(REC, "점검_결과.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
