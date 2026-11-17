"""팀원 수정 요청 반영판 — JSONL 4개를 쓴다 (T11, 2026-11-17).

먼저 build.py → build_labels.py 를 돌려 work/ 의 pkl 을 만든 뒤 실행한다.
결과: data/processed/{products,corpus,validation,test}.jsonl (UTF-8, 한 줄에 JSON 하나, 결측치는 null)
"""
import json
import math
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "work")
OUT = os.path.join(os.path.dirname(HERE), "data", "processed")
SRC11 = ["product_name", "product_type", "protected_type", "maturity_type", "income_rate", "risk_grade",
         "tax_type", "payment_type", "loss_rate", "liquidity_conditions", "investment_strategy"]


def clean(v):
    """pandas 의 NaN · NA · numpy 수를 JSON 값으로."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if v is pd.NA:
        return None
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    return v


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(clean(r), ensure_ascii=False) + "\n")
    return len(rows)


def search_text(u):
    """최종 속성 + 투자전략. 원천 값을 그대로 이어 붙이고 새 말을 지어내지 않는다.
    유형 자리는 원천 product_type (6개) 그대로 (11-17 진우님: 상품마다 원래 6개)."""
    parts = [u["fund_name"], u["product_type"], f"위험등급 {u['risk_grade_raw']}", u["protected_type"], u["maturity_type"],
             f"세제 혜택 {u['tax_type']}", u["payment_type"], f"손실 {u['loss_rate']}", u["liquidity_conditions"]]
    return (". ".join(str(x) for x in parts) + ". " + (u["strategy"] or "")).strip()


def main():
    p = pd.read_pickle(os.path.join(WORK, "products.pkl"))
    u = pd.read_pickle(os.path.join(WORK, "units.pkl"))
    q = pd.read_pickle(os.path.join(WORK, "questions.pkl"))
    os.makedirs(OUT, exist_ok=True)
    raw_grade = dict(zip(p.unit_id, p.risk_grade))  # 단위 안 상품은 위험등급이 모두 같다 (build 의 단위 규칙)

    prows = [dict(product_id=r.product_id, fund_id=r.fund_id, unit_id=r.unit_id, fund_name=r.fund_name,
                  share_class=r.share_class, class_status=r.class_status, account_type=r.account_type,
                  account_type_basis=r.account_type_basis, investment_type=r.investment_type,
                  source_locations=r.source_locations, source={c: r[c] for c in SRC11})
             for _, r in p.sort_values("product_id").iterrows()]

    crows = []
    for _, r in u.sort_values("unit_id").iterrows():
        d = r.to_dict()
        d["risk_grade_raw"] = raw_grade[r.unit_id]
        crows.append(dict(unit_id=r.unit_id, fund_id=r.fund_id, fund_name=r.fund_name, account_type=r.account_type,
                          source_product_ids=r.source_product_ids, class_list=r.class_list, n_products=r.n_products,
                          product_type=r.product_type, investment_type=r.investment_type, risk_grade=r.risk_grade,
                          protected_type=r.protected_type, maturity_type=r.maturity_type, tax_type=r.tax_type,
                          payment_type=r.payment_type, loss_rate=r.loss_rate, liquidity_conditions=r.liquidity_conditions,
                          strategy=r.strategy,  # strategy_status · strategy_flag 는 내보내지 않는다 (11-17 진우님)
                         
                          search_text=search_text(d)))

    def qrows(name):
        return [dict(question_id=r.question_id, cot_id=r.cot_id, source_split=r.source_split, source_file=r.source_file,
                     evaluation_role=r.evaluation_role, query_type=r.query_type, question_raw=r.question_raw,
                     question=r.question, reference_product_names=r.reference_product_names,
                     reference_product_ids=r.reference_product_ids, reference_unit_ids=r.reference_unit_ids,
                     unresolved_product_names=r.unresolved_product_names, mapping_status=r.mapping_status)
                for _, r in q[q.out_file == name].sort_values("question_id").iterrows()]

    n = {"products.jsonl": write(os.path.join(OUT, "products.jsonl"), prows),
         "corpus.jsonl": write(os.path.join(OUT, "corpus.jsonl"), crows),
         "validation.jsonl": write(os.path.join(OUT, "validation.jsonl"), qrows("validation")),
         "test.jsonl": write(os.path.join(OUT, "test.jsonl"), qrows("test"))}
    print(n)


if __name__ == "__main__":
    main()
