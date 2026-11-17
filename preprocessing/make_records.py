"""최종 수정 요청 (11-17) 의 수정 기록을 만든다. JSONL 을 만든 뒤 돌린다.

실행 (저장소 맨 위에서):
    git show 6eb3d7e:data/processed/products.jsonl > /tmp/products_prev.jsonl
    python preprocessing/make_records.py /tmp/products_prev.jsonl
6eb3d7e 는 11-17 에 처음 올린 판 (검색 단위 3,871개) 이다.

결과 (preprocessing/records/):
- 단위ID_대응표.csv     : 이전 판 상품마다 이전 product_id · unit_id 와 최종 product_id · unit_id (원천 파일 이름으로 이음)
- 단위_분리_이유.csv    : 같은 펀드 · 같은 가입 구분인데 단위가 둘 이상인 묶음마다, 무엇이 달라 나뉘었는지
- 투자전략_처리_내역.csv : 투자전략이 복구 · 사용 제외 · 미확인인 상품
- 정답_이름_보정_기록.csv : 정답 상품 이름을 고쳐 이은 기록 (build_labels.py 결과 복사)
"""
import json
import os
import shutil
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "processed")
REC = os.path.join(HERE, "records")
KEY = ["investment_type", "risk_grade", "protected_type", "maturity_type", "tax_type", "payment_type",
       "liquidity_conditions", "restriction_kind"]
KO = {"investment_type": "투자유형", "risk_grade": "위험등급", "protected_type": "원금보장", "maturity_type": "만기",
      "tax_type": "세제혜택", "payment_type": "납입 방식", "liquidity_conditions": "유동성", "restriction_kind": "가입 제한 종류"}


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def id_map(prev_path, P):
    prev = read(prev_path)
    new_by_file = {p["source_locations"][0]["source_file"]: p for p in P}
    rows = []
    for o in prev:
        f = o["source_locations"][0]["source_file"]
        n = new_by_file.get(f)
        rows.append(dict(원천_파일=f, 상품_이름=o["source"]["product_name"], 이전_product_id=o["product_id"],
                         이전_unit_id=o["unit_id"], 최종_product_id=n["product_id"] if n else "",
                         최종_unit_id=n["unit_id"] if n else "", 상태="뺌 (손실 값이 다른 상품)" if n is None else ""))
    d = pd.DataFrame(rows)
    # 이전 단위의 상품이 최종 판에서도 같은 묶음이면 "그대로", 아니면 바뀐 모양을 적는다
    old_groups = d[d.최종_unit_id != ""].groupby("이전_unit_id").최종_unit_id.agg(lambda s: frozenset(s))
    new_groups = d[d.최종_unit_id != ""].groupby("최종_unit_id").이전_unit_id.agg(lambda s: frozenset(s))
    for i, r in d.iterrows():
        if r.상태:
            continue
        a, b = len(old_groups[r.이전_unit_id]), len(new_groups[r.최종_unit_id])
        d.at[i, "상태"] = {(1, 1): "그대로"}.get((a, b), "나뉨" if b == 1 else ("합쳐짐" if a == 1 else "섞임"))
    d.to_csv(os.path.join(REC, "단위ID_대응표.csv"), index=False, encoding="utf-8-sig")
    return d


def split_reasons(C):
    g = defaultdict(list)
    for u in C:
        g[(u["fund_id"], u["account_type"])].append(u)
    rows = []
    for (fid, acc), us in g.items():
        if len(us) < 2:
            continue
        if acc == "unknown":
            why = "가입 구분 미확인 상품은 하나씩 둔다 (일괄 통합 안 함)"
        else:
            diff = [KO[k] for k in KEY if len({str(u[k]) for u in us}) > 1]
            texts = {(u["strategy_status"], u["strategy"] or "") for u in us}
            if len(texts) > 1:
                diff.append("투자전략 (사실상 같은 글이 아님, 또는 한쪽은 쓸 수 있는 글이 없음)")
            why = " · ".join(diff) + " 이(가) 다름"
        rows.append(dict(fund_id=fid, 펀드_이름=us[0]["fund_name"], 가입_구분=acc, 단위_수=len(us),
                         unit_ids=" ".join(u["unit_id"] for u in us), 나뉜_이유=why))
    pd.DataFrame(rows).to_csv(os.path.join(REC, "단위_분리_이유.csv"), index=False, encoding="utf-8-sig")
    return rows


def strategy_log(P):
    rows = [dict(product_id=p["product_id"], unit_id=p["unit_id"], 상품_이름=p["source"]["product_name"],
                 strategy_status=p["strategy_status"], 처리=p["strategy_flag"])
            for p in P if p["strategy_flag"]]
    pd.DataFrame(rows).to_csv(os.path.join(REC, "투자전략_처리_내역.csv"), index=False, encoding="utf-8-sig")
    return rows


def main():
    os.makedirs(REC, exist_ok=True)
    P, C = read(os.path.join(DATA, "products.jsonl")), read(os.path.join(DATA, "corpus.jsonl"))
    d = id_map(sys.argv[1], P)
    s = split_reasons(C)
    t = strategy_log(P)
    shutil.copy(os.path.join(HERE, "work", "정답_이름_보정_기록.csv"), os.path.join(REC, "정답_이름_보정_기록.csv"))
    print("ID 대응", d.상태.value_counts().to_dict(), "| 이전 단위", d.이전_unit_id.nunique(), "→ 최종 단위", d.최종_unit_id[d.최종_unit_id != ""].nunique())
    print("나뉜 묶음", len(s), pd.Series([r["나뉜_이유"] for r in s]).value_counts().head(8).to_dict())
    print("투자전략 처리", pd.Series([r["처리"] for r in t]).str.split(":").str[0].value_counts().to_dict())


if __name__ == "__main__":
    main()
