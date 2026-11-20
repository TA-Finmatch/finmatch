"""팀원 수정 요청 반영판 — 질문과 정답 연결 (T10, 2026-11-17).

정답 상품 이름 → product_id → unit_id 로 잇는다 (팀원 요청 5절).
  - 원천 라벨: data/raw/{Training,Validation}/02.라벨링데이터/{TL_1*,VL_1*}/*.json (소비자 폴더 TL_2 · VL_2 는 쓰지 않음)
  - 상품 · 단위: work/products.pkl (build.py 결과)
  - 지운 기록 → 남긴 기록: work/지운이름_남긴이름.csv (공백만 다른 같은 이름 34쌍, review/시점_조사/)
결과: work/questions.pkl, work/정답_이름_보정_기록.csv
"""
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pipeline.py · class_codes.py 가 같은 폴더에 있다
import pipeline as pl  # noqa: E402

OUT = os.path.join(HERE, "work")
NOWS = lambda s: re.sub(r"\s", "", s)  # noqa: E731
TYPO = {"일고객덱스": "일본인덱스", "ㅐ ": ""}  # 11-08 에 찾은 원천 오타 (pipeline.fix_label_names 와 같음)
ROLE = {"Training": ("TR", "development_validation", "validation"), "Validation": ("VA", "final_test", "test")}


def split_names(x):
    """원천 product_names 는 거의 목록이지만, 4개는 괄호 한쪽이 빠진 문자열이다 (3개는 닫는 ], TR-05890 은 여는 [)."""
    if isinstance(x, list) and not (len(x) > 5 and all(len(s) <= 1 for s in x)):
        return [pl.N(s) for s in x], None
    s = "".join(x)
    return [pl.N(n) for n in pl._QUOTED.findall(s)], "목록이 문자열로 깨져 있어 따옴표 안의 이름을 뽑음"


def link(name, by_name, by_nows, alias):
    """(product_id or None, 적용한 규칙)."""
    if name in by_name:
        return by_name[name], "글자 그대로"
    if name in alias:
        return by_name[alias[name]], "지운 기록 → 남긴 기록 (공백만 다른 같은 이름 쌍)"
    fixed = name
    for a, b in TYPO.items():
        fixed = fixed.replace(a, b)
    if fixed != name and fixed in by_name:
        return by_name[fixed], "원천 오타 고침"
    k = NOWS(fixed)
    if k in by_nows:
        return by_nows[k], "공백만 다름" + (" + 원천 오타 고침" if fixed != name else "")
    return None, "못 이음"


def main():
    p = pd.read_pickle(os.path.join(OUT, "products.pkl"))
    by_name = dict(zip(p.product_name, p.product_id))
    by_nows = {}
    for n, i in zip(p.product_name, p.product_id):
        assert NOWS(n) not in by_nows, n  # 34쌍을 지웠으므로 공백 없는 이름이 겹치지 않아야 한다
        by_nows[NOWS(n)] = i
    unit_of = dict(zip(p.product_id, p.unit_id))
    a = pd.read_csv(os.path.join(OUT, "지운이름_남긴이름.csv"), encoding="utf-8-sig")
    alias = dict(zip(a["지운_기록"], a["남길_기록"]))
    removed = set(pd.read_csv(os.path.join(HERE, "review", "뺀_상품.csv"), encoding="utf-8-sig")["상품_이름"])
    rows, log = [], []
    for folder, pat in [("Training", "TL_1*"), ("Validation", "VL_1*")]:
        pre, role, out_file = ROLE[folder]
        for d in pl._read(f"{pl.RAW}/{folder}/02.라벨링데이터/{pat}/*.json*"):
            qid = f"{pre}-{int(d['cot_id']):05d}"
            names, note = split_names(d["product_names"])
            if note:
                log.append(dict(question_id=qid, 원래_이름="(목록 전체)", 이은_상품="", 규칙=note))
            pids, bad = [], []
            for n in names:
                pid, rule = link(n, by_name, by_nows, alias)
                if pid is None and n in removed:
                    rule = "뺀 상품 (손실 값이 달라 11-17 에 뺌) → 정답을 잇지 않고 needs_review"
                if pid is None:
                    bad.append(n)
                else:
                    pids.append(pid)
                if rule != "글자 그대로":
                    log.append(dict(question_id=qid, 원래_이름=n, 이은_상품=pid or "", 규칙=rule))
            pids = list(dict.fromkeys(pids))
            uids = list(dict.fromkeys(unit_of[i] for i in pids))
            rows.append(dict(question_id=qid, cot_id=int(d["cot_id"]), source_split=folder, source_file=d["_file"],
                             evaluation_role=role, out_file=out_file, query_type=d["query_type"],
                             question_raw=d["question"], question=re.sub(r"\s+", " ", d["question"]).strip(),
                             reference_product_names=names, reference_product_ids=pids, reference_unit_ids=uids,
                             unresolved_product_names=bad, mapping_status="needs_review" if bad else "matched"))
    q = pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)
    assert q.question_id.is_unique
    q.to_pickle(os.path.join(OUT, "questions.pkl"))
    lg = pd.DataFrame(log)
    lg.to_csv(os.path.join(OUT, "정답_이름_보정_기록.csv"), index=False, encoding="utf-8-sig")
    print("질문", q.out_file.value_counts().to_dict(), "| mapping_status", q.mapping_status.value_counts().to_dict())
    print("정답 이름", int(q.reference_product_names.map(len).sum()), "| 이은 상품", int(q.reference_product_ids.map(len).sum()),
          "| 못 이은 이름", int(q.unresolved_product_names.map(len).sum()))
    print("보정 규칙별", lg.규칙.value_counts().to_dict() if len(lg) else {})
    print("질문당 정답 단위 수", q.reference_unit_ids.map(len).value_counts().sort_index().to_dict())
    print("정답 상품이 같은 단위로 합쳐진 질문", int((q.reference_unit_ids.map(len) < q.reference_product_ids.map(len)).sum()))


if __name__ == "__main__":
    main()
