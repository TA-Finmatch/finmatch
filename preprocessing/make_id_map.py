"""옛 ID (11-09 판) ↔ 새 ID (data/processed/v2) 대응표를 만든다.
11-09 판 product.csv · corpus.csv 를 preprocessing/_base_1109/ 에 두어야 돈다 (git 에는 올리지 않음).

팀원 요청 3절: "기존 검색 단위를 변경했다면 이전 ID와 새 ID의 대응 관계 기록".

입력
- _base_1109/product.csv          : 11-09 판 원천 상품 8,000개 (unit_id · fund_id · product_name · source_file)
- data/processed/v2/products.jsonl            : 새 판 상품 7,966개 (product_id · unit_id · fund_id · source_locations)
- work/지운이름_남긴이름.csv        : 공백만 다른 같은 이름 34쌍에서 지운 기록 → 남긴 기록

잇는 열쇠는 원천 JSON 파일 이름 (source_file) 이다. 옛 판 8,000개와 새 판 7,966개의 파일 이름이
모두 서로 다르고, 새 판에서 빠진 34개가 지운 기록 34개와 정확히 같아야 한다 (아래 assert).
상품 이름으로도 한 번 더 맞춰 본다.

출력 (preprocessing/records/)
- 옛ID_새ID_상품.csv : 원천 상품 8,000개마다 한 줄
- 옛ID_새ID_단위.csv : 옛 unit_id → 새 unit_id 짝마다 한 줄 (상품 수 · 관계 종류)
- 마지막에 요약 숫자를 화면에 찍는다 (옛ID_새ID_설명.md 에 옮겨 적는다)

실행 (저장소 맨 위에서): python preprocessing/make_id_map.py
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OLD_PRODUCT = HERE / "_base_1109" / "product.csv"
OLD_CORPUS = HERE / "_base_1109" / "corpus.csv"
PROCESSED = HERE.parent / "data" / "processed" / "v2"
NEW_PRODUCTS = PROCESSED / "products.jsonl"
NEW_CORPUS = PROCESSED / "corpus.jsonl"
DELETED = HERE / "work" / "지운이름_남긴이름.csv"
OUT_DIR = HERE / "records"

REL_NAME = {
    (False, False): "그대로 1:1",
    (True, False): "나뉨 1:N",
    (False, True): "합쳐짐 N:1",
    (True, True): "섞임 N:M",
}


def main():
    old = pd.read_csv(OLD_PRODUCT, dtype=str)
    old_corpus_units = set(pd.read_csv(OLD_CORPUS, dtype=str)["unit_id"])
    new_rows = [json.loads(line) for line in NEW_PRODUCTS.open(encoding="utf-8")]
    new_corpus_units = {json.loads(line)["unit_id"] for line in NEW_CORPUS.open(encoding="utf-8")}
    deleted = pd.read_csv(DELETED, dtype=str, encoding="utf-8-sig")

    # ---- 새 판 상품: 원천 파일 이름 → 새 ID
    new = []
    for r in new_rows:
        files = {loc["source_file"] for loc in r["source_locations"]}
        assert len(files) == 1, r["product_id"]  # 두 폴더의 파일 이름이 같다 (T1)
        new.append({
            "source_file": files.pop(),
            "new_name": r["source"]["product_name"],
            "새_product_id": r["product_id"],
            "새_unit_id": r["unit_id"],
            "새_fund_id": r["fund_id"],
            "새_account_type": r["account_type"],
        })
    new = pd.DataFrame(new)

    # ---- 열쇠가 깨끗한지 확인
    assert old["source_file"].is_unique and len(old) == 8000
    assert new["source_file"].is_unique and new["새_product_id"].is_unique
    missing = set(old["source_file"]) - set(new["source_file"])
    assert not (set(new["source_file"]) - set(old["source_file"])), "새 판에만 있는 상품이 있다"
    missing_names = set(old.loc[old["source_file"].isin(missing), "product_name"])
    assert missing_names == set(deleted["지운_기록"]), "새 판에서 빠진 상품 ≠ 지운 기록 34개"
    assert set(old["unit_id"]) == old_corpus_units
    assert set(new["새_unit_id"]) == new_corpus_units

    m = old.merge(new, on="source_file", how="left")
    kept = m["새_product_id"].notna()
    assert (m.loc[kept, "product_name"] == m.loc[kept, "new_name"]).all(), "파일 이름과 상품 이름이 서로 다르게 이어짐"

    # ---- 지운 기록 → 남긴 기록의 새 ID
    name_to_new = m.loc[kept].set_index("product_name")[["새_product_id", "새_unit_id"]]
    del_map = dict(zip(deleted["지운_기록"], deleted["남길_기록"]))
    m["지움"] = (~kept).map({True: "예", False: "아니오"})
    m["남긴_상품_이름"] = m["product_name"].map(del_map)
    m["남긴_상품_새_product_id"] = m["남긴_상품_이름"].map(name_to_new["새_product_id"])
    m["남긴_상품_새_unit_id"] = m["남긴_상품_이름"].map(name_to_new["새_unit_id"])
    m["남긴_상품_옛_unit_id"] = m["남긴_상품_이름"].map(m.set_index("product_name")["unit_id"])
    assert m.loc[~kept, "남긴_상품_새_product_id"].notna().all()

    prod = m.rename(columns={
        "source_file": "원천_파일",
        "product_name": "원천_상품_이름",
        "unit_id": "옛_unit_id",
        "fund_id": "옛_fund_id",
    })[[
        "원천_상품_이름", "원천_파일", "옛_fund_id", "옛_unit_id",
        "지움", "새_product_id", "새_fund_id", "새_unit_id", "새_account_type",
        "남긴_상품_이름", "남긴_상품_새_product_id", "남긴_상품_새_unit_id", "남긴_상품_옛_unit_id",
    ]]

    # ---- 단위 짝 (지우지 않은 상품만으로 관계를 정한다)
    k = prod[prod["지움"] == "아니오"]
    pair = k.groupby(["옛_unit_id", "새_unit_id"]).size().rename("상품_수").reset_index()
    n_new_per_old = k.groupby("옛_unit_id")["새_unit_id"].nunique()
    n_old_per_new = k.groupby("새_unit_id")["옛_unit_id"].nunique()
    pair["옛_단위가_간_새_단위_수"] = pair["옛_unit_id"].map(n_new_per_old)
    pair["새_단위가_받은_옛_단위_수"] = pair["새_unit_id"].map(n_old_per_new)
    pair["관계"] = [
        REL_NAME[(a > 1, b > 1)]
        for a, b in zip(pair["옛_단위가_간_새_단위_수"], pair["새_단위가_받은_옛_단위_수"])
    ]
    # 1:1 이면서 상품 묶음까지 똑같은지 (지운 기록이 빠지면 묶음이 달라질 수 있다)
    old_sets = prod.groupby("옛_unit_id")["원천_파일"].apply(frozenset)
    new_sets = k.groupby("새_unit_id")["원천_파일"].apply(frozenset)
    pair["상품_묶음_같음"] = [
        "예" if old_sets[o] == new_sets[n] else "아니오"
        for o, n in zip(pair["옛_unit_id"], pair["새_unit_id"])
    ]
    del_per_old = prod[prod["지움"] == "예"].groupby("옛_unit_id").size()
    pair["옛_단위에서_지운_상품_수"] = pair["옛_unit_id"].map(del_per_old).fillna(0).astype(int)

    # 지운 기록만 있던 옛 단위가 있는지 (있으면 짝 없이 사라진 단위)
    old_only_deleted = sorted(set(prod["옛_unit_id"]) - set(k["옛_unit_id"]))
    extra = [
        {"옛_unit_id": u, "새_unit_id": "", "상품_수": 0, "옛_단위가_간_새_단위_수": 0,
         "새_단위가_받은_옛_단위_수": 0, "관계": "지운 기록만 있던 단위",
         "상품_묶음_같음": "", "옛_단위에서_지운_상품_수": int(del_per_old[u])}
        for u in old_only_deleted
    ]
    pair = pd.concat([pair, pd.DataFrame(extra)], ignore_index=True)
    pair = pair.sort_values(["옛_unit_id", "새_unit_id"]).reset_index(drop=True)

    OUT_DIR.mkdir(exist_ok=True)
    prod.to_csv(OUT_DIR / "옛ID_새ID_상품.csv", index=False, encoding="utf-8-sig")
    pair.to_csv(OUT_DIR / "옛ID_새ID_단위.csv", index=False, encoding="utf-8-sig")

    # ---- 요약
    print("== 상품")
    print("원천 상품", len(prod), "· 남김", int((prod["지움"] == "아니오").sum()),
          "· 지움", int((prod["지움"] == "예").sum()))
    same_unit_del = (prod.loc[prod["지움"] == "예", "옛_unit_id"]
                     == prod.loc[prod["지움"] == "예", "남긴_상품_옛_unit_id"]).sum()
    print("지운 기록 중 남긴 기록과 옛 단위가 같았던 것", int(same_unit_del))
    print("== 단위")
    print("옛 단위", prod["옛_unit_id"].nunique(), "· 새 단위", k["새_unit_id"].nunique())
    print("짝 수", int((pair["새_unit_id"] != "").sum()))
    print("짝 관계별:", dict(Counter(pair["관계"])))
    real = pair[pair["새_unit_id"] != ""]
    print("짝 관계별 상품 수:", real.groupby("관계")["상품_수"].sum().to_dict())
    # 옛 단위 기준 / 새 단위 기준 분류
    old_kind = real.groupby("옛_unit_id").agg(a=("옛_단위가_간_새_단위_수", "first"),
                                             b=("새_단위가_받은_옛_단위_수", "max"))
    old_kind["k"] = [REL_NAME[(a > 1, b > 1)] for a, b in zip(old_kind["a"], old_kind["b"])]
    print("옛 단위 기준 (가장 복잡한 짝으로):", dict(Counter(old_kind["k"])))
    new_kind = real.groupby("새_unit_id").agg(a=("옛_단위가_간_새_단위_수", "max"),
                                             b=("새_단위가_받은_옛_단위_수", "first"))
    new_kind["k"] = [REL_NAME[(a > 1, b > 1)] for a, b in zip(new_kind["a"], new_kind["b"])]
    print("새 단위 기준 (가장 복잡한 짝으로):", dict(Counter(new_kind["k"])))
    one = real[real["관계"] == "그대로 1:1"]
    print("1:1 짝 중 상품 묶음까지 같음", int((one["상품_묶음_같음"] == "예").sum()),
          "· 다름", int((one["상품_묶음_같음"] == "아니오").sum()))
    print("옛 단위 하나가 간 새 단위 수 분포:", dict(sorted(Counter(n_new_per_old).items())))
    print("새 단위 하나가 받은 옛 단위 수 분포:", dict(sorted(Counter(n_old_per_new).items())))
    print("지운 기록만 있던 옛 단위", old_only_deleted)
    print("새 판 단위 크기 1인 것 (상품 1개)", int((k.groupby("새_unit_id").size() == 1).sum()))
    print("옛 판 단위 크기 1인 것 (상품 1개)", int((prod.groupby("옛_unit_id").size() == 1).sum()))


if __name__ == "__main__":
    main()
