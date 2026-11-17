"""팀원 수정 요청 반영판 — 상품 줄과 검색 단위를 만든다 (T9, 2026-11-17).

원천 (data/raw) 은 읽기만 한다. 결정의 근거는 preprocessing/review/ 아래 표들과 preprocessing/README.md 다.
  - review/클래스_직접확인.csv               : 이름이 깨진 상품 105개의 펀드 이름 · 클래스를 사람이 정한 표
  - review/투자설명서_조사/클래스_뜻_조사결과.csv : 판매 경로를 몰랐던 클래스 32개의 가입 구분 (투자설명서 조사)
  - review/시점_조사/최종_남길기록.csv         : 공백만 다른 같은 이름 34쌍에서 남길 기록 (지울 기록 34개)

실행 (저장소 맨 위에서): python preprocessing/build.py
결과: preprocessing/work/products.pkl · units.pkl (T11 에서 JSONL 로 쓴다)
"""
import os
import re
import sys

import difflib

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)  # pipeline.py · class_codes.py 가 같은 폴더에 있다
import class_codes as cc  # noqa: E402
import pipeline as pl  # noqa: E402

REV = os.path.join(HERE, "review")
OUT = os.path.join(HERE, "work")
SRC11 = ["product_name", "product_type", "protected_type", "maturity_type", "income_rate", "risk_grade",
         "tax_type", "payment_type", "loss_rate", "liquidity_conditions", "investment_strategy"]
NOWS = lambda s: re.sub(r"\s", "", s)  # noqa: E731

# 판매 경로 표 (fund_class_codes.md) 로 바로 정해지는 클래스 (11-17, 클래스 뜻 조사 때 함께 정리)
CLASS_RULE = {
    "Class I USD": ("restricted", "클래스 부호 I = 고액 · 기관 · 법인 전용 (fund_class_codes.md)"),
}
# 운용사 직판 C-J: 같은 운용사 (한국투자) 의 C-J 정의 = 퇴직연금사업자 대상 직판 (투자설명서 조사 C8)
CJ_PENSION = ("pension", "한국투자 C-J = 수수료미징구-직판-퇴직연금 (같은 운용사 투자설명서, 조사 C8)")


def load():
    p = pl.fund_names(pl.load_products(), {**pl.BASE, "pension_rule": "class_table"})
    p = p.rename(columns={"_file": "source_file"})
    # 1. 공백만 다른 같은 이름 34쌍: 옛 기록 34개를 지운다 (지운 이름 → 남긴 이름 표는 정답 잇기에 쓴다)
    keep = pd.read_csv(os.path.join(REV, "시점_조사", "최종_남길기록.csv"), encoding="utf-8-sig")
    drop = set(keep["지울_기록"])
    assert drop <= set(p.product_name)
    alias = dict(zip(keep["지울_기록"], keep["남길_기록"]))
    p = p[~p.product_name.isin(drop)].reset_index(drop=True)
    # 1-2. 손실 값이 다른 상품 3개를 뺀다 (11-17 진우님). 이 이름을 정답으로 쓴 질문은 정답 잇기에서 needs_review 가 된다
    out = pd.read_csv(os.path.join(REV, "뺀_상품.csv"), encoding="utf-8-sig")
    assert set(out["상품_이름"]) <= set(p.product_name)
    p = p[~p.product_name.isin(set(out["상품_이름"]))].reset_index(drop=True)
    # 2. 이름이 깨진 105개: 사람이 정한 펀드 이름 · 클래스
    t = pd.read_csv(os.path.join(REV, "클래스_직접확인.csv"), encoding="utf-8-sig", dtype=str, keep_default_na=False)
    fixf, fixc = dict(zip(t["상품_이름_원문"], t["펀드_이름_고침"])), dict(zip(t["상품_이름_원문"], t["클래스_고침"]))
    p["fund_name"] = [fixf.get(n, f) for n, f in zip(p.product_name, p.fund_name)]
    p["share_class"] = [(fixc[n] or None) if n in fixc else c for n, c in zip(p.product_name, p.share_class)]
    p["share_class"] = p.share_class.replace("없음", None)  # 괄호 없는 "(없음)" 표기 4개
    p["class_status"] = ["손으로 정함" if n in fixc else ("클래스 없음" if (c is None or c in cc.NOT_CLASS) else "이름에서 읽음")
                         for n, c in zip(p.product_name, p.share_class)]
    p.loc[p.share_class.isin(cc.NOT_CLASS), "share_class"] = None
    p["fund_key"] = p.fund_name.map(NOWS).str.replace(r"형([\])])", r"\1", regex=True)
    # 3. 띄어쓰기만 다른 펀드 이름 (11개 펀드): fund_id 는 이미 같다. 보이는 이름도 펀드마다 하나로 맞춘다
    #    (가장 많이 쓰인 표기, 같으면 가나다순 앞) (11-17 진우님 "합치자")
    pick = {k: sorted(g.value_counts().items(), key=lambda x: (-x[1], x[0]))[0][0] for k, g in p.groupby("fund_key").fund_name}
    p["fund_name"] = p.fund_key.map(pick)
    return p, alias


def account(p):
    k = p.apply(lambda r: cc.pension_kind(r.share_class or "", r.product_name, r.product_type), axis=1)
    p["pension_kind"], basis = k.str[0], k.str[1]
    ch = p.share_class.fillna("(없음)").map(cc.channel)
    for n, (_, c, _f) in cc.OVERRIDE.items():
        ch[p.product_name == n] = c
    name_only = basis.str.contains("펀드 이름") & ~basis.str.contains("부호|P ")
    type_only = basis.str.contains("원본 상품 유형")
    a = pd.Series("general", index=p.index)
    a[p.pension_kind != "일반"] = "pension"
    a[name_only | type_only] = "unknown"
    lim = (p.pension_kind == "일반") & ch.isin(["기관 · 고액", "랩 · 특정금전신탁"])
    a[lim] = "restricted"
    why = basis.where(~lim, "판매 경로: " + ch)
    why = why.where(a != "general", "연금 · 가입 제한 표시 없음 (판매 경로: " + ch + ")")
    # 투자설명서로 뜻을 찾은 클래스 32개 (가입_구분_결정) 가 위 규칙보다 앞선다
    r = pd.read_csv(os.path.join(REV, "투자설명서_조사", "클래스_뜻_조사결과.csv"), encoding="utf-8-sig", dtype=str).fillna("")
    for _, x in r.iterrows():
        m = p.product_name.map(NOWS) == NOWS(x["상품_이름"])
        if m.any():
            a[m] = x["가입_구분_결정"]
            why[m] = f"투자설명서 조사: {x['가입_구분']} ({x['근거_종류']})"
    for c, (acc, w) in CLASS_RULE.items():
        a[p.share_class == c], why[p.share_class == c] = acc, w
    cj = (p.share_class == "C-J") & p.product_name.str.startswith("한국투자")
    fund_corp = cj & p.product_name.str.contains("법인")
    a[cj & ~fund_corp], why[cj & ~fund_corp] = CJ_PENSION
    a[fund_corp], why[fund_corp] = "restricted", "펀드 이름이 법인용 (C-J 는 운용사 직판)"
    p["channel"], p["account_type"], p["account_type_basis"] = ch, a, why
    return p


def product_cols(p):
    p["investment_type"] = p.product_type.where(p.product_type != "퇴직연금", None)  # 퇴직연금 폴더는 투자 유형을 모름 → null
    p["risk_grade_num"] = p.risk_grade.str.extract(r"(\d)")[0].astype("Int64")
    p["source_locations"] = [[{"source_split": "Training", "source_file": f}, {"source_split": "Validation", "source_file": f}]
                             for f in p.source_file]  # 두 폴더의 원천 8,000개는 바이트까지 같다 (T1)
    p = p.sort_values("product_name").reset_index(drop=True)
    p.insert(0, "product_id", [f"P{i:05d}" for i in range(1, len(p) + 1)])  # T11 에서 원천 파일 기준 고정 ID 로 바꿀 수 있음
    f_ids = {k: f"F{i:04d}" for i, k in enumerate(sorted(p.fund_key.unique()), 1)}
    p.insert(1, "fund_id", p.fund_key.map(f_ids))
    return p


MINOR = re.compile(r"개방형|추가형|종류형|고난도|위험등급|등급으로분류|유의하여투자")


def same_text(a, b):
    """공백 · 문장부호를 지운 정리된 투자전략 글 a, b 가 사실상 같은 글인가 (11-17 진우님: "아예 하나로").
    다른 곳이 모두 (1) 숫자가 아닌 5글자 이하이거나 (2) 한쪽에만 있는 분류 줄 · 위험등급 안내면 같은 글로 본다."""
    if a == b:
        return True
    for t, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if t == "equal":
            continue
        x, y = a[i1:i2], b[j1:j2]
        if re.search(r"\d", x + y):
            return False
        if max(len(x), len(y)) <= 5:
            continue
        if t in ("insert", "delete") and MINOR.search(x + y):
            continue
        return False
    return True


KEY_ATTR = ["risk_grade", "protected_type", "maturity_type", "tax_type", "payment_type", "liquidity_conditions"]  # 검색 문서에 들어가는 속성
ACCOUNT_KO = {"general": "일반", "pension": "연금", "restricted": "가입 제한", "unknown": "미확인"}


def _raw(s):
    s = str(s) if s is not None and s == s else ""
    s = s.replace("_x000D_", "").replace("_x0002_", "-")
    return re.sub(r"_x00[0-9A-Fa-f]{2}_", " ", s)


def product_strategy(p):
    """상품마다 검색에 쓸 투자전략과 상태를 정한다 (11-17 팀원 최종 요청 3절).
    - review/투자전략_판정.csv: 자동 검사 (위험등급 문장 · 형제 클래스와 다른 글 · 기존 표시) 에 걸린 361개를 에이전트가 읽고 판정한 표
    - available 만 검색 문서에 쓴다. 원천 글 (source.investment_strategy) 은 그대로 둔다."""
    opts = dict(pl.BASE)
    j = pd.read_csv(os.path.join(REV, "투자전략_판정.csv"), encoding="utf-8-sig", dtype=str).fillna("")
    judge = {NOWS(r["상품_이름"]): r for _, r in j.iterrows()}
    raw_by_name = {NOWS(n): _raw(t) for n, t in zip(p.product_name, p.investment_strategy)}
    clean = lambda t: pl.merge_sentences([pl.clean_strategy(t, opts)]) if t.strip() else ""  # noqa: E731
    st, fl, txt = [], [], []
    for n, t in zip(p.product_name, p.investment_strategy):
        raw, r = _raw(t), judge.get(NOWS(n))
        if not raw.strip():
            s_, f_, c_ = "missing", "원천 글 없음", ""
        elif r is None:  # 자동 검사에 안 걸림
            c_ = clean(raw)
            s_, f_ = ("available", None) if c_ else ("missing", "경고문뿐")
        elif r["판정"] == "맞음" and r["운용사_불일치"]:
            s_, f_, c_ = "needs_review", "글의 운용사 이름이 펀드와 다름", ""
        elif r["판정"] == "맞음":
            c_ = clean(raw)
            s_, f_ = ("available", "자동 검사에 걸렸으나 읽어 보니 맞음") if c_ else ("missing", "경고문뿐")
        elif r["복구_상품_이름"]:
            c_ = clean(raw_by_name[NOWS(r["복구_상품_이름"])])
            s_, f_ = "available", "같은 펀드 다른 클래스 글로 복구: " + r["복구_상품_이름"]
        elif r["판정"] == "남의 글":
            s_, f_, c_ = "missing", "다른 상품의 글로 판정, 복구할 글 없음", ""
        else:
            s_, f_, c_ = "needs_review", "글이 이 상품 것인지 확인하지 못함", ""
        st.append(s_), fl.append(f_), txt.append(c_ if s_ == "available" else "")
    # 쓸 글이 없는 상품은 투자설명서 · 공시에서 찾은 글로 채운다. 끝까지 못 찾은 펀드는 Claude 가 쓴 설명 (generated)
    # (11-17 진우님: "찾을 수 있는건 찾아서 넣고 찾아봐도 없으면 비슷하게 채워넣자")
    ext = pd.read_csv(os.path.join(REV, "투자전략_외부보완.csv"), encoding="utf-8-sig", dtype=str).fillna("")
    ext = {NOWS(r["펀드_이름"]): r for _, r in ext.iterrows()}
    for i, (f, s_) in enumerate(zip(p.fund_name, st)):
        r = ext.get(NOWS(f))
        if s_ != "available" and r is not None:
            gen = r["출처_종류"].startswith("생성")
            st[i] = "generated" if gen else "available"
            fl[i] = f"{fl[i]} → {'Claude 가 쓴 설명 (원문 아님)' if gen else '투자설명서 · 공시 글로 채움'}: {r['출처_종류']} {r['출처_URL']}"
            txt[i] = r["투자전략"]
    p["strategy_status"], p["strategy_flag"], p["strategy_use"] = st, fl, txt
    return p


def text_groups(p):
    """같은 펀드 · 가입 구분 · 핵심 속성 안에서 사실상 같은 검색용 투자전략끼리 한 무리로 묶는다. 무리 이름 = 가장 긴 글."""
    p["_c"] = p.strategy_use.map(lambda t: re.sub(r"[\s\W_]", "", t))
    rep = {}
    for _, d in p.groupby(["fund_key", "account_type", "_inv"] + KEY_ATTR):
        heads = []
        for t in sorted(d._c.unique(), key=lambda t: (-len(t), t)):
            h = next((h for h in heads if (h == "") == (t == "") and same_text(h, t)), None)
            if h is None:
                heads.append(t)
            for i in d.index[d._c == t]:
                rep[i] = h or t
    return pd.Series(rep).reindex(p.index)


def restriction_kind(basis):
    """가입 제한 근거 글을 제한 종류 하나로 줄인다. 같은 종류끼리만 한 단위로 묶는다."""
    for k, pat in [("랩 · 특정금전신탁", "랩"), ("장기주택마련", "주택마련"), ("사원복지연기금", "사원복지"),
                   ("계열사 자금", "계열사"), ("시드", "시드"), ("기관 · 법인 · 고액", "기관|고액|법인|I =")]:
        if re.search(pat, basis):
            return k
    return basis


def units(p):
    """같은 펀드 + 같은 가입 구분 + 검색 문서에 들어가는 값 (투자유형 · 속성 6개 · 검색용 투자전략) 이 같으면 한 단위.
    restricted 는 가입 제한 근거까지 같아야 하고, unknown 은 상품마다 따로 둔다 (11-17 팀원 최종 요청 4절)."""
    p = product_strategy(p)
    p["_inv"] = p.investment_type.fillna("(없음)")
    p["_t"] = text_groups(p)
    sig = p._inv + "||" + p[KEY_ATTR].astype(str).agg("|".join, axis=1) + "||" + p._t
    key = p.fund_key + "|" + p.account_type + "##" + sig
    lim = p.account_type == "restricted"
    key[lim] = key[lim] + "##" + p.account_type_basis[lim].map(restriction_kind)
    key[p.account_type == "unknown"] = "unknown##" + p.product_id[p.account_type == "unknown"]
    p["unit_key"] = key
    rows = []
    for uk, d in p.groupby("unit_key", sort=False):
        r0 = d.loc[d.strategy_use.map(len).idxmax()]  # 글이 조금씩 다르면 가장 긴 글을 쓴다
        sts = set(d.strategy_status)
        status = next(s for s in ("available", "generated", "needs_review", "missing") if s in sts)
        flags = sorted({f for f in d.strategy_flag if f})
        loss = sorted({"최대 100%" if x == "100%" else x for x in d.loss_rate})  # "100%" 는 "최대 100%" 로 (11-17 진우님)
        assert len(loss) == 1, (uk, loss)
        rows.append(dict(unit_key=uk, fund_id=r0.fund_id, fund_name=r0.fund_name, account_type=r0.account_type,
                         restriction_kind=restriction_kind(r0.account_type_basis) if r0.account_type == "restricted" else None,
                         source_product_ids=list(d.product_id), class_list=sorted({c for c in d.share_class if c}),
                         n_products=len(d), product_type=r0.product_type, investment_type=r0.investment_type,
                         risk_grade=r0.risk_grade_num, risk_grade_raw=r0.risk_grade,
                         **{c: r0[c] for c in KEY_ATTR if c != "risk_grade"}, loss_rate=loss[0],
                         strategy=r0.strategy_use or None, strategy_status=status,
                         strategy_flag=" / ".join(flags) if flags else None))
    u = pd.DataFrame(rows).sort_values(["fund_id", "account_type", "unit_key"]).reset_index(drop=True)
    u.insert(0, "unit_id", [f"U{i:05d}" for i in range(1, len(u) + 1)])
    p["unit_id"] = p.unit_key.map(dict(zip(u.unit_key, u.unit_id)))
    return p.drop(columns=["_c", "_t", "_inv"]), u


def main():
    p, alias = load()
    p = product_cols(account(p))
    p, u = units(p)
    os.makedirs(OUT, exist_ok=True)
    p.to_pickle(os.path.join(OUT, "products.pkl"))
    u.to_pickle(os.path.join(OUT, "units.pkl"))
    pd.Series(alias, name="남길_기록").rename_axis("지운_기록").to_csv(os.path.join(OUT, "지운이름_남긴이름.csv"), encoding="utf-8-sig")
    print("상품", len(p), "| 펀드", p.fund_id.nunique(), "| 단위", len(u))
    print("가입 구분 (상품)", p.account_type.value_counts().to_dict())
    print("가입 구분 (단위)", u.account_type.value_counts().to_dict())
    print("클래스 상태", p.class_status.value_counts().to_dict())
    print("investment_type null", int(p.investment_type.isna().sum()))
    print("strategy_status (단위)", u.strategy_status.value_counts().to_dict(), "| (상품)", p.strategy_status.value_counts().to_dict())
    print("단위 크기", u.n_products.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
