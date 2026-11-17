"""Build the corpus and labels for one set of preprocessing choices.

build(opts) returns (products, units, queries). Each option is one of the
ambiguous decisions; experiments flip one option at a time against BASE.
Raw files under data/raw are only read, never changed.
"""

import glob
import json
import re
import unicodedata

import pandas as pd

import os

# 저장소의 data/raw 에 AI Hub 원천 (Training · Validation 폴더) 을 둔다
RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

BASE = dict(
    key_rule="current",      # "current": strip last (...) + attach classless; "english": strip only if it has A-Z
    hyung_merge=True,        # ignore a trailing 형 inside the type bracket when grouping
    display_type=True,       # keep a type-only tail like (채권혼합) in the shown fund name
    manager_rule=True,       # drop strategy texts whose first line names another asset manager
    all_foreign="empty",     # units whose texts all belong to another manager: "empty" or "keep"
    owner_rule=True,         # drop a shared text that names another fund and none of this fund's own name words (added 11-09)
    strip_x000d=True,        # remove the Excel line-break code "_x000D_" (added 11-09; the 11-08 package had both off)
    pension_rule="class_table",  # "type": product_type == 퇴직연금 (11-08); "type_or_class": or the class name says 퇴직·IRP·RP·사전지정·연금저축·개인연금; "class_table": class_codes.py (11-09, final)
    use_strategy=True,       # False: search_text has the name and the 8 attributes only (tried 11-09)
    pension_text=True,       # 연금 units write 퇴직연금 in the product-type slot of search_text (11-09)
    strategy_mode="merge",   # "pick": one text per unit (11-08); "merge": join this fund's texts sentence by sentence, without boilerplate (11-09)
    merge_same_sides=True,   # put back together the two sides of a fund when they carry the same information (11-09 final; False gives the 3,080-unit variant)
    mixed_mark=False,        # True: a 일반 unit that also holds pension classes says "(연금 클래스 있음)" in search_text (tried 11-09)
    strategy_level="unit",   # "unit": pick a text per unit; "fund": one text per fund, shared by its 연금 and 일반 units (tried 11-09)
    grade_sentence="remove", # "N등급으로 분류하였습니다" sentence: "remove" or "keep"
    class_list=False,        # append the unit's class list to search_text
    newline="space",         # line breaks: "space", "join" (sentence ends -> space, others joined), "kiwi" (join + Kiwi spacing)
)

N = lambda s: unicodedata.normalize("NFC", s) if isinstance(s, str) else s
_nows = lambda s: re.sub(r"\s+", "", s)

_TAIL = re.compile(r"\s*\(((?:[^()]|\([^()]*\))*)\)\s*$")
_TYPE_ONLY = re.compile(r"^[\s\-·]*(?:주식|채권|혼합|재간접|파생|형|[\s\-·])+$")
_KIWI = None
_NAME_TYPE = re.compile(r"[\(\[][^\)\]]*(?:주식|채권|혼합|재간접|파생)")
# "4등급으로 분류하였습니다." / "(3)등급으로" / "4 등급(보통위험)으로 분류됩니다" — the whole verb, spaces allowed (PDF text).
_RISK = re.compile(r"\(?\s*(\d)\s*\)?\s*등\s*급(?:\s*\([^)]{0,80}\))?\s*으\s*로\s*분\s*류"
                   r"(?:\s*하\s*였\s*습\s*니\s*다|\s*됩\s*니\s*다)?\s*\.?")

# Asset-manager rule (checked on 2026-11-07 data, see opcheck.py in the scratchpad).
_ALIAS = {"한국투자밸류": ["한국밸류"], "케이씨지아이": ["KCGI"], "마이다스에셋": ["마이다스"], "삼성액티브": ["삼성"],
          "얼라이언스번스틴": ["AB"], "프랭클린템플턴": ["프랭클린", "템플턴"], "신한비엔피파리바": ["신한BNPP", "신한"],
          "하나유비에스": ["하나UBS", "하나"], "현대인베스트먼트": ["현대인베스트"], "알리안츠글로벌인베스터스": ["알리안츠"],
          "브이아이피": ["VIP"], "유경PSG": ["유경"], "스트스프링": ["이스트스프링"], "한국": ["한국"], "리": ["우리"],
          "움": ["키움"], "브이": ["브이"]}
_OP = re.compile(r"^\s*[㈜(주)]*\s*([^\s]{1,20}?)\s?(?:투자자산운용|자산운용|투자신탁운용|자산 운용)")

_PENSION_CLASS = r"퇴직|IRP|사전지정|연금저축|개인연금|(?:^|[-(])RP"

ATTRS = ["product_type", "risk_grade", "protected_type", "maturity_type", "tax_type",
         "payment_type", "loss_rate", "liquidity_conditions"]


def _read(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as fp:
            d = json.load(fp)
        rows.append({k: (N(v) if isinstance(v, str) else v) for k, v in d.items()} | {"_file": N(path.split("/")[-1])})
    return rows


def load_products():
    return pd.DataFrame(_read(f"{RAW}/Training/01.원천데이터/*/*.json*"))


# ------------------------------------------------------------ fund names
def _split_tail(name):
    m = _TAIL.search(name)
    return (name[:m.start()].strip(), m.group(1)) if m else (name.strip(), None)


def fund_names(p, opts):
    p = p.copy()
    split = p["product_name"].map(_split_tail)
    head, tail = split.str[0], split.str[1]
    if opts["key_rule"] == "english":
        keep = tail.isna() | ~tail.fillna("").str.contains("[A-Za-z]")
        p["fund_name"] = head.where(~keep, p["product_name"].str.strip())
        p["share_class"] = tail.where(~keep, None)
        p["fund_key"] = p["fund_name"].map(_nows)
    else:
        p["fund_name"], p["share_class"] = head, tail
        p["fund_key"] = p["fund_name"].map(_nows)
        # Classless product: its whole name already is another product's fund key.
        whole = p["product_name"].map(_nows)
        classless = whole.isin(set(p["fund_key"])) & (whole != p["fund_key"])
        p.loc[classless, "fund_name"] = p.loc[classless, "product_name"].str.strip()
        p.loc[classless, "fund_key"] = whole[classless]
        p.loc[classless, "share_class"] = "(없음)"
    p["_tail"] = tail
    if opts["hyung_merge"]:
        p["fund_key"] = p["fund_key"].str.replace(r"형([\])])", r"\1", regex=True)
    p["pension"] = p["product_type"].eq("퇴직연금")
    if opts.get("pension_rule", "type") == "type_or_class":
        # e.g. "KB 액티브 배당 증권 자투자신탁(주식)(C-퇴직e)" has product_type 주식형 but is a 퇴직연금 class.
        p["pension"] |= p["share_class"].fillna("").str.contains(_PENSION_CLASS, regex=True)
    if opts.get("pension_rule", "type") == "class_table":
        # 11-09: class codes read with the source-checked table in class_codes.py (P, R, RP, PRS, Korean labels, ...).
        import class_codes
        k = p.apply(lambda r: class_codes.pension_kind(r["share_class"] or "", r["product_name"], r["product_type"]), axis=1)
        p["pension_kind"], p["pension_basis"] = k.str[0], k.str[1]
        p["pension"] = p["pension_kind"].ne("일반")
    p["unit_key"] = p["fund_key"] + "|" + p["pension"].map({True: "연금", False: "일반"})
    return p


def _display_name(d, opts):
    name = d["fund_name"].iloc[0]
    if not opts["display_type"] or opts["key_rule"] == "english":
        return name
    tails = d["_tail"].dropna().unique()
    # Only when the type word was cut away and no other type word is left in the name.
    if len(d) == d["_tail"].notna().sum() and len(tails) == 1 and _TYPE_ONLY.match(tails[0]) \
            and d["share_class"].iloc[0] != "(없음)" and not _NAME_TYPE.search(name):
        return f"{name}({tails[0]})"
    return name


# ------------------------------------------------------------ strategy text
def text_manager(t):
    m = _OP.search(t)
    return re.sub(r"^[㈜(주)]+", "", m.group(1)) if m else None


def manager_mismatch(t, fund_key):
    o = text_manager(t)
    if o is None:
        return False
    return not any(fund_key.startswith(k) for k in [o] + _ALIAS.get(o, []))


# Text-ownership rule (11-09). A text shared by several funds belongs to the fund whose name words it mentions.
# Example: the "코스피200 선물 인버스 2X" text also sits under "NH-Amundi 하나로 TDF 2025"; it names 인버스/레버리지,
# never 하나로/TDF/2025, so it is dropped there. Name words skip fund-structure words and share-class letters.
_OWN_STOP = ["운용전환완료일이후", "운용전환완료일부터", "운용전환일이후", "자투자신탁", "모투자신탁", "투자신탁", "투자회사",
             "신탁", "증권", "전환형", "종류형", "목표전환", "펀드", "주식", "채권", "혼합", "재간접", "파생",
             "형", "호", "제", "자", "의", "子"]
_OWN_SKIP_LATIN = {"UH", "H", "E", "C", "A", "S", "P", "W", "F", "I", "J"}
# Words that appear in many texts: finding one is not evidence that the text belongs to another fund.
_OWN_GENERIC = {"글로벌", "코리아", "플러스", "인덱스", "아시아", "미국", "배당", "인컴", "국공채"}


def _name_words(name):
    out = []
    for m in re.finditer(r"[가-힣]+|[A-Za-z]+|\d+", name):
        r = m.group()
        if r.isdigit():
            if len(r) >= 2:
                out.append(r)
        elif re.match(r"[A-Za-z]", r):
            if r.upper() not in _OWN_SKIP_LATIN and len(r) >= 2:
                out.append(r.upper())
        else:
            for s in _OWN_STOP:
                r = r.replace(s, " ")
            out += [x for x in r.split() if len(x) >= 2]
    return out


def _distinct_words(a, b):
    """Name words that tell fund a from fund b, and b from a. Near-identical words keep only their differing part."""
    import difflib
    ta, tb = _name_words(a), _name_words(b)
    sa, sb = [t for t in ta if t not in tb], [t for t in tb if t not in ta]
    da, db, used = [], [], set()
    for x in sa:
        best = max(((difflib.SequenceMatcher(None, x, y).ratio(), y) for y in sb if y not in used), default=(0, None))
        if best[0] >= 0.5 and not x.isdigit():
            y = best[1]
            used.add(y)
            sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op != "equal":
                    for frag, out in ((x[i1:i2], da), (y[j1:j2], db)):
                        if len(frag) >= 2 and not frag.isdigit():
                            out.append(frag)
        else:
            da.append(x)
    db += [y for y in sb if y not in used]
    return da, db


def _in_master_name(T, end):
    """True if the text right after position end runs into "모투자신탁" within a few name characters.
    A feeder fund often invests in a master fund with another name ("…차이나그로스증권모투자신탁에 90% 투자")."""
    j = T.find("모투자신탁", end)
    return 0 <= j - end <= 12 and not re.search(r"[.,%]\D|다|년|을|를|에|은|는", T[end:j])


def _found(piece, T, skip_master):
    return any(not (skip_master and _in_master_name(T, m.end())) for m in re.finditer(re.escape(piece), T))


def _word_in(w, text, skip_master=False):
    T = text.upper()
    if w.isdigit():  # a fund number like 30 or 2025, not a share like 30%
        return any(not (skip_master and _in_master_name(T, m.end()))
                   for m in re.finditer(rf"(?<!\d){w}(?![\d%])", T))
    if len(w) <= 3 or _found(w, T, skip_master):
        return _found(w, T, skip_master)
    return sum(_found(w[i:i + 3], T, skip_master) for i in range(len(w) - 2)) >= 2


def owner_of(fund_key, others, text):
    """Return the other fund this text seems to belong to, or None."""
    t = _nows(text)
    for b in sorted(others):
        da, db = _distinct_words(fund_key, b)
        hb = [x for x in db if x not in _OWN_GENERIC and _word_in(x, t, skip_master=True)]
        if hb and da and not any(_word_in(x, t) for x in da):
            return b
    return None


# Units whose text a person checked against the real fund page and found wrong (strategy emptied, flag 손으로 비움).
MANUAL_EMPTY = {
    # 11-09 진우님: hanwhafund.co.kr fund page shows REITs assets; the text is the 멀티에셋크루즈5.0 fund's.
    "한화100세시대퇴직연금글로벌리츠증권자투자신탁2호(채권혼합)|연금",
}


# Sentence merge (11-09, 진우님 (가)). No new words are added: a kept sentence is original text, or original text with
# the 「예금자보호법 … 실적배당상품이며,」 clause cut out (_DEPOSIT). Warning-only sentences are dropped.
# Split after "다." even with no space ("추구합니다.그러나"), after "니다" + space, before bullets,
# after a "…참조" note written without a period, and before a "그러나 상기의/이 …" that follows a sentence written
# without a period ("…목적으로 함 그러나, 상기의").
_SENT = re.compile(r"(?<=다\.)\s*(?=\S)|(?<=니다)\s+(?=\S)|(?<=바랍니)\s+(?=\S)|(?<=참조)\s+(?=\S)|\s*(?=[▪※■‧ㆍ])|\s+(?=-\s)"
                   r"|\s+(?=그러나\s*,?\s*(?:상기|이|당사))")
def _sp(word):
    """Regex for a word that PDF text may break with spaces anywhere ("실적배당상 품")."""
    return r"\s*".join(map(re.escape, word))


_DEPOSIT = re.compile(rf"((이\s*)?{_sp('집합투자증권은')}\s*)?[｢「『\[(]?\s*{_sp('예금자보호법')}\s*[｣」』\])]?\s*에\s*({_sp('따라')}|{_sp('의해')}|{_sp('의한')})?[^.]*?"
                      rf"{_sp('실적배당상품')}\s*({_sp('입니다')}\s*\.?|{_sp('이며')}\s*,?|{_sp('으로')}\s*,?)")
_BOILER = re.compile("|".join(_sp(w) for w in [
    "반드시달성된다는보장", "보장하지아니", "보장은없습니다", "분류등급과는상이", "집합투자업자가분류한것", "금융위원회가",
    "기재사항이진실", "간이투자설명서", "참조하시기바랍니다", "법령에서정한절차", "공시될예정", "은행의예금과달리"])
    + rf"|{_sp('투자설명서를')}\s*({_sp('반드시')}\s*)?({_sp('읽')}|{_sp('참조')}|{_sp('참고')})")


_BOILER2 = re.compile("|".join(_sp(w) for w in [
    "투자원금의손실이발생할수있", "투자위험등급을확인하시고", "소규모펀드", "확인하시기바랍니다", "투자결정을하시기바랍니다",
    "예금보험공사가보호하지않는", "투자원리금전액이보장", "투자에신중을기하여", "실적배당상품",
    "보장하지않", "과거의운용실적", "운용성과를보장"]))
# Pointers to the prospectus / trust contract are dropped (11-09 진우님). A ※ note with such a pointer anywhere is dropped;
# otherwise the pointer word and 참조 / 참고하 / 확인하시기 must be within 40 letters, counted without spaces.
_REF_HEAD = "|".join(_sp(w) for w in ["투자설명서", "신탁계약서", "모집합투자기구에관한사항", "모펀드에관한사항", "제2부", "별첨",
                                       "인터넷홈페이지"])
_REF_TAIL = "|".join(_sp(w) for w in ["참조", "참고하", "확인하시기"])
_NOTE_REF = re.compile(rf"^\s*※.*({_REF_HEAD}).*({_REF_TAIL})")
_REF = re.compile(rf"({_REF_HEAD})(?:\s*\S){{0,40}}?\s*({_REF_TAIL})")
# Words that show a sentence says what the fund invests in; such a sentence is kept even if it also warns.
_CONTENT = re.compile("|".join(_sp(w) for w in ["투자하", "투자대상", "주로", "추구", "편입", "운용전략", "지수", "헤지", "모투자신탁"]) + "|%")

_CONTENT_STRONG = re.compile("|".join(_sp(w) for w in ["투자하", "투자대상", "주로", "추구", "편입"]) + "|%")


_SUMMARY_HEAD = re.compile(_sp("중요사항을발췌"))  # "이 요약정보는 '…'의 투자설명서의 내용 중 중요사항을 발췌 요약한 정보를 담고 있습니다"
_NOTE_DETAIL = re.compile(rf"^\s*※.*{_sp('자세한')}.*({_REF_HEAD})")  # a ※ note sending the reader to the prospectus, even if cut off


def _boiler(sent):
    """A warning sentence that says nothing about what the fund invests in."""
    if _NOTE_REF.search(sent) or _SUMMARY_HEAD.search(sent) or _NOTE_DETAIL.search(sent):
        return True
    # 11-09 진우님 ("오류가 나는 건 없애자"): any pointer to the prospectus goes, and a warning sentence stays only if it
    # says what the fund invests in (risk names such as 운용전략위험 · 지수추적오차위험 · 환헤지위험 do not count).
    if _REF.search(sent):
        return True
    if _BOILER.search(sent) or _BOILER2.search(sent):
        return not _CONTENT_STRONG.search(sent)
    # a short sentence with a strong disclaimer ("그러나 환헤지 전략이 반드시 달성된다는 보장은 없습니다.") has no room for content
    if _BOILER.search(sent) and len(re.sub(r"[\W_]", "", sent)) < 60:
        return True
    return bool(_BOILER.search(sent) or _BOILER2.search(sent) or _REF.search(sent)) and not _CONTENT.search(sent)


def merge_sentences(texts):
    return " ".join(merge_sentence_list(texts))


def merge_sentence_list(texts):
    """Join cleaned texts of one unit: drop boilerplate sentences, keep each other sentence once, in order.
    A sentence that is (almost) the same as one already kept is skipped."""
    import difflib
    out, keys = [], []
    for t in texts:
        t = _DEPOSIT.sub(" ", t)
        for sent in _SENT.split(t):
            sent = sent.strip(" -▪")
            key = re.sub(r"[\W_]", "", sent)
            if len(key) < 8 or _boiler(sent):
                continue
            # a kept sentence that already contains this one, or is almost the same: skip this one
            nums = re.findall(r"\d+", key)
            if any(key in k or (abs(len(k) - len(key)) < 0.2 * len(key) and re.findall(r"\d+", k) == nums
                                and difflib.SequenceMatcher(None, k, key, autojunk=False).ratio() > 0.9)
                   for k in keys):
                continue
            # this sentence contains a shorter kept one: keep the longer sentence in its place
            inside = [i for i, k in enumerate(keys) if k in key]
            if inside:
                keys[inside[0]], out[inside[0]] = key, sent
                for i in reversed(inside[1:]):
                    del keys[i], out[i]
                continue
            keys.append(key)
            out.append(sent)
    return out


def pick_strategy(p, opts, unrelated_name_sim=0.6):
    """One strategy text per unit. Same rules as prep.step2 plus the manager rule."""
    import difflib
    p = p.copy()
    if opts.get("strip_x000d"):
        # Excel escape codes (11-09 Fable review): _x000D_ = line break, _x0002_ sits where a hyphen belongs
        # ("[주식혼합_x0002_재간접형]"), any other _x00XX_ becomes a space.
        p["investment_strategy"] = (p["investment_strategy"].str.replace("_x000D_", "", regex=False)
                                    .str.replace("_x0002_", "-", regex=False)
                                    .str.replace(r"_x00[0-9A-Fa-f]{2}_", " ", regex=True))
    p["_s"] = p["investment_strategy"].map(_nows)
    # Who really uses each text: copies that name another manager do not count.
    bad = p.apply(lambda r: opts["manager_rule"] and manager_mismatch(r["investment_strategy"], r["fund_key"]), axis=1)
    funds_by_text = p[~bad].groupby("_s")["fund_key"].agg(lambda s: sorted(set(s))).to_dict()
    all_funds_by_text = p.groupby("_s")["fund_key"].agg(lambda s: set(s)).to_dict()
    rows = []
    for uk, d in p.groupby("unit_key"):
        if uk in MANUAL_EMPTY:
            rows.append({"unit_key": uk, "strategy": "", "strategy_flag": "손으로 비움"})
            continue
        counts = d["_s"].value_counts()
        texts = list(counts.index)
        fund = d["fund_key"].iloc[0]
        orig = {t: d.loc[d["_s"] == t, "investment_strategy"].iloc[0] for t in texts}
        mgr_bad = {t for t in texts if opts["manager_rule"] and manager_mismatch(orig[t], fund)}
        if opts.get("owner_rule"):
            # A text shared with other funds that names one of them, and none of this fund's own name words.
            mgr_bad |= {t for t in texts if t and owner_of(fund, all_funds_by_text[t] - {fund}, orig[t])}

        def is_foreign(t):
            others = [f for f in funds_by_text.get(t, []) if f != fund]
            return any(difflib.SequenceMatcher(None, fund, f).ratio() < unrelated_name_sim for f in others)

        flag = ""
        if mgr_bad and len(mgr_bad) == len(texts):
            if opts["all_foreign"] == "empty":
                rows.append({"unit_key": uk, "strategy": "", "strategy_flag": "남의 글뿐"})
                continue
            mgr_bad = set()
            flag = "남의 글뿐(유지)"
        if len(texts) == 1:
            shared = is_foreign(texts[0])
            rows.append({"unit_key": uk, "strategy": orig[texts[0]], "strategy_flag": flag or ("공유글" if shared else "")})
            continue
        drop = set(mgr_bad)
        for t in texts:
            if t in drop:
                continue
            if any(t != o and t in o for o in texts) or is_foreign(t):
                drop.add(t)
        left = [t for t in texts if t not in drop]
        if opts.get("strategy_mode") == "merge":
            # keep every text of this fund (not another manager's, not shared with a very different fund);
            # a text inside a longer one is fine: repeated sentences are dropped in merge_sentences
            keep = [t for t in texts if t not in mgr_bad and not is_foreign(t)] or left
            if keep:
                keep.sort(key=lambda t: (counts[t], len(t)), reverse=True)
                rows.append({"unit_key": uk, "strategy": "\x00".join(orig[t] for t in keep),
                             "strategy_flag": flag or ("글 합침" if len(keep) > 1 else "")})
                continue
        if not left:
            left = [max((t for t in texts if t not in mgr_bad), key=len)]
            flag = flag or "손검토필요"
        left.sort(key=lambda t: (counts[t], len(t)), reverse=True)
        if len(left) >= 2:
            sim = min(difflib.SequenceMatcher(None, left[0], t).ratio() for t in left[1:])
            if sim < 0.5 or counts[left[0]] == counts[left[1]]:
                flag = flag or "손검토필요"
        rows.append({"unit_key": uk, "strategy": orig[left[0]], "strategy_flag": flag})
    return pd.DataFrame(rows)


def clean_strategy(t, opts):
    if not t:
        return ""
    if opts["grade_sentence"] == "remove":
        # The text is sometimes pasted twice, so remove every occurrence.
        # A sentence may also start right after "…바랍니다" with no period.
        while (m := _RISK.search(t)):
            before = t[:m.start()]
            cut = max(before.rfind(s) for s in (". ", ".\n", "다 ", "다\n"))
            start = cut + 1 if cut >= 0 else 0
            t = t[:start].rstrip() + " " + t[m.end():].lstrip()
    if opts["newline"] == "join":
        # A break after a sentence end becomes a space; a break inside a line is joined.
        t = re.sub(r"(?<=[.다:])\s*\n", " ", t).replace("\n", "")
    elif opts["newline"] == "kiwi":
        from kiwipiepy import Kiwi
        global _KIWI
        if _KIWI is None:
            _KIWI = Kiwi()
        t = _KIWI.space(re.sub(r"(?<=[.다:])\s*\n", " ", t).replace("\n", ""))
    else:
        t = t.replace("\n", " ")
    return re.sub(r"[ \t]+", " ", t).strip()


# ------------------------------------------------------------ units
def build_units(p, opts):
    if opts.get("strategy_level") == "fund":
        f = pick_strategy(p.assign(unit_key=p["fund_key"]), opts).set_index("unit_key")
        keys = p.drop_duplicates("unit_key").set_index("unit_key")["fund_key"]
        strat = f.reindex(keys.values).set_axis(keys.index)
    else:
        strat = pick_strategy(p, opts).set_index("unit_key")
    rows = []
    for uk, d in p.groupby("unit_key"):
        a = {c: d[c].mode().iloc[0] for c in ATTRS}
        name = _display_name(d, opts)
        # some class names contain ", " so "; " joins them. 11-09 진우님 (가): values that are not classes
        # ((없음), 주가연계증권, 신탁, 채권혼합 ...) are left out; a unit with no real class gets "".
        import class_codes
        classes = "; ".join(sorted(c for c in d["share_class"].fillna("(없음)").unique() if c not in class_codes.NOT_CLASS))
        raw = strat.at[uk, "strategy"]
        flag_now = strat.at[uk, "strategy_flag"]
        if opts.get("strategy_mode") == "merge" and raw:
            s = merge_sentences([clean_strategy(t, opts) for t in raw.split("\x00")])
            if not s:
                flag_now = "경고문뿐"  # every attached text was boilerplate only
        else:
            s = clean_strategy(raw, opts)
        if not opts.get("use_strategy", True):
            s = ""
        # A 연금 unit that got its side from the class name (e.g. "C-퇴직e") keeps its raw product_type in the
        # column, but its text says 퇴직연금 in that slot like every other 연금 unit (else both sides read the same).
        # 11-09 (Fable review, 진우님 (가)): the word follows the unit's classes: only 퇴직연금 classes -> 퇴직연금,
        # only 연금저축 classes -> 연금저축, mixed or unknown kind -> 연금.
        ptype = a["product_type"]
        if opts.get("pension_text", True) and d["pension"].iloc[0]:
            kinds = set(d["pension_kind"]) if "pension_kind" in d else {"퇴직연금"}
            ptype = "퇴직연금" if kinds == {"퇴직연금"} else "연금저축" if kinds == {"연금저축"} else "연금"
        if opts.get("mixed_mark") and not d["pension"].iloc[0] and "pension_kind" in d and (d["pension_kind"] != "일반").any():
            ptype = f"{ptype} (연금 클래스 있음)"  # 11-09 test: a merged unit that also sells pension classes
        parts = [name, ptype, f"위험등급 {a['risk_grade']}", a["protected_type"], a["maturity_type"],
                 f"세제 혜택 {a['tax_type']}", a["payment_type"], f"손실 {a['loss_rate']}", a["liquidity_conditions"]]
        if opts["class_list"]:
            parts.append(f"클래스 {classes}")
        text = ". ".join(parts) + ". " + s
        rows.append({"unit_key": uk, "fund_name": name, "pension": d["pension"].iloc[0], "class_list": classes,
                     "n_products": len(d), **a, "strategy": s, "strategy_flag": flag_now,
                     "search_text": text.strip()})
    u = pd.DataFrame(rows).sort_values("unit_key").reset_index(drop=True)
    u.insert(0, "unit_id", [f"U{i:04d}" for i in range(1, len(u) + 1)])
    return u


# ------------------------------------------------------------ labels
_QUOTED = re.compile(r"[\"“”']([^\"“”']{5,})[\"“”']")


def fix_label_names(names, known):
    """Return (fixed names that exist as products, names that could not be matched)."""
    if len(names) > 5 and all(len(x) <= 1 for x in names):  # list exploded into characters
        names = _QUOTED.findall("".join(names))
    known_nows = {_nows(k): k for k in known}
    ok, bad = [], []
    for n in names:
        n = N(n).replace("일고객덱스", "일본인덱스").replace("ㅐ ", "")
        if n in known:
            ok.append(n)
        elif _nows(n) in known_nows:
            ok.append(known_nows[_nows(n)])
        else:
            bad.append(n)
    return ok, bad


def load_queries(known, seed=42, n_val=801):
    from sklearn.model_selection import train_test_split
    out = []
    for folder, pat in [("Training", "TL_1*"), ("Validation", "VL_1*")]:
        for d in _read(f"{RAW}/{folder}/02.라벨링데이터/{pat}/*.json*"):
            ok, bad = fix_label_names(d["product_names"], known)
            out.append({"folder": folder, "file": d["_file"], "cot_id": d["cot_id"], "query_type": d["query_type"],
                        "question": d["question"], "gold_products": ok, "unmatched": bad})
    q = pd.DataFrame(out)
    t = q[q["folder"] == "Training"]
    _, val_idx = train_test_split(t.index, test_size=n_val, stratify=t["query_type"], random_state=seed)
    q["split"] = "train"
    q.loc[val_idx, "split"] = "validation"
    q.loc[q["folder"] == "Validation", "split"] = "test"
    return q


def build(opts=None):
    opts = {**BASE, **(opts or {})}
    p = fund_names(load_products(), opts)
    u = build_units(p, opts)
    if opts.get("merge_same_sides"):
        # 11-09 option: a split fund whose two sides carry the same information
        # (same strategy text and the same 8 attributes apart from product_type) goes back to one 일반 unit.
        # Each product keeps its own pension_kind in product.csv.
        # 11-09 (Fable review, 진우님 (가)): texts are compared without spaces, punctuation, "분류" and "형",
        # so two sides that differ only in such letters are not split.
        cols = [c for c in ATTRS if c != "product_type"] + ["_strategy_cmp"]
        f = u.assign(fund_key=u["unit_key"].str.split("|").str[0],
                     _strategy_cmp=u["strategy"].str.replace(r"[\W_]|분류|형", "", regex=True))
        twin = f.groupby("fund_key").filter(lambda d: len(d) == 2)
        same = twin.groupby("fund_key").apply(lambda d: all(d[c].nunique() == 1 for c in cols))
        merge = set(same[same].index)
        p.loc[p["fund_key"].isin(merge), "pension"] = False
        p["unit_key"] = p["fund_key"] + "|" + p["pension"].map({True: "연금", False: "일반"})
        u = build_units(p, opts)
    p = p.merge(u[["unit_key", "unit_id"]], on="unit_key")
    return p, u
