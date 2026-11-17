"""Read a share-class name into three plain columns: 연금 구분, 판매 경로, 수수료 방식.

Meanings come from asset managers' trust contracts and prospectuses saved on 2026-11-09
(see the class-code table in the package README). A letter can mean different things at
different managers, so the order is: Korean label in the name, then a code that the sources
agree on, then the fund name / product_type, else 미확인.
"""

import re

NOT_CLASS = {"(없음)", "주가연계증권", "신탁", "투자신탁", "투자회사", "채권혼합", "채권", "주식", "주식형", "재간접",
             "모자형", "추가형", "적립식", "종류형", "운용", "법인 신종 MMF", "H", "UH", "E",
             # values that are a fund name, a fund type or a table heading, not a class (checked by eye, 11-09)
             "채권혼합-재간접", "채권형", "주식혼합-재간접", "증권(혼합채권형)", "재형펀드", "AA-이상", "종류(클래스)",
             "클래스별", "모투자신탁[채권]", "신종 법인용 MMF1호", "PAM 부동산투자신탁3호",
             "한국투자 인컴추구 증권 모투자신탁(채권혼합-재간접)",
             "투자신탁, 단기금융(MMF)(법인용), 개방형(중도환매가능), 추가형(추가납입가능)"}


# Classes checked one by one against the manager's own document (11-09, 진우님: trust the prospectus).
# 우리지속가능ESG: the dataset labels are shifted by one class; values below follow the 2026-10-29 prospectus
# (https://www.wooriam.kr/investment/fund-view/90ae814d093c11ec9).
_W = "우리지속가능ESG증권자투자신탁1호(주식)"
OVERRIDE = {
    f"{_W}(오프라인-퇴직연금(C-P1e))": ("연금저축", "온라인", "미징구 (C)"),
    f"{_W}(오프라인-퇴직연금(C-Pe))": ("퇴직연금", "온라인", "미징구 (C)"),
    f"{_W}(오프라인-퇴직연금(S))": ("일반", "온라인슈퍼", "후취 (B · S)"),
    f"{_W}(오프라인-퇴직연금(S-P))": ("연금저축", "온라인슈퍼", "미징구 (C)"),
    f"{_W}(온라인(C-P))": ("퇴직연금", "오프라인", "미징구 (C)"),
    f"{_W}(오프라인(B))": ("일반", "오프라인", "선취 (A)"),
    # 신한밸런스프로적극형: 신탁계약서 제3조③ · 투자설명서 (2026-06-24) 의 종류 표 (11-09 확인).
    "신한밸런스프로적극형증권투자신탁[주식혼합-재간접](C-O)": ("퇴직연금", "디폴트옵션", "미징구 (C)"),
    "신한밸런스프로적극형증권투자신탁[주식혼합-재간접](C-pe)": ("연금저축", "온라인", "미징구 (C)"),
    # 피델리티 EMEA: 문서에 RPS 는 없고 같은 자리에 PRS-e · S-PRS (연금저축) 가 있다. 글자가 바뀐 오기로 본다 (추정).
    "피델리티 EMEA 증권 자투자신탁 (주식-재간접)(RPS-e)": ("연금저축", "온라인", "미징구 (C)"),
    "피델리티 EMEA 증권 자투자신탁 (주식-재간접)(S-RPS)": ("연금저축", "온라인슈퍼", "미징구 (C)"),
    # 11-09: both funds are 연금저축-account-only as a whole (KCGI 규약 제3조의2①, 2025-01-03 · 투자설명서 2026-02-03;
    # 삼성 신탁계약서 제53조① 2024-08-21 판 · 투자설명서 2026-06-17). Checked by a Claude agent and Gemini Argon.
    "KCGI 코리아연금증권전환형자투자신탁 1 호 [채권](A)": ("연금저축", "오프라인", "선취 (A)"),
    "KCGI 코리아연금증권전환형자투자신탁 1 호 [채권](C-1)": ("연금저축", "오프라인", "미징구 (C)"),
    "KCGI 코리아연금증권전환형자투자신탁 1 호 [채권](Ce)": ("연금저축", "온라인", "미징구 (C)"),
    "KCGI 코리아연금증권전환형자투자신탁 1 호 [채권](S-P)": ("연금저축", "온라인슈퍼", "미징구 (C)"),
    # 삼성: the documents list only C, Ce, Cg, S-P. Ae · Cf · Cw · Cp · Cpe in the data are not in them (left as read).
    "삼성클래식중국본토중소형FOCUS연금증권자투자신탁[주식](C)": ("연금저축", "오프라인", "미징구 (C)"),
    "삼성클래식중국본토중소형FOCUS연금증권자투자신탁[주식](Ce)": ("연금저축", "온라인", "미징구 (C)"),
    "삼성클래식중국본토중소형FOCUS연금증권자투자신탁[주식](S-P)": ("연금저축", "온라인슈퍼", "미징구 (C)"),
}


def _manager(product_name):
    for k in ["NH-Amundi", "대신", "DB", "삼성", "미래에셋", "KB", "교보악사", "우리", "한화", "신한", "피델리티", "IBK"]:
        if product_name.replace(" ", "").startswith(k.replace(" ", "")):
            return k
    return ""


def pension_kind(cls, product_name, product_type):
    """퇴직연금 / 연금저축 / 연금(종류 미확인) / 일반, and the rule that decided it."""
    if product_name in OVERRIDE:
        return OVERRIDE[product_name][0], "운용사 투자설명서로 하나씩 확인 (OVERRIDE)"
    c = "" if cls in NOT_CLASS else (cls or "")
    # 1. Korean label in the class name.
    if re.search(r"퇴직|사전지정|디폴트", c):
        return "퇴직연금", "클래스 이름의 한글 (퇴직 · 디폴트옵션)"
    if re.search(r"연금저축|개인연금", c):
        return "연금저축", "클래스 이름의 한글 (연금저축 · 개인연금)"
    if re.search(r"\(연금\)", c):
        return "연금저축", "클래스 이름의 '(연금)' (교보악사: 연금저축계좌)"
    code = _core(c)[1] if c else ""
    m = _manager(product_name)
    # 2. Codes the sources agree on.
    if re.search(r"RPS", code):
        code = "?"  # RPS-e, S-RPS: no source found; decided by fund name / product_type below
    if re.search(r"(?:^|[-(])R[Pp]|^Crp|^RP|J-RP", code):
        return "퇴직연금", "RP 부호 (퇴직연금)"
    if code in ("O", "0", "○") or re.fullmatch(r"O-퇴직|O", code):
        return "퇴직연금", "O 부호 (퇴직연금 사전지정운용 · 디폴트옵션 전용)"
    if re.search(r"PRS", code):
        return "연금저축", "PRS 부호 (피델리티: 연금저축계좌)"
    if re.fullmatch(r"(?:[ACS]-|C)R(?:e|E|p)?|C-r|C-re|A-RJ", code):
        return "퇴직연금", "R 부호 (퇴직연금; 삼성ABF 의 맨 R 은 제외)"
    if m == "삼성" and re.fullmatch(r"Cp(?:e|-f|f)?", code):
        return "퇴직연금", "삼성 Cp 부호 (삼성: 퇴직연금)"
    p12 = re.search(r"P(?:e)?([12])|Pe?([12])", code)
    if p12 and m in ("NH-Amundi", "대신", "DB"):
        n = p12.group(1) or p12.group(2)
        return ("연금저축" if n == "1" else "퇴직연금"), f"{m} P{n} 부호 ({m}: P1 연금저축 · P2 퇴직연금)"
    # 11-09: lowercase p (트러스톤 Cp, 플러스 C-p1, 흥국 C-p2, 대신 cp) was missed before; same reading as P.
    if (re.search(r"(?:^|[-(])[A-Z]?P(?:\d|e|E|-|$)|^[A-Z]P|^P|[Cc]P|-p(?:e|$)|C-pe", code)
            or re.match(r"[Cc]-?p(?:\d|e|E|-|$)", code)) and "PRS" not in code:
        # 11-09: the kind is unknown from P alone, but a fund named 퇴직연금 / 연금저축 tells it.
        if "퇴직연금" in product_name:
            return "퇴직연금", "P 부호 + 펀드 이름의 '퇴직연금'"
        if "연금저축" in product_name:
            return "연금저축", "P 부호 + 펀드 이름의 '연금저축'"
        return "연금(종류 미확인)", "P 부호 (연금 클래스. 연금저축인지 퇴직연금인지는 운용사 · 펀드마다 다름)"
    # 3. Fund name / product_type.
    if "퇴직연금" in product_name:
        return "퇴직연금", "펀드 이름의 '퇴직연금'"
    if "연금저축" in product_name:
        return "연금저축", "펀드 이름의 '연금저축'"
    if product_type == "퇴직연금":
        return "연금(종류 미확인)", "원본 상품 유형 '퇴직연금' (이 유형에는 연금저축 클래스도 있음)"
    return "일반", "연금 표시 없음"


def _core(cls):
    """Split a class name into its Korean words and its letter code.
    "Cpe(퇴직연금)" -> ("퇴직연금", "Cpe"); "수수료미징구-오프라인-퇴직연금(C)" -> ("수수료미징구-오프라인-퇴직연금", "C")."""
    c = (cls or "").replace(" ", "")
    words = "".join(re.findall(r"[가-힣]+", c))
    code = re.sub(r"\((?:퇴직연금|연금저축|개인연금|연금|퇴직)[^)]*\)", "", c)
    m = re.fullmatch(r"[가-힣\-]+\((.+)\)", code)
    if m:
        code = m.group(1)
    code = re.sub(r"[가-힣]", "", code).strip("-")
    return words, code


def channel(cls):
    if cls in NOT_CLASS or not (cls or "").strip():
        return "해당 없음 (클래스 아님)"
    w, c = _core(cls)
    if re.search(r"사전지정|디폴트", w) or c in ("O", "0", "○"):
        return "디폴트옵션"
    if "온라인슈퍼" in w:
        return "온라인슈퍼"
    if "온라인" in w:
        return "온라인"
    if "오프라인" in w:
        return "오프라인"
    if "변액" in w:
        return "변액보험 · 고액 (운용사마다 다름)"
    if re.match(r"[Jj]", c):
        return "운용사 직판"
    if re.match(r"S", c):
        return "온라인슈퍼"
    if re.search(r"(?:^|-)?[Cc]?-?s$", c) and c.lower().endswith("s"):
        return "운용사 · 지정 투자자"
    if c in ("e", "E") or re.search(r"e|E", c[1:] if len(c) > 1 else ""):
        return "온라인"
    if re.search(r"[Ww]", c):
        return "랩 · 특정금전신탁"
    if re.search(r"[Ff]$|^F|CF|[Ii]$|C-a$", c):
        return "기관 · 고액"
    if re.search(r"G|g$", c):
        return "무권유 저비용"
    if re.search(r"^U|-u", c):
        return "엄브렐러 (갈아타기)"
    if re.search(r"[Vv]$", c):
        return "변액보험 · 고액 (운용사마다 다름)"
    if re.fullmatch(r"(?:Class)?[ABCabc]\d?|C-?\d|[A-Ca-c]-?[Pp]\d?|C-?R[Pp]?|C-?r|R[Pp]?|CP\d?|Cp\d?|P\d?|C-CP|PRS|퇴직", c) or w in ("퇴직연금", "연금저축") and not c:
        return "오프라인"
    return "미확인"


def fee(cls):
    if cls in NOT_CLASS or not (cls or "").strip():
        return "해당 없음 (클래스 아님)"
    w, c = _core(cls)
    if "보수체감" in w or "보체감" in w:
        return "보수체감 (C1~C5)"
    if "선취" in w:
        return "선취 (A)"
    if "후취" in w:
        return "후취 (B · S)"
    if "미징구" in w:
        return "미징구 (C)"
    if re.fullmatch(r"C[2-5]e?", c) or re.fullmatch(r"C1", c):
        return "보수체감 (C1~C5)"
    if re.match(r"[Aa]|UA", c):
        return "선취 (A)"
    if re.match(r"B", c):
        return "후취 (B · S)"
    if re.match(r"S-|S[Pp]", c):
        return "미징구 (C)"
    if c == "S" or re.fullmatch(r"Se?", c):
        return "후취 (B · S)"
    if re.match(r"[Cc]", c) or c == "ClassC":
        return "미징구 (C)"
    return "미확인"
