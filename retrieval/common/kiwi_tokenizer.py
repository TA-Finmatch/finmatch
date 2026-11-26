"""팀 공통 한국어 토큰화 (Kiwi 형태소 분석).

TF-IDF · BM25 · LSA · LDA · NMF · Word2Vec 처럼 낱말을 직접 세거나 낱말마다 벡터를 배우는 방법이
같이 쓰려고 만든 파일이다. 방법마다 토큰화가 다르면 점수 차이가 방법 때문인지 토큰화 때문인지
가를 수 없어서, 표현 방식을 비교하는 우리 실험에서는 토큰화를 한 벌로 고정하려는 것이다.
(2026-11-27 기준 Word2Vec 과 하이브리드 BM25 가 이 파일을 쓴다. 다른 방법은 각 담당자가 옮겨 온다.)

**문장 임베딩 모델 (KURE-v3 · multilingual-e5 · Qwen3-Embedding) 에는 쓰지 않는다.**
그 모델들은 학습 때 정한 자기 토크나이저가 있어서 원문 (`question`, `search_text`) 을 그대로 넣는다.

규칙 (한 벌뿐이다)
------------------
1. 문자 모양 통일: 유니코드 NFKC, 줄바꿈 · 탭 · 연속 공백은 공백 하나로. 빈 값 (None, 파이썬 float 의 NaN) 은 빈 글.
   pandas 의 `pd.NA` 나 numpy float32 의 NaN 은 처리하지 않는다 (넣기 전에 `fillna("")` 를 한다).
2. `search_text` 의 틀 이름표 지우기: "상품명: ", "가입 구분: ", "위험등급: " 처럼 모든 상품 글에 같은 자리에
   들어간 이름표 10개 (`TEMPLATE_LABELS`) 를 지운다. 이름표 뒤의 값 ("2등급(높은위험)") 은 남는다.
   이름표가 남으면 3,930개 글 전부에 같은 낱말이 생겨 LDA · Word2Vec 이 그 낱말에 끌려간다.
3. Kiwi 로 형태소를 나눈다. 금융 용어 (`USER_WORDS`) 는 사용자 사전에 넣어 한 낱말로 둔다.
   예: 기본 사전은 "재간접" 을 접두사 "재" + "간접", "주식형" 을 "주식" + 접미사 "형" 으로 나눈다.
4. 숫자와 단위를 한 낱말로 묶는다: "2등급", "3년", "1개월", "100%", "40대", "100만원".
   기본 품사 고르기만 하면 숫자가 버려져 1등급 상품과 5등급 상품을 구별하지 못한다.
   숫자에 빈칸 없이 바로 붙어 있으면 의존명사 (년 · 개월 · 대 · 원 · 호 · 차 · 명 등) 와 수사 (만 · 천만 · 억) 는
   무엇이든 붙이고, "%" 와 일반명사 "등급" 도 붙인다. 그래서 원문에 붙여 쓴 "제93조등" 은 "93조등" 이 된다.
   숫자와 단위 사이에 빈칸이 하나 있어도 붙인다: "3 년" · "2 등급" · "100만 원" 도 "3년" · "2등급" · "100만원".
   빈칸을 넘어 붙이는 단위는 `SPACED_UNITS` 목록에 있는 것만이다 ("4호 등" 의 "등" 같은 말은 붙이지 않는다).
   알려진 한계: 신용등급 "A2 등급" 도 "a", "2등급" 이 되어 위험등급 낱말과 같아진다. "9, 000만 원" 처럼
   쉼표 뒤에 빈칸이 있으면 "9", "000만원" 으로 갈린다. "8천만원" · "8,000만원" 은 서로 다른 낱말이다.
   질문은 띄어 쓰고 상품 글은 붙여 쓰는 일이 많아서, 붙이지 않으면 같은 말이 다른 낱말로 갈린다.
   명사 (외국어 포함) 바로 뒤에 붙은 접미사 "형" 도 앞 낱말에 붙인다: "채권형펀드" → "채권형", "펀드". "MMF형" → "mmf형".
   명사 바로 앞에 붙은 접두사 (비 · 재 · 고 · 초 · 저 등) 도 뒤 명사에 붙인다: "비보장형", "고배당", "저위험".
   접두사를 버리면 "원금 비보장형" 이 "원금 보장형" 으로 뜻이 뒤집힌다.
5. 내용 낱말만 남긴다: 일반명사 NNG · 고유명사 NNP · 외국어 SL · 한자 SH · 어근 XR, 그리고 4의 숫자 낱말.
   조사 · 어미 · 동사 · 형용사 · 문장부호는 버린다.
6. 영어 글자는 품사와 상관없이 모두 소문자로 바꾼다 ("ETF" → "etf", 고유명사 "NH투자증권" → "nh투자증권").
   띄어쓰기가 다르면 여전히 다른 낱말이다 ("NH 투자증권" → "nh", "투자", "증권").

아주 흔한 낱말을 더 거르는 일은 토큰화가 아니라 각 모델의 설정으로 한다
(scikit-learn 의 `min_df` · `max_df`, gensim 의 `min_count` 등).

쓰는 법
-------
    import sys; sys.path.insert(0, str(ROOT / "retrieval" / "common"))
    import kiwi_tokenizer as kt

    kt.tokenize("채권형펀드를 추천해 주세요")          # ['채권형', '펀드', '추천']
    doc_tokens = kt.tokenize_all(texts, cache_dir=ROOT / "scratch" / "cache" / "tokens")
    TfidfVectorizer(analyzer=kt.tokenize)            # 원문을 넣고 이 함수로 자르게 하기
    TfidfVectorizer(analyzer=lambda toks: toks)      # 이미 자른 목록 (tokenize_all 결과) 을 넣기

    python retrieval/common/kiwi_tokenizer.py --self-test

kiwipiepy 는 0.22.2 로 고정한다 (팀 requirements.txt 와 같다). 판이 다르면 같은 글도 다르게 잘릴 수 있어서
다른 판이면 오류를 낸다.
"""

import hashlib
import math
import pickle
import re
import sys
import unicodedata
from pathlib import Path

KIWI_VERSION = "0.22.2"

KEEP_TAGS = ("NNG", "NNP", "SL", "SH", "XR")

# search_text 의 틀 이름표 (preprocessing/README.md 5절의 형식). 글 맨 앞이나 ". " 바로 뒤에 있을 때만 지운다
TEMPLATE_LABELS = ("상품명", "가입 구분", "투자유형", "위험등급", "원금보장 여부",
                   "만기", "세제혜택", "납입 방식", "유동성", "투자전략")

# 한 낱말로 둘 금융 용어. Kiwi 기본 사전이 둘 이상으로 나누는 말이다.
# Kiwi 사용자 사전에 넣고, 사전이 안 걸린 자리 ("단기금융상품", "퇴직연금제도") 도 붙어 있는 조각을 이어 붙여 한 낱말로 만든다
USER_WORDS = (
    # 투자유형 · 상품 종류
    "재간접", "주식형", "채권형", "파생형", "혼합형", "재간접형", "단기금융", "퇴직연금", "연금저축",
    # 투자성향 (질문에 나온다)
    "공격투자형", "적극투자형", "위험중립형", "안정추구형", "안정형",
)

# 숫자 (SN) 바로 뒤에 붙어 있으면 숫자와 한 낱말로 묶는 것: 의존명사 NNB (년 · 개월 · 대 · 원 · 호 · 차),
# 수사 NR (천만 · 억), "%", 그리고 일반명사 중 UNIT_NOUNS 만. 아무 명사나 붙이면 "200지수", "60%이상" 처럼
# 붙여 쓴 곳과 띄어 쓴 곳이 다른 낱말로 갈린다
UNIT_NOUNS = ("등급",)
# 숫자 (또는 "만" · "억" 같은 수사) 와 빈칸 하나를 사이에 두고 있어도 붙이는 단위. 이 목록에 있는 것만 붙인다.
# "등" (= 따위) · "수" · "내" 같은 의존명사가 숫자에 붙지 않게 하려고 목록으로 좁혔다
SPACED_UNITS = ("년", "개월", "월", "일", "주", "등급", "%", "원", "만", "천", "천만", "억", "조",
                "호", "차", "대", "세", "개", "회", "위", "엔")
_USER_MAXLEN = 4                   # 사용자 사전 낱말을 이어 붙일 때 볼 조각 수

_USER_SET = set(USER_WORDS)
_LABEL_RE = re.compile(r"(?:^|(?<=\. ))(?:" + "|".join(re.escape(x) for x in TEMPLATE_LABELS) + r"): ")
_kiwi = None


def _get_kiwi():
    global _kiwi
    if _kiwi is None:
        import kiwipiepy
        from kiwipiepy import Kiwi
        if kiwipiepy.__version__ != KIWI_VERSION:
            raise RuntimeError(f"kiwipiepy {kiwipiepy.__version__} 이다. 팀 공통 토큰화는 {KIWI_VERSION} 로 고정한다: "
                               f"pip install kiwipiepy=={KIWI_VERSION}")
        _kiwi = Kiwi()
        for w in USER_WORDS:
            _kiwi.add_user_word(w, "NNG", 0)
    return _kiwi


def normalize(text):
    """문자 모양을 통일하고 search_text 의 틀 이름표를 지운다."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = " ".join(text.split())
    return _LABEL_RE.sub("", text)


def _select(tokens):
    """Kiwi 결과에서 숫자+단위를 묶고 내용 낱말만 남긴다."""
    out, i, n = [], 0, len(tokens)
    while i < n:
        t = tokens[i]
        # 금융 용어: 붙어 있는 조각을 이어 붙인 것이 USER_WORDS 에 있으면 한 낱말로 (가장 긴 것 먼저)
        hit = None
        for m in range(min(_USER_MAXLEN, n - i), 1, -1):
            run = tokens[i:i + m]
            if all(run[x].start + run[x].len == run[x + 1].start for x in range(m - 1)) \
                    and all(x.tag in ("NNG", "NNP", "XPN", "XSN") for x in run) \
                    and "".join(x.form for x in run) in _USER_SET:
                hit = m
                break
        if hit:
            out.append("".join(x.form for x in tokens[i:i + hit]))
            i += hit
            continue
        if t.tag == "SN":
            form, end, j = t.form.rstrip(".,"), t.start + t.len, i + 1   # "2." (번호 매기기) → "2"
            last_tag = "SN"
            while j < n:
                nt = tokens[j]
                if not (nt.tag in ("NNB", "NR") or (nt.tag == "SW" and nt.form == "%")
                        or (nt.tag == "NNG" and nt.form in UNIT_NOUNS)):
                    break
                gap = nt.start - end
                # 바로 붙어 있으면 붙인다. 빈칸 하나 (위치가 1 벌어짐. normalize 가 공백을 하나로 줄였다) 뒤라면
                # 바로 앞이 숫자나 수사이고, 단위가 SPACED_UNITS 에 있고, 숫자가 번호 매기기 ("2.") 가 아닐 때만 붙인다
                if not (gap == 0 or (gap == 1 and last_tag in ("SN", "NR") and nt.form in SPACED_UNITS
                                     and not t.form.endswith("."))):
                    break
                form += nt.form; end = nt.start + nt.len; last_tag = nt.tag; j += 1
            if form:
                out.append(form)
            i = j
            continue
        if (t.tag == "XPN" and i + 1 < n and tokens[i + 1].tag in ("NNG", "NNP")
                and t.start + t.len == tokens[i + 1].start):
            out.append(t.form + tokens[i + 1].form)   # 접두사를 뒤 명사에 붙인다: "비보장", "재간접", "고배당"
            i += 2
            continue
        if (t.tag == "XSN" and t.form == "형" and out and i > 0 and tokens[i - 1].tag in ("NNG", "NNP", "SL")
                and tokens[i - 1].start + tokens[i - 1].len == t.start):
            out[-1] += "형"                      # "채권형펀드" 처럼 사용자 사전이 안 걸려도 명사 + 형 을 붙인다
            i += 1
            continue
        if t.tag in KEEP_TAGS:
            out.append(t.form)
        i += 1
    return [w.lower() for w in out]                # 영어 글자는 품사와 상관없이 소문자 (한글은 그대로)


def tokenize(text):
    """글 하나를 공통 규칙으로 잘라 낱말 목록을 돌려준다."""
    return _select(_get_kiwi().tokenize(normalize(text)))


def config_id():
    """규칙의 지문. 캐시 파일 이름과 결과 기록에 쓴다.
    이 파일의 내용 전체와 Kiwi 판으로 만든다. 그래서 규칙 코드를 고치면 (주석만 고쳐도) 지문이 바뀌어 옛 캐시를 읽지 않는다."""
    # 윈도우에서 git 이 줄바꿈을 \r\n 으로 바꿔도 같은 지문이 나오게 줄바꿈을 맞춘 뒤 잰다
    h = hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n"))
    h.update(KIWI_VERSION.encode("utf-8"))
    return h.hexdigest()[:10]


def tokenize_all(texts, cache_dir=None, name="tokens"):
    """글 여러 개를 자른다. cache_dir 를 주면 결과를 저장하고, 글과 규칙이 같으면 다시 읽는다."""
    texts = list(texts)
    path = None
    if cache_dir is not None:
        h = hashlib.sha256()
        for t in texts:
            h.update(str(t).encode("utf-8")); h.update(b"\x00")
        path = Path(cache_dir) / f"{name}_{h.hexdigest()[:10]}_kt{config_id()}.pkl"
        if path.exists():
            return pickle.loads(path.read_bytes())
    kiwi = _get_kiwi()
    result = [_select(toks) for toks in kiwi.tokenize([normalize(t) for t in texts])]
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(result))
    return result


def self_test(verbose=True):
    """우리 데이터에서 나오는 모양으로 규칙이 지켜지는지 본다."""
    cases = [
        ("숫자와 단위를 묶음", "위험등급: 2등급(높은위험).", ["2등급", "위험"]),
        ("만기의 년 · 개월", "만기: 최초설정일로부터 3년 1개월.", ["최초", "설정", "일", "3년", "1개월"]),
        ("퍼센트", "최대 100% 손실", ["최대", "100%", "손실"]),
        ("틀 이름표를 지움 (지어낸 상품 이름)", "상품명: XY 하늘 미국 증권투자신탁[주식-재간접]. 투자유형: 재간접.",
         ["xy", "하늘", "미국", "증권", "투자", "신탁", "주식", "재간접", "재간접"]),
        ("이름표가 아닌 '만기' 는 남김", "만기: 만기 없음.", ["만기"]),
        ("조사 · 어미를 버리고 -형 을 붙임", "채권형펀드를 추천해 주세요", ["채권형", "펀드", "추천"]),
        ("투자성향 낱말", "적극투자형 위험중립형 안정추구형 공격투자형 안정형", ["적극투자형", "위험중립형", "안정추구형", "공격투자형", "안정형"]),
        ("상품 종류", "단기금융 퇴직연금 연금저축 파생형", ["단기금융", "퇴직연금", "연금저축", "파생형"]),
        ("접두사 '비' 를 지킴", "원금보장 여부: 원금 비보장형. 비과세 고배당", ["원금", "비보장형", "비과세", "고배당"]),
        ("사전이 안 걸린 자리도 한 낱말로", "단기금융상품에 투자하는 퇴직연금제도", ["단기금융", "상품", "투자", "퇴직연금", "제도"]),
        ("등급 말고 다른 명사는 숫자에 붙이지 않음", "KOSPI200지수 60%이상 2. 운용", ["kospi", "200", "지수", "60%", "이상", "2", "운용"]),
        ("영어는 소문자", "ETF MMF IRP", ["etf", "mmf", "irp"]),
        ("영어가 붙은 고유명사도 소문자 (Kiwi 사전에 고유명사로 있는 회사 이름)", "LG전자", ["lg전자"]),
        ("띄어 쓴 숫자와 단위도 붙임", "3 년 1 개월, 2 등급, 약 100만 원", ["3년", "1개월", "2등급", "100만원"]),
        ("번호 매기기 '2.' 뒤에는 붙이지 않음", "2. 3 년", ["2", "3년"]),
        ("빈칸 뒤 '등' (따위) 은 붙이지 않음", "제1호 및 제4호 등", ["1호", "4호"]),
        ("영어 약어 뒤 형", "MMF형 상품", ["mmf형", "상품"]),
        ("빈 값", float("nan"), []),
        ("문자 모양 통일 (전각 숫자)", "２등급", ["2등급"]),
    ]
    ok = True
    for label, text, want in cases:
        got = tokenize(text)
        passed = got == want
        ok &= passed
        if verbose:
            print(f"[{'통과' if passed else '실패'}] {label}: {got}" + ("" if passed else f" (기대 {want})"))
    batch = tokenize_all([c[1] for c in cases])
    same = batch == [tokenize(c[1]) for c in cases]
    ok &= same
    if verbose:
        print(f"[{'통과' if same else '실패'}] tokenize_all 이 tokenize 와 같은 결과")
    return ok


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    print(__doc__)
