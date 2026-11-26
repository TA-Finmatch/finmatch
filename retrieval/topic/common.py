
import json
import unicodedata
from pathlib import Path

import numpy as np

from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer
)
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. Common Settings
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "v3"
OUTPUT_DIR = ROOT / "results" / "topic"

RANDOM_STATE = 42

MIN_DF = 2
MAX_DF = 1.0

TOP_K = 3
DIAGNOSTIC_K = [3, 5, 10, 20, 50, 100]

BATCH_SIZE = 256


# ============================================================
# 2. Data Loading
# ============================================================

def load_jsonl(path):
    """JSONL 파일 불러오기"""

    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def load_corpus():
    """검색 대상 상품 코퍼스 불러오기"""

    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")

    # 유사도 동점 발생 시 일관된 순위 반환
    corpus = sorted(
        corpus,
        key=lambda row: row["unit_id"]
    )

    unit_ids = [
        row["unit_id"] for row in corpus
    ]

    texts = [
        row["search_text"] for row in corpus
    ]

    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Duplicate unit_id detected.")

    return corpus, unit_ids, texts


def load_questions(split="validation"):
    """고객 질문 불러오기"""

    if split not in ["validation", "test"]:
        raise ValueError(
            "split must be validation or test."
        )

    questions = load_jsonl(
        DATA_DIR / f"{split}.jsonl"
    )

    question_ids = [
        row["question_id"] for row in questions
    ]

    texts = [
        row["question"] for row in questions
    ]

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question_id detected.")

    return questions, question_ids, texts


# ============================================================
# 3. Korean Tokenization
# ============================================================

kiwi = Kiwi()

PUNCTUATION_TAGS = {
    "SF", "SP", "SS", "SSO", "SSC",
    "SE", "SO", "SW", "SB"
}

# 금융 데이터에서 의미가 있을 수 있는 기호
KEEP_SYMBOLS = {"%", "+", "-", "~"}


def normalize_text(text):
    """문자 표현 정규화"""

    if text is None:
        return ""

    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()

    return " ".join(text.split())


def tokenize(text):
    """Kiwi 기반 한국어 형태소 분석"""

    text = normalize_text(text)

    tokens = []

    for token in kiwi.tokenize(text):

        if (
            token.tag in PUNCTUATION_TAGS
            and token.form not in KEEP_SYMBOLS
        ):
            continue

        form = token.form.lower().strip()

        if form:
            tokens.append(form)

    return tokens


def preprocess_texts(texts, description=""):
    """텍스트 목록을 토큰화된 문자열로 변환"""

    processed = []
    cache = {}

    total = len(texts)

    for index, text in enumerate(texts, start=1):

        if text not in cache:
            cache[text] = " ".join(tokenize(text))

        processed.append(cache[text])

        if index % 500 == 0 or index == total:
            print(f"{description}: {index:,}/{total:,}")

    return processed


# ============================================================
# 4. Vectorization
# ============================================================

def create_tfidf_vectorizer(
    min_df=MIN_DF,
    max_df=MAX_DF
):
    """LSA 및 NMF에 사용할 TF-IDF Vectorizer"""

    return TfidfVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1),
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=False,
        use_idf=True,
        smooth_idf=True,
        norm="l2"
    )


def create_count_vectorizer(
    min_df=MIN_DF,
    max_df=MAX_DF
):
    """LDA에 사용할 Count Vectorizer"""

    return CountVectorizer(
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 1),
        min_df=min_df,
        max_df=max_df
    )


# ============================================================
# 5. Similarity Search
# ============================================================

def retrieve_top_k(
    query_vectors,
    corpus_vectors,
    question_ids,
    unit_ids,
    k=TOP_K,
    batch_size=BATCH_SIZE,
    valid_query_mask=None
):
    """코사인 유사도 기반 Top-K 검색"""

    if k < 1:
        raise ValueError("k must be at least 1.")

    if query_vectors.shape[0] != len(question_ids):
        raise ValueError("Question size mismatch.")

    if corpus_vectors.shape[0] != len(unit_ids):
        raise ValueError("Corpus size mismatch.")

    if len(unit_ids) != len(set(unit_ids)):
        raise ValueError("Duplicate corpus IDs.")

    k = min(k, len(unit_ids))
    results = []

    for start in range(0, len(question_ids), batch_size):

        end = min(
            start + batch_size,
            len(question_ids)
        )

        similarities = cosine_similarity(
            query_vectors[start:end],
            corpus_vectors
        )

        for index, scores in enumerate(similarities):

            question_index = start + index

            # 어휘에 없는 질문 등 유효하지 않은 벡터 처리
            is_valid = (
                valid_query_mask is None
                or valid_query_mask[question_index]
            )

            if not is_valid:
                results.append({
                    "question_id": question_ids[question_index],
                    "retrieved_unit_ids": []
                })
                continue

            # 유사도 내림차순 정렬
            # stable sort를 사용하여 동점 처리 일관성 유지
            top_indices = np.argsort(
                -scores,
                kind="stable"
            )[:k]

            results.append({
                "question_id": question_ids[question_index],
                "retrieved_unit_ids": [
                    unit_ids[i] for i in top_indices
                ]
            })

    return results


# ============================================================
# 6. Result Saving
# ============================================================

def save_jsonl(rows, path):
    """검색 결과를 공통 평가 형식으로 저장"""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:

        for row in rows:
            f.write(
                json.dumps(row, ensure_ascii=False)
                + "\n"
            )

    print(f"Saved: {path}")


def save_json(data, path):
    """실험 설정 및 요약 결과 저장"""

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )

    print(f"Saved: {path}")


# ============================================================
# 7. Tokenization Check
# ============================================================

def main():

    _, unit_ids, corpus_texts = load_corpus()

    _, question_ids, question_texts = load_questions(
        "validation"
    )

    print("\n" + "=" * 60)
    print("Corpus Tokenization Sample")
    print("=" * 60)

    print("\nUnit ID:")
    print(unit_ids[0])

    print("\nOriginal text:")
    print(corpus_texts[0])

    corpus_tokens = tokenize(corpus_texts[0])

    print("\nTokens:")
    print(corpus_tokens)

    print("\nToken count:")
    print(len(corpus_tokens))

    print("\n" + "=" * 60)
    print("Question Tokenization Sample")
    print("=" * 60)

    print("\nQuestion ID:")
    print(question_ids[0])

    print("\nOriginal question:")
    print(question_texts[0])

    question_tokens = tokenize(question_texts[0])

    print("\nTokens:")
    print(question_tokens)

    print("\nToken count:")
    print(len(question_tokens))

    print("\n" + "=" * 60)
    print("Financial Expression Check")
    print("=" * 60)

    test_texts = [
        "원금 비보장형",
        "원금 보장이 필요하지 않습니다.",
        "중도 환매가 가능한 상품",
        "2등급 높은위험",
        "퇴직연금 및 해외 주식형 펀드",
        "연 5% 이상의 수익률",
        "국내 주식에 투자한 경험이 있습니다.",
        "앞으로 국내 주식에 투자하고 싶습니다."
    ]

    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Tokens: {tokenize(text)}")


if __name__ == "__main__":
    main()
