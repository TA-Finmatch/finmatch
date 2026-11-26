
import json
from pathlib import Path
from collections import Counter
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed" / "v3"
OUTPUT_DIR = ROOT / "results" / "topic"


def load_jsonl(path):
    """JSONL 파일 불러오기"""
    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def length_statistics(texts):
    """텍스트 길이 통계"""
    lengths = [len(text) for text in texts]

    if not lengths:
        return {}

    return {
        "min": min(lengths),
        "max": max(lengths),
        "mean": round(mean(lengths), 2),
        "median": median(lengths)
    }


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():

    # 1. 데이터 불러오기
    corpus = load_jsonl(DATA_DIR / "corpus.jsonl")
    products = load_jsonl(DATA_DIR / "products.jsonl")
    validation = load_jsonl(DATA_DIR / "validation.jsonl")

    print_section("1. Dataset Overview")

    print(f"Corpus: {len(corpus):,}")
    print(f"Products: {len(products):,}")
    print(f"Validation: {len(validation):,}")

    # 2. 데이터 필드 확인
    print_section("2. Dataset Schema")

    corpus_fields = sorted({
        key for row in corpus for key in row
    })

    product_fields = sorted({
        key for row in products for key in row
    })

    question_fields = sorted({
        key for row in validation for key in row
    })

    print("Corpus fields:", corpus_fields)
    print("Product fields:", product_fields)
    print("Question fields:", question_fields)

    # 3. 상품 문서 분석
    print_section("3. Corpus Analysis")

    corpus_texts = [
        row["search_text"] for row in corpus
    ]

    corpus_length = length_statistics(corpus_texts)

    empty_documents = sum(
        not text.strip() for text in corpus_texts
    )

    text_counts = Counter(corpus_texts)

    identical_groups = sum(
        count > 1 for count in text_counts.values()
    )

    identical_documents = sum(
        count for count in text_counts.values()
        if count > 1
    )

    print("Text length:", corpus_length)
    print("Empty documents:", empty_documents)
    print("Unique search texts:", len(text_counts))
    print("Identical text groups:", identical_groups)
    print("Documents in identical groups:", identical_documents)

    # 4. 고객 질문 분석
    print_section("4. Question Analysis")

    questions = [
        row["question"] for row in validation
    ]

    question_length = length_statistics(questions)

    empty_questions = sum(
        not text.strip() for text in questions
    )

    query_types = Counter(
        row.get("query_type", "unknown")
        for row in validation
    )

    print("Question length:", question_length)
    print("Empty questions:", empty_questions)

    print("\nQuery type distribution:")

    for query_type, count in query_types.most_common():
        print(f"  {query_type}: {count:,}")

    # 5. 정답 상품 개수 분석
    print_section("5. Reference Analysis")

    reference_counts = Counter(
        len(set(row.get("reference_unit_ids") or []))
        for row in validation
    )

    mapping_status = Counter(
        row.get("mapping_status", "unknown")
        for row in validation
    )

    print("Reference count distribution:")

    for count, frequency in sorted(reference_counts.items()):
        print(f"  {count} references: {frequency:,}")

    print("\nMapping status:")

    for status, count in mapping_status.items():
        print(f"  {status}: {count:,}")

    # 6. 데이터 무결성 검증
    print_section("6. Data Integrity")

    corpus_ids = [
        row["unit_id"] for row in corpus
    ]

    corpus_id_set = set(corpus_ids)

    question_ids = [
        row["question_id"] for row in validation
    ]

    reference_ids = {
        unit_id
        for row in validation
        for unit_id in (row.get("reference_unit_ids") or [])
    }

    product_unit_ids = {
        row["unit_id"] for row in products
    }

    unknown_reference_ids = sorted(
        reference_ids - corpus_id_set
    )

    unknown_product_ids = sorted(
        product_unit_ids - corpus_id_set
    )

    duplicate_corpus_ids = (
        len(corpus_ids) - len(corpus_id_set)
    )

    duplicate_question_ids = (
        len(question_ids) - len(set(question_ids))
    )

    print("Duplicate corpus IDs:", duplicate_corpus_ids)
    print("Duplicate question IDs:", duplicate_question_ids)
    print("Unknown reference IDs:", len(unknown_reference_ids))
    print("Unknown product unit IDs:", len(unknown_product_ids))

    # 7. 평가 대상 및 이론적 최대 Recall 계산
    print_section("7. Evaluation Analysis")

    scored_questions = [
        row for row in validation
        if row.get("mapping_status") != "needs_review"
        and row.get("reference_unit_ids")
    ]

    def recall_ceiling(row, k):
        n_reference = len(set(row["reference_unit_ids"]))
        return min(k, n_reference) / n_reference

    max_recall = {
        f"Recall@{k}": round(
            mean([
                recall_ceiling(row, k)
                for row in scored_questions
            ]),
            6
        )
        for k in [1, 3, 5, 10]
    } if scored_questions else {}

    print("Total questions:", len(validation))
    print("Scored questions:", len(scored_questions))
    print("Excluded questions:", len(validation) - len(scored_questions))
    print("Theoretical maximum recall:", max_recall)

    # 8. 데이터 샘플 확인
    print_section("8. Data Samples")

    for index, row in enumerate(corpus[:2], start=1):
        print(f"\n[Corpus Sample {index}]")
        print("Unit ID:", row["unit_id"])
        print("Search text:", row["search_text"])

    for index, row in enumerate(validation[:2], start=1):
        print(f"\n[Question Sample {index}]")
        print("Question ID:", row["question_id"])
        print("Question:", row["question"])
        print("Query type:", row.get("query_type"))
        print("References:", row.get("reference_unit_ids"))

    # 9. 분석 결과 저장
    report = {
        "dataset_counts": {
            "corpus": len(corpus),
            "products": len(products),
            "validation": len(validation)
        },
        "schema": {
            "corpus": corpus_fields,
            "products": product_fields,
            "validation": question_fields
        },
        "corpus_analysis": {
            "text_length": corpus_length,
            "empty_documents": empty_documents,
            "unique_texts": len(text_counts),
            "identical_text_groups": identical_groups,
            "identical_documents": identical_documents
        },
        "question_analysis": {
            "text_length": question_length,
            "empty_questions": empty_questions,
            "query_types": dict(query_types)
        },
        "reference_analysis": {
            "reference_counts": dict(reference_counts),
            "mapping_status": dict(mapping_status)
        },
        "data_integrity": {
            "duplicate_corpus_ids": duplicate_corpus_ids,
            "duplicate_question_ids": duplicate_question_ids,
            "unknown_reference_ids": unknown_reference_ids,
            "unknown_product_unit_ids": unknown_product_ids
        },
        "evaluation": {
            "scored_questions": len(scored_questions),
            "excluded_questions": (
                len(validation) - len(scored_questions)
            ),
            "maximum_recall": max_recall
        },
        "samples": {
            "corpus": corpus[:2],
            "questions": validation[:2]
        }
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "data_audit.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=4
        )

    print_section("9. Completed")
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
