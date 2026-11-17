# finmatch 전처리 데이터 (2026-11-17 최종 수정판)

팀원 수정 요청과 최종 수정 요청 (11-17) 에 맞춰 만든 데이터다. 모든 모델이 같은 검색 문서 목록 (`data/processed/corpus.jsonl`) 과 같은 질문 목록으로 검색하고, 정답은 `unit_id` 로 비교한다.

## 1. 파일과 실험 역할

| 파일 (`data/processed/`) | 한 줄이 뜻하는 것 | 줄 수 | 원천 |
|---|---|---|---|
| `products.jsonl` | 클래스별 원천 상품 1개 | 7,963 | AI Hub 원천 상품 8,000개 − 지운 기록 34개 − 뺀 상품 3개 |
| `corpus.jsonl` | 검색 단위 1개 (검색할 문서) | 3,867 | 상품을 묶은 것 |
| `validation.jsonl` | 설정을 고를 때 쓰는 질문 1개 | 6,400 | AI Hub **Training** 폴더 라벨 |
| `test.jsonl` | 설정을 정한 뒤 최종 평가에만 쓰는 질문 1개 | 801 | AI Hub **Validation** 폴더 라벨 |

- AI Hub 폴더를 다시 나누지 않았다. Training 폴더를 validation 역할로, Validation 폴더를 test 역할로 지정했다. test 는 모델 · 설정 선택에 쓰지 않는다.
- Training · Validation 폴더의 원천 상품 8,000개는 파일 내용까지 같다. 그래서 상품은 한 번만 두고 `source_locations` 에 두 폴더를 함께 적었다.
- 평가는 `evaluation/evaluation.py` 로 한다. 질문마다 Recall@3 · NDCG@3 를 구한 뒤 평균한다. 검색 결과는 한 줄에 `{"question_id": …, "retrieved_unit_ids": [순위대로]}` 로 낸다. `mapping_status` 가 `needs_review` 인 질문 7개는 기본으로 채점에서 빠진다.

## 2. 가입 구분

`account_type` 은 네 값이다. 근거는 상품마다 `account_type_basis` 에 있다.

| 값 | 뜻 | 상품 | 단위 |
|---|---|---|---|
| `general` | 일반 | 4,952 | 1,969 |
| `pension` | 연금 (연금저축 · 퇴직연금을 합침) | 2,091 | 991 |
| `restricted` | 가입 제한 (기관 · 고액, 랩, 장기주택마련 등) | 321 | 308 |
| `unknown` | 미확인 (근거가 펀드 이름이나 원천 폴더뿐) | 599 | 599 |

- 연금저축과 퇴직연금을 나누지 않았다. 질문 7,201개 중 연금저축을 말한 질문이 10개뿐이고, 라벨도 두 클래스를 가리지 않고 붙였다. `pension` 은 "연금 클래스" 라는 뜻이고, 모든 연금계좌에서 가입할 수 있다는 뜻이 아니다. 구체적인 클래스 이름과 근거는 상품에 남아 있다. 두 클래스가 모두 정답이던 질문 8개는 정답 단위가 하나로 합쳐졌다.
- 판매 경로를 몰랐던 클래스 32개는 운용사 투자설명서를 찾아 정했다 (`review/투자설명서_조사/클래스_뜻_조사결과.csv`).

## 3. 검색 단위로 묶은 기준

같은 펀드이고 가입 구분이 같으며, **검색 문서에 들어가는 값이 모두 같은** 상품을 한 단위로 묶었다. 그 값은 투자유형 · 위험등급 · 원금보장 · 만기 · 세제혜택 · 납입 방식 · 유동성 · 검색용 투자전략이다.

- 투자전략은 사실상 같으면 같은 글로 본다. 엑셀 코드 · 위험등급 문장을 지운 뒤 비교해서, 다른 곳이 숫자가 아닌 5글자 이하이거나 한쪽에만 있는 분류 줄 · 위험등급 안내뿐이면 합쳤다.
- `restricted` 는 가입 제한 종류 (`restriction_kind`) 까지 같아야 묶는다. `unknown` 상품은 하나씩 따로 둔다.
- 값이 달라 나뉜 단위는 이유를 `records/단위_분리_이유.csv` 에 적었다 (같은 펀드 · 가입 구분인데 단위가 둘 이상인 409묶음). 가장 많은 이유는 unknown 하나씩 189, 투자전략 56, 세제혜택 41, 가입 제한 종류 25 이다.
- 클래스마다 다른 수익률 (`income_rate`) 은 `products.jsonl` 에만 있다. 단위의 대표값으로 고르거나 평균 내지 않았다.

## 4. 투자전략

`strategy_status` 는 네 값이다. `available` 과 `generated` 만 `strategy` 와 `search_text` 에 들어간다. LLM 에 원본 상세정보를 넘길 때는 원천 글 대신 검색 단위의 `strategy` 를 쓴다 (원천 글 중 35개 상품은 다른 상품의 글이었다).

| 값 | 뜻 | 단위 | 상품 |
|---|---|---|---|
| `available` | 이 상품의 글로 쓸 수 있음 (원천 글 · 같은 펀드 다른 클래스 글 · 투자설명서 글) | 3,865 | 7,961 |
| `generated` | 원문을 끝까지 못 찾아 Claude 가 쓴 설명 | 2 | 2 |
| `needs_review` | 이 상품 글인지 확인하지 못함 | 0 | 0 |
| `missing` | 쓸 글이 없음 | 0 | 0 |

- 자동 검사 (글의 위험등급 문장이 상품과 다름 · 같은 펀드 다른 클래스와 글이 크게 다름 · 이전 판의 needs_review) 에 걸린 상품 361개를 에이전트가 하나씩 읽고 판정했다 (`review/투자전략_판정.csv`). 252개는 맞는 글이었고, 74개는 같은 펀드 다른 클래스의 맞는 글로 복구했다.
- 남은 35개 상품 (14개 펀드) 은 원천 글이 다른 상품의 글이거나 이 상품 것인지 확인할 수 없었다. 13개 펀드는 투자설명서 · 운용사 페이지 · 펀드 정보 사이트에서 투자목적 · 투자전략 글을 찾아 넣었고, 1개 펀드 (한화 100세시대 퇴직연금 글로벌리츠 2호) 는 같은 이름 펀드의 운용보고서를 참고해 Claude 가 설명을 썼다 (`review/투자전략_외부보완.csv`). 출처는 상품마다 `strategy_flag` 에 있다.
- 찾은 글 가운데 펀드닥터에서 가져온 5개 펀드의 글은 요약 도구를 거쳐 인용해서, 원문과 글자 대조를 하지 않았다.
- 자동 검사에 걸리지 않은 상품은 사람이 읽지 않았다. 그래서 `available` 은 "검사를 통과함" 이라는 뜻이다.
- `strategy` 는 원천 글 앞머리의 위험등급 분류 · 예금자보호 안내 문장을 뺀 글이다. "투자에 신중을 기하여" 같은 유의 문장은 1,275개 단위에 남아 있다.
- 원천 글 (`products.jsonl` 의 `source.investment_strategy`) 은 고치지 않았다.

## 5. 검색 문서 (`search_text`)

코드가 아래 꼴로 만든다. 값이 없는 칸은 줄을 넣지 않는다. 원천 값을 그대로 쓰고 새 말을 지어내지 않는다.

```
상품명: ○○. 가입 구분: 연금. 투자유형: 주식형. 위험등급: 2등급(높은위험). 원금보장 여부: 원금 비보장형. 만기: 만기 없음. 세제혜택: 유. 납입 방식: 추가납입. 유동성: 중도해지 가능. 투자전략: (available 일 때만)
```

- 가입 제한은 종류까지 적는다 (예: "가입 구분: 가입 제한 (랩 · 특정금전신탁)").
- 투자유형은 원천 유형이 퇴직연금인 상품 2,204개에서 비워 두었다 ("퇴직연금" 은 투자유형이 아니다).
- 손실 칸은 넣지 않았다. 원천 "100%" 3개는 "최대 100%" 로 합쳤고, 남은 상품이 모두 "최대 100%" 라 구별에 쓸 수 없다.
- 완전히 같은 문서를 쓰는 단위 묶음은 162개이고, 모두 unknown 상품이다.

## 6. 원천 대비 처리 건수

| 처리 | 건수 | 기록 |
|---|---|---|
| 공백만 다른 같은 이름 상품 쌍에서 옛 기록 지움 | 34 | `review/시점_조사/최종_남길기록.csv` |
| 손실 값이 다른 상품 뺌 (0.15% 2개 · 97.50% 1개) | 3 | `review/뺀_상품.csv` |
| 이름이 깨진 상품의 펀드 이름 · 클래스를 직접 정함 | 105 | `review/클래스_직접확인.csv` |
| 투자전략: 같은 펀드 다른 클래스 글로 복구 · 투자설명서 글로 채움 · Claude 가 씀 | 74 · 33 · 2 | `records/투자전략_처리_내역.csv` |
| 지운 기록을 가리키던 정답을 남긴 기록으로 이음 | 68 | `records/정답_이름_보정_기록.csv` |
| 정답 상품 이름의 원천 오타를 고쳐 이음 | 34 | 같은 파일 |
| 뺀 상품이 정답이라 잇지 못한 질문 (`needs_review`) | 7 | 같은 파일 |
| validation · test 사이의 같은 질문 | 0 | |

질문당 정답 단위 수는 validation 1개 85 · 2개 4,730 · 3개 1,585, test 1개 5 · 2개 36 · 3개 760 이다.

## 7. 칸 설명

**products.jsonl**: `product_id` · `fund_id` · `unit_id`, `fund_name` · `share_class` · `class_status`, `account_type` · `account_type_basis`, `investment_type` (모르면 null), `strategy_status` · `strategy_flag`, `source_locations`, `source` (원천 11개 칸을 원래 값 그대로).

**corpus.jsonl**: `unit_id` · `fund_id` · `fund_name` · `account_type` · `restriction_kind`, `source_product_ids` · `class_list` · `n_products`, `product_type` · `investment_type`, `risk_grade` (숫자 1~6), `protected_type` · `maturity_type` · `tax_type` · `payment_type` · `loss_rate` · `liquidity_conditions`, `strategy` · `strategy_status` · `strategy_flag`, `search_text`.

**validation.jsonl · test.jsonl**: `question_id` (`TR-` · `VA-` + cot_id), `cot_id`, `source_split` · `source_file`, `evaluation_role`, `query_type`, `question_raw` (원문) · `question` (공백 · 줄바꿈 정리), `reference_product_names` · `reference_product_ids` · `reference_unit_ids`, `unresolved_product_names`, `mapping_status`. CoT · answer 는 넣지 않았다.

## 8. 코드와 기록

- 순서 (저장소 맨 위에서): `python preprocessing/build.py` → `build_labels.py` → `write_jsonl.py` → `check.py` → `make_records.py <이전 판 products.jsonl>`. pandas 가 필요하고, AI Hub 원천 (Training · Validation 폴더) 을 `data/raw/` 에 둔다.
- `review/` 의 표는 사람이 정한 결정이라 코드가 읽는다.
- `records/`: `점검_결과.md` (최종 점검), `단위ID_대응표.csv` (11-17 첫 판 ↔ 최종 판 ID), `단위_분리_이유.csv`, `투자전략_처리_내역.csv`, `정답_이름_보정_기록.csv`.
- 상품 3개를 빼면서 product_id 와 unit_id 가 모두 다시 매겨졌다. 첫 판 ID 로 해 둔 것이 있으면 `단위ID_대응표.csv` 로 옮긴다.
- 주의: test 질문 801개 중 688개는 정답 단위가 모두 validation 정답에도 나온다. validation 질문과 정답 쌍으로 학습하거나 검색 글에 섞으면 test 점수가 실제보다 높게 나올 수 있다.
