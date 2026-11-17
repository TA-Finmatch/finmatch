# finmatch 전처리 데이터 (2026-11-17 판)

팀원 수정 요청에 맞춰 다시 만든 데이터다. 모든 모델이 같은 검색 문서 목록 (`corpus.jsonl`) 과 같은 질문 목록으로 검색하고, 정답은 `unit_id` 로 비교한다.

## 1. 파일과 실험 역할

| 파일 | 한 줄이 뜻하는 것 | 줄 수 | 원천 |
|---|---|---|---|
| `products.jsonl` | 클래스별 원천 상품 1개 | 7,966 | AI Hub 원천 상품 8,000개 − 지운 기록 34개 |
| `corpus.jsonl` | 검색 단위 1개 (검색할 문서) | 3,871 | 상품을 묶은 것 |
| `validation.jsonl` | 설정을 고를 때 쓰는 질문 1개 | 6,400 | AI Hub **Training** 폴더 라벨 |
| `test.jsonl` | 설정을 정한 뒤 최종 평가에만 쓰는 질문 1개 | 801 | AI Hub **Validation** 폴더 라벨 |

- 팀원 요청대로 AI Hub 폴더를 다시 나누지 않았다. Training 폴더를 validation 역할로, Validation 폴더를 test 역할로 지정했다. test 는 모델 · 설정 선택에 쓰지 않는다.
- Training · Validation 폴더의 원천 상품 8,000개는 파일 내용까지 같다. 그래서 상품은 한 번만 두고 `source_locations` 에 두 폴더를 함께 적었다.
- 평가는 `evaluation/evaluation.py` 로 한다. 질문마다 Recall@3 · NDCG@3 를 구한 뒤 평균한다. 검색 결과는 한 줄에 `{"question_id": …, "retrieved_unit_ids": [순위대로]}` 로 낸다.

## 2. 상품을 검색 단위로 묶은 기준

같은 펀드이고 가입 구분 (`account_type`) 이 같은 상품을 한 단위로 묶었다. 다만 아래 경우는 묶지 않았다.

- 원천 유형 · 위험등급 · 원금보장 · 만기 · 세제혜택 · 납입 · 손실 · 환매 조건 중 하나라도 다르면 따로 둔다.
- 투자전략 글이 다르면 따로 둔다. 다만 사실상 같은 글은 같은 글로 본다. 엑셀 코드 · 위험등급 문장을 지운 뒤 비교해서, 다른 곳이 숫자가 아닌 5글자 이하이거나 한쪽에만 있는 분류 줄 · 위험등급 안내뿐이면 합쳤다. 이때 단위의 글은 가장 긴 쪽을 쓴다.
- 가입 구분이 `unknown` 인 상품은 하나씩 따로 둔다 (불확실한 상품끼리 임의로 묶지 않는다).

그래서 한 단위 안의 상품은 위 값이 모두 같다. 클래스마다 다른 수익률 (`income_rate`) 은 `products.jsonl` 에만 있다.

가입 구분은 클래스 이름 · 클래스 부호 · 판매 경로로 정했다. 근거는 상품마다 `account_type_basis` 에 있다.

| 값 | 뜻 | 상품 | 단위 |
|---|---|---|---|
| `general` | 일반 계좌 클래스 | 4,955 | 1,995 |
| `pension` | 연금저축 · 퇴직연금 클래스 (하나로 합침) | 2,091 | 994 |
| `restricted` | 기관 · 고액 · 랩 · 특수 가입 자격 클래스 | 321 | 283 |
| `unknown` | 연금일 수 있으나 근거가 펀드 이름이나 원천 폴더뿐 | 599 | 599 |

- 팀원 요청은 연금저축과 퇴직연금을 나누는 것이었지만, 하나로 합쳤다. 질문 7,201개 중 연금저축을 말한 질문은 10개뿐이다. 또 정답 라벨이 연금저축 클래스와 퇴직연금 클래스를 가리지 않고 붙어 있어서, 나눠도 어느 쪽이 정답인지 가릴 근거가 없다. 그래서 연금저축 클래스와 퇴직연금 클래스가 같은 단위에 들어간 경우가 있고, 두 클래스가 모두 정답이던 질문 8개는 정답 단위가 하나로 합쳐졌다.
- 판매 경로를 몰랐던 클래스 32개는 운용사 투자설명서를 찾아 정했다. 그중 투자설명서를 찾지 못한 3개는 판매사 상품 페이지로 정했고, 수정 · 미확인 목록에 적었다. 장기주택마련 · 사원복지연기금 · 계열사 자금 · 시드 클래스는 가입 자격이 정해져 있어 `restricted` 로 넣었다.

## 3. 원천 대비 처리 건수

| 처리 | 건수 | 비고 |
|---|---|---|
| 공백만 다른 같은 이름 상품 쌍에서 옛 기록 지움 | 34 | 간이투자설명서 · 기준가로 시점을 가려 최근 기록을 남김 |
| 이름이 깨진 상품의 펀드 이름 · 클래스를 직접 정함 | 105 | `class_status` = 손으로 정함 |
| 클래스 표기가 없는 상품 | 474 | `class_status` = 클래스 없음 (예: 주가연계증권 · 신탁) |
| 원천 유형이 퇴직연금이라 투자 유형을 모르는 상품 | 2,204 | `investment_type` = null |
| 투자전략 글이 이상한 단위 (남의 펀드 글 · 경고문뿐) | 57 | 원천 글 그대로 둠. 표시 칸은 두지 않음 |
| 지운 기록을 가리키던 정답을 남긴 기록으로 이음 | 68 | |
| 정답 상품 이름의 원천 오타를 고쳐 이음 | 34 | |
| 정답을 잇지 못한 질문 · 평가에서 뺀 질문 | 0 | |
| validation · test 사이에 글자 유사도 0.8 이상인 질문 짝 | 0 | 완전히 같은 질문 포함. 뜻이 비슷한 질문은 보지 않음 |

## 4. 칸 설명

**products.jsonl**: `product_id` · `fund_id` · `unit_id`, `fund_name` · `share_class` · `class_status`, `account_type` · `account_type_basis`, `investment_type` (주식형 · 채권형 · 재간접 · 단기금융 · 파생형, 모르면 null), `source_locations`, `source` (원천 11개 칸을 원래 값 그대로).

**corpus.jsonl**: `unit_id` · `fund_id` · `fund_name` · `account_type`, `source_product_ids` · `class_list` · `n_products`, `product_type` (원천 유형 6개 그대로) · `investment_type`, `risk_grade` (숫자 1~6), `protected_type` · `maturity_type` · `tax_type` · `payment_type` · `loss_rate` · `liquidity_conditions` (원천 문자열), `strategy`, `search_text`. 팀원 요청의 `strategy_status` · `strategy_flag` 는 넣지 않았다.
- `search_text` 는 코드가 만든다. 펀드 이름, 원천 유형, 속성 7개, 투자전략 글을 이어 붙였고 새 말을 지어내지 않았다.
- `strategy` 는 원천 글 앞머리의 위험등급 분류 · 예금자보호 안내 문장을 뺀 글이다. "투자에 신중을 기하여 주시기 바랍니다" 같은 유의 문장은 남아 있다 (글이 정상인 3,814개 단위 중 1,267개). 글이 이상한 57개 단위는 원천 글 그대로다.

**validation.jsonl · test.jsonl**: `question_id` (`TR-` · `VA-` + cot_id), `cot_id`, `source_split` · `source_file`, `evaluation_role`, `query_type`, `question_raw` (원문) · `question` (공백 · 줄바꿈 정리), `reference_product_names` (원천 정답 이름) · `reference_product_ids` · `reference_unit_ids`, `unresolved_product_names`, `mapping_status`.
- 정답 상품 여러 개가 같은 단위로 묶이면 `reference_unit_ids` 에는 한 번만 들어간다. 질문당 정답 단위는 1~3개다.
- CoT · answer 는 검색 입력에 쓰지 않으므로 넣지 않았다.

## 5. 함께 보는 파일

- 수정 · 미확인 목록: 고친 것, 판단이 들어간 결정, 미확인으로 남은 것 (`data/processed/수정_미확인_목록.md`)
- 11-09 판 ID 와 이번 판 ID 대응표 (`data/processed/옛ID_새ID_*.csv`, 설명은 `옛ID_새ID_설명.md`)
- 정답 상품 이름을 고쳐 이은 기록 (`data/processed/정답_이름_보정_기록.csv`)
- 만드는 코드: `build.py` (상품 · 단위) → `build_labels.py` (정답 잇기) → `write_jsonl.py` (JSONL) → `check.py` (점검, 결과는 `data/processed/점검_결과.md`). `make_id_map.py` 는 ID 대응표를 만든다.
- 다시 돌리려면 pandas 가 필요하고, AI Hub 원천 (Training · Validation 폴더) 을 `data/raw/` 에 둔다. 코드는 저장소 맨 위에서 `python preprocessing/build.py` 처럼 돌린다. 손으로 정한 표 (`preprocessing/review/시점_조사/최종_남길기록.csv` · `preprocessing/review/클래스_직접확인.csv` · `preprocessing/review/투자설명서_조사/클래스_뜻_조사결과.csv`) 도 코드가 읽는다. `make_id_map.py` 는 11-09 판 `_base_1109/product.csv` · `corpus.csv` 를 `preprocessing/_base_1109/` 에 두어야 돈다.
- 주의: test 질문 801개 중 689개는 정답 단위가 모두 validation 질문의 정답에도 나온다 (상품으로 세면 447개). 질문을 학습에 쓰지 않으면 큰 문제는 없다. 하지만 validation 질문과 정답 쌍으로 모델을 학습하거나 검색 글에 섞으면 test 점수가 실제보다 높게 나올 수 있다.
