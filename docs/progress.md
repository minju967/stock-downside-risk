# 진행 기록

마지막 갱신: 2026-09-29

## 현재 위치
- **EDA(Step 0 ~ 6) 완료, v2(KRX 정규장 일봉 기준).** 리포트: `outputs/eda_report.md`, `outputs/eda_report.pdf`
- **모델링 8 ~ 11단계 완료(2026-09-29).** 계획: `docs/modeling_plan.md`, 리포트: `outputs/model_report.md`
- 검증 구간(2026-07 ~ 08)은 1회 평가를 마쳤다. **이 구간으로 모델·환산표·등급을 다시 고르면 안 된다.**
- 일별 점수 산출 스크립트 추가(2026-09-29): `python -m src.daily_score --date YYYY-MM-DD`(`stock_risk` 환경). 점수·등급과 함께 **오늘 종목 간 순위**(1 = 그날 가장 위험, 3일 평균 예측 기준)를 보여 준다. 결과는 `outputs/daily/risk_scores_<기준일>.csv`
  - 2026-08-31 결과가 11단계 검증 예측과 반올림 범위 안에서 같다
  - 신용·대차 테이블이 늦게 적재되면 해당 변수가 비고 경고가 나온다(2026-09-16 기준 `margin_rate`, `lend_days_20` 결측)
  - **처리 규칙(2026-09-29 결정): 경고가 나오면 그 결과는 쓰지 않고, 원천 적재가 끝난 뒤 다시 실행한다.** 전일 값 대체는 하지 않는다
  - 2026-09-15 결과(`risk_scores_2026-09-15.csv`)는 결측 경고 없이 정상 산출(201종목, 순위 1 ~ 201 중복 없음)
  - 2026-09-16 결과(`outputs/daily/risk_scores_2026-09-16.csv`)는 신용·대차 결측 상태라 재실행 대기 중이다
- **보유 종목 위험도 대시보드(Streamlit) 초안(2026-09-29)**: `streamlit run app/dashboard.py`(`stock_risk` 환경)
  - 표시: 종목별 위험도 5단계, KOSPI200 내 순위·위치, 전일 대비 등급 변화, 최근 등급 추이, 보유 vs 시장 등급 분포, 등급 설명(학습 기간 실제 결과)
  - **종목별 예상 낙폭(%)은 표시하지 않는다**(결정: 개별 수치 오차가 커서. 검증 평균 절대오차 4.1%p)
  - 보유 종목 입력: 화면 직접 입력 / 고객 조회(연령대 선택 → 고객 선택). 목록은 연령대(20대 이하 ~ 70대 이상)별로 기준일 KOSPI200 보유 종목이 가장 많은 고객 10명(`customers_by_age`). 교육용 데이터라 이름 표시 허용(사용자 확인)
  - **고객정보 DB 연결(2026-09-29)**: 같은 MySQL 서버의 `fintech_realistic_class`(`.env` DB 계정에 조회 권한 부여, 읽기 전용). `customers`(80,000명), `customer_stock_positions`(고객 × 계좌 × 종목, 85,356건)를 쓴다. `customer_stock_lots`는 수량 합이 positions와 같아 쓰지 않는다. 기준일 이후 매수분은 빼고, 계좌 간 같은 종목은 합산(`src/holdings.py`)
  - 보유 종목 600개 중 KOSPI200 구성종목은 50개(포지션의 9.1%). KOSPI200 종목을 1개 이상 가진 고객 6,825명. 나머지 보유는 '평가하지 않은 종목'으로 표시
  - 교육용 데이터지만 리포트·로그에 개별 고객 정보를 남기지 않는다
  - 데이터는 `outputs/daily/`의 점수 파일. `usable = False`(원천 미적재) 날짜는 결과를 보여 주지 않는다
  - 시장 '매우 높음' 비중이 40% 이상이면 순위로 비교하라는 안내를 띄운다
- **시장 현황(실시간, 2026-09-29)**: 대시보드 상단에 KOSPI 현재가·전일 대비·장중 1분봉, 원/달러 매매기준율·전일 같은 시각 대비. 토스증권 Open API(`src/market_live.py`), 5분마다 자동 갱신(`st.fragment`)
  - 필요: `.env`에 `TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET`, 토스증권 개발자센터에 서버 IP 허용 등록. 키가 없으면 안내만 표시
  - 토큰은 client당 1개만 유효(재발급 시 이전 토큰 무효) → 프로세스당 클라이언트 1개(`st.cache_resource`)
  - 공포탐욕지수는 제외(사용자 결정: 한국 지수를 받을 수 없음)
- **원천 데이터 일별 수집(2026-09-29)**: `src/collect.py` → `data/live/<DB 테이블명>.parquet`(DB와 같은 컬럼, 나중에 DB 적재 예정)
  - KRX 정규장 일봉: KRX Open API `stk_bydd_trd`(`.env` KRX_KEY). 09-16 값이 DB와 229종목·전 컬럼 일치. 원주가 저장, 수정주가는 읽을 때 back-adjust(기준가/전일 원종가 규칙, DB 이벤트 31건으로 검증)
  - 나머지(통합 일봉, 투자자별·기관 세부, 프로그램, 공매도, 신용 T+1, 대차, 지수·국채, 시장 투자자별): 토스증권 Open API. 09-15 값이 DB와 일치
  - DB의 09-16 토스 계열 값은 장중 잠정치였다(통합 거래량 201종목 모두 다름) → 수집 확정치가 대체
  - `load_table(..., live=True)`: DB + live 합침(같은 키는 live 우선). `daily_score`만 사용(학습 노트북 영향 없음)
  - KOSPI200 구성종목은 수집원이 없어 DB 최신 스냅샷(09-16)을 계속 쓴다. **12월 정기변경 때 갱신 필요**
  - 백필 완료: 09-16 ~ 09-28(7거래일, 추석 09-24·25 휴장). 점수 파일도 생성(모두 usable)
  - **자동 실행**: Windows 작업 스케줄러 `StockRisk\DailyCollect`(매일 09:00·11:00, 놓친 실행은 PC가 켜진 뒤 실행, 로그온 상태에서만) → `wsl.exe -d Ubuntu-24.04 -u sd2-01 -- scripts/run_collect.sh`. 정의 파일 `scripts/stock_risk_collect_task.xml`(재등록: `schtasks /Create /TN "StockRisk\DailyCollect" /XML <경로>`). WSL cron 항목은 중복 실행 방지를 위해 지웠다. 스크립트 동작:(마지막 수집일 ~ 어제를 수집 후 점수 계산, 중복 실행 방지, 로그 `logs/collect_YYYYMM.log`). KRX는 다음 날 아침 게시, 신용은 T+1 새벽 확정이라 마지막 날을 매번 다시 받는다
  - 토스 토큰은 `~/.cache/stock_risk/toss_token.json`(권한 600)으로 대시보드와 공유(재발급 시 이전 토큰 무효 문제 대응)
- 다음 후보(사용자 결정 필요): 수집 데이터 DB 적재, 재학습 주기, KOSPI200 밖 보유 종목(코스닥 등) 평가 여부

## 확정된 결정
| 항목 | 결정 |
|---|---|
| 가격 원천 | KRX 정규장 일봉 `krx_daily_candles`(기존 `stock_daily_candles`는 수급 분모용 통합 거래량으로만 사용) |
| 타깃 | `mdd_low = min(Low[t+1..t+5]) / Close[t] − 1`, 연속값, 양수 그대로. 거래정지 관련 쌍 제외 |
| 기간 | 워밍업 2024-01-02 ~ 03-29, 학습 2024-04-01 ~ 2026-06-30, 변수 윈도우 최대 60거래일 |
| 예측 시점 | t+1일 장 시작 전. t일 장 마감 후 공표 수급은 lag 없이 사용 |
| 용도 | **종목별 위험 등급 표시**("내일 폭락" 예측은 목적이 아니다) |
| 점수 | D 방식: 예측값의 학습 기간 백분위(0 ~ 100). 환산표 `outputs/model/score_reference.csv` |
| 등급 | 5등급(20점 단위). 2026-09-29 확정(검증 평가 전) |
| 평활화 | 종목별 최근 3일 예측 MDD_5 평균을 같은 환산표로 점수화. 2026-09-29 확정(검증 평가 전) |
| 최종 모델 | M3 `VolResidModel`(변동성 3개 Ridge 수준 + 종목 간 잔차 LightGBM 100트리), `outputs/model/vol_resid_model.joblib` |

## 검증 구간 결과 (2026-07 ~ 08, 1회)
- 강한 충격 국면: KOSPI -19.5%, 일간 변동성 5.1%(학습 2.1%), 평균 MDD_5 -6.5%
- 날짜별 IC 0.456, 풀링 Spearman 0.340(교차검증 0.339 / 0.321, B1 0.427 / 0.268)
- 등급 단조: 실제 평균 -1.5% → -7.5%, -10% 하락 비율 0% → 27.8%. 단, 종목-일 72%가 '매우 높음'
- 날짜별 수준 상관 -0.14(7월 22 ~ 27일 급락 미반영). 횡단면 보정(LightGBM) 단독 IC -0.04로 기여 작음

## 성능 요약 (학습 구간 교차검증, 표본 밖)
- 풀링 Spearman 0.321, 날짜별 IC 0.339, MAE 0.0296(B1 `gk_5` 선형: 0.299 / 0.316 / 0.0299)
- 등급별 실제 평균 낙폭: 매우 낮음 -2.6% → 매우 높음 -6.9%. -10% 이상 하락 비율 1.7% → 25.2%
- 한계: 시장 전체 급락(예: 2026-03)은 미리 반영하지 못하고, 변동성이 오른 뒤에 따라간다.

## 파일 위치
| 경로 | 내용 |
|---|---|
| `outputs/notebooks/01 ~ 11_*.ipynb`, `03b_*.ipynb` | 단계별 노트북(`stock_risk` 커널, 위에서부터 다시 실행 가능) |
| `outputs/model/` | 최종 모델, 점수 환산표, 등급표(`grade_table_s3.csv` = 3일 평균 기준), 메타데이터 |
| `app/dashboard.py`, `src/dashboard_data.py`, `src/holdings.py`, `src/market_live.py` | 보유 종목 위험도 대시보드와 데이터·고객 보유 종목 함수 |
| `src/collect.py`, `scripts/run_collect.sh`, `data/live/`, `logs/` | 원천 데이터 일별 수집(KRX·토스), cron 실행 스크립트, 수집 데이터, 로그 |
| `src/daily_score.py`, `outputs/daily/` | 일별 점수·등급·오늘 순위 산출 스크립트와 결과 CSV |
| `outputs/model_report.md` | 모델 리포트(선택, 점수·등급, 검증 평가) |
| `outputs/model_cv_results.csv` | 9단계 후보·제거 실험 결과 |
| `outputs/feature_candidates.csv`, `outputs/data_dictionary.md` | 변수 평가표, 데이터 사전(결정 사항 포함) |
| `outputs/archive_v1/` | v1(기존 DB 일봉) 리포트 |
| `src/` | 재사용 코드(`data`, `universe`, `target`, `features`, `evaluate`, `modeling`, `scoring`, `plotting`, `report_pdf`) |
| `data/cache/` | Parquet 캐시(약 270MB, git 제외 대상). 노트북을 다시 실행하면 DB에서 다시 만들어진다 |
