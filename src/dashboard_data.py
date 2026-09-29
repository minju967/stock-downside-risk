"""보유 종목 위험도 대시보드용 데이터 함수(`app/dashboard.py`가 쓴다).

입력은 `src.daily_score`가 만든 `outputs/daily/risk_scores_<기준일>.csv`와 `.json`,
등급 설명은 `outputs/model/grade_table_s3.csv`(학습 구간 교차검증, 3일 평균 기준 실제 결과)다.
종목별 예상 낙폭은 개별 정확도가 낮아 화면에 쓰지 않는다(2026-09-29 결정).
"""
import json
from pathlib import Path

import pandas as pd

from src.scoring import GRADE_LABELS

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "outputs" / "daily"
GRADE_TABLE = ROOT / "outputs" / "model" / "grade_table_s3.csv"
GRADE_RANK = {g: i for i, g in enumerate(GRADE_LABELS)}


def list_dates() -> pd.DataFrame:
    """점수 파일이 있는 기준일 목록.

    입력: 없음
    출력: DataFrame[date(str), usable(bool), warn(dict)] 최신순. json이 없으면 usable = False
    누수: 해당 없음
    """
    rows = []
    for p in DAILY.glob("risk_scores_*.csv"):
        d = p.stem.replace("risk_scores_", "")
        meta = p.with_suffix(".json")
        info = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        rows.append({"date": d, "usable": bool(info.get("usable", False)),
                     "warn": info.get("결측률_높은_변수", {} if info else {"점검 파일 없음": 1})})
    return pd.DataFrame(rows, columns=["date", "usable", "warn"]).sort_values("date", ascending=False).reset_index(drop=True)


def load_check(date: str) -> dict:
    """기준일 점검 정보(daily_score가 저장한 json). 없으면 빈 dict. 구성종목 스냅샷·초과 편입 정보를 담는다."""
    p = DAILY / f"risk_scores_{date}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def load_scores(date: str) -> pd.DataFrame:
    """기준일 점수 파일을 읽는다. 출력: daily_score 결과 DataFrame(종목코드는 6자리 문자열)."""
    return pd.read_csv(DAILY / f"risk_scores_{date}.csv", dtype={"종목코드": str})


def grade_reference() -> pd.DataFrame:
    """등급 설명표(과거 실제 결과).

    출력: DataFrame[등급, 점수, 평균_낙폭, 하위10_낙폭, 하락10_확률] (낙폭은 % 단위, 확률은 0 ~ 1)
    누수: 학습 구간 교차검증 예측으로 만든 고정표
    """
    g = pd.read_csv(GRADE_TABLE)
    return pd.DataFrame({"등급": g["등급"], "점수": g["점수"], "평균_낙폭": g["실제_평균"] * 100,
                         "하위10_낙폭": g["실제_하위10"] * 100, "하락10_확률": g["하락10_비율"]})


def holdings_view(holdings: pd.DataFrame, date: str, history: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """보유 종목에 위험도, 오늘 순위, 전일 대비 변화, 최근 등급 추이를 붙인다.

    입력: holdings(symbol, name, quantity), date(기준일), history(추이에 쓸 최근 기준일 수)
    출력: (평가 대상 DataFrame[symbol, 종목명, quantity, 등급, 점수(0 ~ 100, 3일 평균), 순위, 종목수, 상위_퍼센트, 전일_등급, 변화, 추이],
           평가 제외 DataFrame[symbol, name, quantity]: 그날 KOSPI200 구성종목이 아닌 보유 종목)
           변화: +n = n단계 위험 상승, -n = 하락, 0 = 같음, NaN = 전일 정보 없음
           추이: 오래된 날부터 [(기준일, 등급 또는 None)] 목록. 원천 미적재 날짜는 건너뛴다
    누수: 기준일 이하 점수 파일만 쓴다
    """
    h = holdings.copy()
    h["symbol"] = h["symbol"].astype(str).str.zfill(6)
    cur = load_scores(date)
    dates = list_dates()
    past = dates[(dates.date < date) & dates.usable].date.tolist()      # 최신순
    trend_dates = sorted(past[: history - 1]) + [date]
    by_date = {d: load_scores(d).set_index("종목코드")["등급"] for d in trend_dates}

    m = h.merge(cur, left_on="symbol", right_on="종목코드", how="left")
    excluded = m[m["등급"].isna()][["symbol", "name", "quantity"]].reset_index(drop=True)
    m = m[m["등급"].notna()].copy()
    prev = by_date[past[0]] if past else pd.Series(dtype=object)
    m["전일_등급"] = m["symbol"].map(prev)
    m["변화"] = [GRADE_RANK[g] - GRADE_RANK[p] if isinstance(p, str) else float("nan")
                 for g, p in zip(m["등급"], m["전일_등급"])]
    m["추이"] = [[(d, by_date[d].get(s)) for d in trend_dates] for s in m["symbol"]]
    m = m.rename(columns={"오늘_상위_퍼센트": "상위_퍼센트"})
    cols = ["symbol", "종목명", "quantity", "등급", "점수", "순위", "종목수", "상위_퍼센트", "전일_등급", "변화", "추이"]
    return m[cols].sort_values("순위").reset_index(drop=True), excluded


def market_grade_share(date: str) -> pd.Series:
    """기준일 KOSPI200 전체의 등급 비중(GRADE_LABELS 순서, 합 1)."""
    s = load_scores(date)["등급"].value_counts(normalize=True)
    return s.reindex(GRADE_LABELS).fillna(0.0)
