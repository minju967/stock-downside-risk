"""일별 위험도 점수 산출: 예상 낙폭, 점수(D 방식), 등급, 오늘 종목 간 순위.

실행: python -m src.daily_score --date 2026-09-16
출력: outputs/daily/risk_scores_<기준일>.csv (순위 1 = 그날 가장 위험)
      outputs/daily/risk_scores_<기준일>.json (점검 결과. usable = False면 원천 데이터 미적재로 쓰지 않는다)
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import features as F
from src.data import load_table
from src.db import get_engine, read_sql
from src.scoring import to_grade, to_score
from src.universe import build_universe, load_calendar, load_membership

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs" / "model"
OUT_DIR = ROOT / "outputs" / "daily"
DATA_START = "2024-01-01"          # DB 데이터 시작
LOOKBACK_DAYS = 250                # 읽을 과거 기간(달력일). 변수 윈도우 최대 60거래일 + 평활화 여유
FLOW_TABLES = ["stock_investor_trading", "stock_institution_breakdown", "stock_program_trades", "stock_short_selling",
               "stock_credit_trades", "stock_securities_lending", "market_indicator_investor_trading"]


def add_daily_rank(d: pd.DataFrame, pred: str = "pred_s") -> pd.DataFrame:
    """같은 날 구성종목끼리 예측 낙폭 순위를 붙인다.

    입력: d(trade_date, pred 포함. pred는 음수일수록 위험)
    출력: d 복사본 + [순위(1 = 그날 가장 위험), 종목수, 오늘_상위_퍼센트(순위 / 종목수 × 100)]
    누수: 같은 날 예측값끼리만 비교한다(누수 없음)
    """
    d = d.copy()
    g = d.groupby("trade_date")[pred]
    d["순위"] = g.rank(method="min", ascending=True).astype("Int64")
    d["종목수"] = g.transform("count").astype("Int64")
    d["오늘_상위_퍼센트"] = (d["순위"] / d["종목수"] * 100).round(1)
    return d


def membership_note(membership: pd.DataFrame, t: pd.Timestamp, stocks: pd.DataFrame, base: int = 200) -> dict:
    """기준일에 쓴 구성종목 스냅샷과, 종목 수가 base(200)를 넘을 때 그 원인이 된 편입 종목을 찾는다.

    입력: membership(load_membership 결과), t(기준일), stocks(symbol, name, list_date), base(정상 종목 수)
    출력: dict{스냅샷_기준일, 구성종목수, 초과_편입: [{symbol, name, list_date, 편입_스냅샷}]}
          초과_편입은 스냅샷을 거슬러 올라가며 종목 수가 base에서 base 초과로 바뀐 시점에 편입만 되고 편출이 없던 종목이다
    누수: 기준일 이하 스냅샷만 쓴다
    """
    snaps = sorted(d for d in membership["snapshot_date"].unique() if d <= t)
    sets = {d: set(g["symbol"]) for d, g in membership.groupby("snapshot_date")}
    cur = snaps[-1]
    note = {"스냅샷_기준일": str(pd.Timestamp(cur).date()), "구성종목수": len(sets[cur]), "초과_편입": []}
    if len(sets[cur]) <= base:
        return note
    info = stocks.set_index("symbol")
    for prev, nxt in zip(snaps[-2::-1], snaps[:0:-1]):          # 최근 전환부터 거꾸로
        if len(sets[prev]) <= base < len(sets[nxt]):
            for sym in sorted(sets[nxt] - sets[prev]):
                if sym in sets[cur]:
                    ld = info["list_date"].get(sym)
                    note["초과_편입"].append({"symbol": sym, "name": info["name"].get(sym, sym),
                                              "list_date": None if pd.isna(ld) else str(pd.Timestamp(ld).date()),
                                              "편입_스냅샷": str(pd.Timestamp(nxt).date())})
            break
    return note


def score_date(date: str, smooth: int | None = None, engine=None) -> tuple[pd.DataFrame, dict]:
    """기준일(t)의 전 종목 위험도를 계산한다. t+1일 장 시작 전에 쓰는 값이다.

    입력: date(YYYY-MM-DD. 거래일이 아니면 그 이전 마지막 거래일), smooth(평활화 일수, 기본은 model_meta 설정)
    출력: (결과 DataFrame: 기준일, 순위, 종목수, 오늘_상위_퍼센트, 종목코드, 종목명, 업종,
           예상_낙폭(3일 평균, %), 점수, 등급, 당일_예상_낙폭(t일 예측만, t 종가 대비 %), 당일_점수,
           점검 정보 dict: 결측률이 높은 모델 변수 등)
    누수: DB에서 기준일 이하 데이터만 읽는다. 구성종목도 기준일 이하 스냅샷만 쓴다.
          모델·환산표는 학습 구간에서 고정한 것을 쓴다
    """
    engine = engine or get_engine()
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    smooth = smooth or meta.get("smoothing", {}).get("days", 1)
    model = joblib.load(MODEL_DIR / "vol_resid_model.joblib")
    ref = pd.read_csv(MODEL_DIR / "score_reference.csv")
    REF = -ref["pred_mdd5"].to_numpy()

    start = max(pd.Timestamp(DATA_START), pd.Timestamp(date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    # 거래일 캘린더: DB 캘린더에 data/live 수집분(KRX 일봉 날짜)을 더한다
    c = load_table("krx_daily_candles", start, date, engine, cache=False, live=True)
    cal = load_calendar(start, date, engine).union(pd.DatetimeIndex(c["trade_date"].unique())).sort_values()
    if len(cal) == 0:
        raise ValueError(f"{date} 이전 거래일이 없다")
    t = cal.max()
    end = t.strftime("%Y-%m-%d")
    c = c[c["trade_date"] <= t]
    membership = load_membership(end, engine)
    uni = build_universe(cal, membership)
    load = lambda tb: load_table(tb, start, end, engine, cache=False, live=True)   # DB + data/live 수집분
    vol_all = load("stock_daily_candles")[["symbol", "trade_date", "volume"]]
    idx = load("market_indicator_daily_candles")
    flows = {tb: load(tb) for tb in FLOW_TABLES}
    stocks = read_sql("SELECT symbol, name, industry, list_date FROM stocks", engine)
    stocks["symbol"] = stocks["symbol"].astype(str)
    industry = stocks.set_index("symbol")["industry"]

    feat, _, _ = F.build_all(c, idx, uni, flows, industry, clean=False, vol_all=vol_all)
    feat["abs_ret_20"] = feat["ret_20"].abs()
    days = cal[-smooth:]
    x = feat[feat.trade_date.isin(days)].reset_index(drop=True)
    x["pred"] = model.predict(x)
    x = x.sort_values(["symbol", "trade_date"])
    x["pred_s"] = x.groupby("symbol")["pred"].transform(lambda s: s.rolling(smooth, min_periods=1).mean())
    today = x[x.trade_date == t].copy()

    # 점검: 기준일 모델 변수 결측률(원천 테이블 적재 지연 확인용)
    cols = list(dict.fromkeys(model.level_features + model.cs_features))
    na = today[cols].isna().mean()
    check = {"기준일": end, "종목수": int(len(today)), "평활화_일수": smooth,
             "결측률_높은_변수": {k: round(float(v), 3) for k, v in na[na > 0.2].items()},
             "구성종목": membership_note(membership, t, stocks)}

    today["점수"] = to_score(today["pred_s"], REF).round(1)
    today["등급"] = to_grade(today["점수"])
    today["당일_점수"] = to_score(today["pred"], REF).round(1)
    today = add_daily_rank(today)
    today = today.merge(stocks[["symbol", "name", "industry"]], on="symbol", how="left")
    out = pd.DataFrame({
        "기준일": end, "순위": today["순위"], "종목수": today["종목수"], "오늘_상위_퍼센트": today["오늘_상위_퍼센트"],
        "종목코드": today["symbol"], "종목명": today["name"], "업종": today["industry"],
        "예상_낙폭": (today["pred_s"] * 100).round(2), "점수": today["점수"], "등급": today["등급"],
        "당일_예상_낙폭": (today["pred"] * 100).round(2), "당일_점수": today["당일_점수"],
    })
    return out.sort_values("순위").reset_index(drop=True), check


def run(date: str) -> tuple[pd.DataFrame, dict]:
    """기준일 점수를 계산해 outputs/daily/에 CSV와 점검 JSON으로 저장한다. 출력: (결과, 점검 정보)."""
    out, check = score_date(date)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"risk_scores_{check['기준일']}.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    # 점검 결과(대시보드가 사용 가능 여부 판단에 쓴다): 결측 경고가 있으면 usable = False
    check["usable"] = not check["결측률_높은_변수"]
    path.with_suffix(".json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"기준일 {check['기준일']}, {check['종목수']}종목 → {path.relative_to(ROOT)}")
    if check["결측률_높은_변수"]:
        print("경고: 결측률 20% 초과 변수(원천 데이터 적재 확인 필요):", check["결측률_높은_변수"])
    return out, check


def main() -> None:
    ap = argparse.ArgumentParser(description="일별 위험도 점수와 오늘 종목 간 순위를 CSV로 저장한다")
    ap.add_argument("--date", required=True, help="기준일 YYYY-MM-DD (t일 장 마감 후 실행, t+1일 장 전 사용)")
    args = ap.parse_args()
    out, _ = run(args.date)
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
