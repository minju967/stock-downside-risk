"""원천 데이터 일별 수집(KRX Open API + 토스증권 Open API) → 로컬 Parquet(`data/live/`).

DB는 읽기 전용이므로 새로 받은 데이터는 DB 테이블과 **같은 이름·같은 컬럼**으로 `data/live/<테이블>.parquet`에 쌓는다.
나중에 DB에 그대로 넣을 수 있다(2026-09-29 결정). `src.data.load_table(..., live=True)`가 DB와 합쳐 읽는다.

원천(2026-09-29 확인: 09-15·09-16 값이 기존 DB와 모든 컬럼에서 일치)
  - krx_daily_candles: KRX Open API 유가증권 일별매매정보(stk_bydd_trd, `.env` KRX_KEY). 원주가를 저장하고
    수정주가·수정계수는 읽을 때 기존 DB와 이어 붙이며 계산한다(src.data 참고). 데이터는 다음 날 아침에 올라온다
  - stock_daily_candles(KRX+NXT 통합), 투자자별·기관 세부, 프로그램, 공매도, 신용(T+1), 대차, 지수·국채 일봉,
    시장 투자자별: 토스증권 Open API
  - kospi200_membership: 수집원이 없다. DB의 최신 스냅샷을 그대로 쓴다(정기변경 때 따로 갱신 필요)

실행
  python -m src.collect --auto                   # 마지막 수집일 다음 날 ~ 어제의 거래일을 수집하고 점수 계산
  python -m src.collect --start 2026-09-16 --end 2026-09-28
누수: 해당 없음(원천 데이터 저장). 각 날짜의 값은 그날 장 마감 뒤 확정치다(신용은 다음 날 새벽 확정)
"""
import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from src import market_live as ML
from src.db import read_sql

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "data" / "live"
KRX_URL = "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
FIRST_LIVE_DAY = "2026-09-16"          # DB의 이 날 값은 장중 잠정치라 확정치로 다시 받는다
INDICATORS = ["KOSPI", "KOSDAQ", "KR_BOND_2Y", "KR_BOND_3Y", "KR_BOND_5Y", "KR_BOND_10Y", "KR_BOND_20Y", "KR_BOND_30Y"]
INST = {"financialInvestment": "financial_investment", "insurance": "insurance", "trust": "trust",
        "privateEquityFund": "private_equity_fund", "bank": "bank",
        "otherFinancialInstitution": "other_financial_institution", "pensionFund": "pension_fund"}
KEYS = {  # 테이블별 기본키(중복 제거 기준)
    "krx_daily_candles": ["symbol", "trade_date"], "stock_daily_candles": ["symbol", "trade_date"],
    "stock_investor_trading": ["symbol", "trade_date"], "stock_institution_breakdown": ["symbol", "trade_date", "category"],
    "stock_program_trades": ["symbol", "trade_date"], "stock_short_selling": ["symbol", "trade_date"],
    "stock_credit_trades": ["symbol", "trade_date"], "stock_securities_lending": ["symbol", "trade_date"],
    "market_indicator_daily_candles": ["symbol", "trade_date"], "market_indicator_investor_trading": ["symbol", "trade_date"],
    "market_institution_breakdown": ["symbol", "trade_date", "category"],
}
log = logging.getLogger("collect")


# ---------------------------------------------------------------- 저장
def upsert(table: str, df: pd.DataFrame) -> int:
    """data/live/<table>.parquet에 기본키 기준으로 덮어쓰며 추가한다(같은 키는 새 값). 출력: 저장 후 행 수."""
    if df.empty:
        return 0
    LIVE.mkdir(parents=True, exist_ok=True)
    path = LIVE / f"{table}.parquet"
    df = df.assign(trade_date=pd.to_datetime(df["trade_date"]), collected_at=pd.Timestamp.now().floor("s"))
    if path.exists():
        df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
    df = df.drop_duplicates(KEYS[table], keep="last").sort_values(KEYS[table]).reset_index(drop=True)
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return len(df)


def live_dates(table: str) -> set:
    path = LIVE / f"{table}.parquet"
    return set(pd.read_parquet(path, columns=["trade_date"])["trade_date"].dt.date) if path.exists() else set()


# ---------------------------------------------------------------- KRX
def krx_day(d: date) -> pd.DataFrame:
    """KRX 유가증권 일별매매정보 1일치(원주가). 휴장일이거나 아직 게시 전이면 빈 DataFrame."""
    load_dotenv(ROOT / ".env")
    r = requests.get(KRX_URL, params={"basDd": d.strftime("%Y%m%d")}, headers={"AUTH_KEY": os.getenv("KRX_KEY", "")},
                     timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json().get("OutBlock_1", []))


def krx_to_table(raw: pd.DataFrame, symbols: set) -> pd.DataFrame:
    """KRX 응답 → krx_daily_candles 행(수정계수 1, 수정주가 = 원주가). 거래정지(거래량 0)는 OHLC를 종가로 채운다."""
    x = raw[raw["ISU_CD"].isin(symbols)].copy()
    n = lambda c: pd.to_numeric(x[c], errors="coerce")
    close, vol = n("TDD_CLSPRC"), n("ACC_TRDVOL")
    halted = vol == 0
    o, h, l = [n(c).where(~halted, close) for c in ("TDD_OPNPRC", "TDD_HGPRC", "TDD_LWPRC")]
    return pd.DataFrame({
        "symbol": x["ISU_CD"].astype(str), "trade_date": pd.to_datetime(x["BAS_DD"]),
        "open_price": o, "high_price": h, "low_price": l, "close_price": close, "volume": vol,
        "trading_value": n("ACC_TRDVAL"), "raw_open": o, "raw_high": h, "raw_low": l, "raw_close": close,
        "raw_volume": vol, "base_price": close - n("CMPPREVDD_PRC"), "adj_factor": 1.0,
        "listed_shares": n("LIST_SHRS"), "halted": halted.astype(int)})


# ---------------------------------------------------------------- 토스 변환
num = lambda v: None if v is None else float(v)


def _pick(records: list, days: set, key: str = "date") -> list:
    return [r for r in records if str(r.get(key))[:10] in days]


def toss_symbol(c: ML.TossClient, sym: str, days: set, kospi200: set) -> dict[str, list]:
    """종목 1개의 토스 데이터(일봉, 투자자별, 기관 세부, 프로그램, 공매도, 신용, 대차)를 DB 테이블 행으로 바꾼다."""
    n = min(100, len(days) + 10)
    out = {t: [] for t in ["stock_daily_candles", "stock_investor_trading", "stock_institution_breakdown",
                           "stock_program_trades", "stock_short_selling", "stock_credit_trades", "stock_securities_lending"]}
    for cd in _pick(c.get("/api/v1/candles", {"symbol": sym, "interval": "1d", "count": min(200, n)})["candles"], days, "timestamp"):
        out["stock_daily_candles"].append({
            "symbol": sym, "trade_date": cd["timestamp"][:10], "open_price": num(cd["openPrice"]), "high_price": num(cd["highPrice"]),
            "low_price": num(cd["lowPrice"]), "close_price": num(cd["closePrice"]), "volume": num(cd["volume"]),
            "currency": cd.get("currency", "KRW"), "adjusted": 1, "is_kospi200": int(sym in kospi200)})
    for r in _pick(c.get(f"/api/v1/stocks/{sym}/investor-trading", {"count": n})["records"], days):
        g = lambda k, f: num((r.get(k) or {}).get(f))
        fh, cfd = r.get("foreignerHolding") or {}, r.get("cfd") or {}
        out["stock_investor_trading"].append({
            "symbol": sym, "trade_date": r["date"], "updated_at": r.get("updatedAt"),
            **{f"{p}_{a}_volume": g(k, f) for p, k in [("individual", "individual"), ("foreigner", "foreigner"),
                                                         ("institution", "institution"), ("other_corp", "otherCorporation")]
               for a, f in [("buy", "buyVolume"), ("sell", "sellVolume"), ("net", "netBuyVolume")]},
            "foreigner_holding_qty": num(fh.get("holdingQuantity")), "foreigner_holding_limit": num(fh.get("limitQuantity")),
            "foreigner_holding_rate": num(fh.get("holdingRate")),
            "cfd_buy_balance_qty": num(cfd.get("buyBalanceQuantity")), "cfd_buy_balance_rate": num(cfd.get("buyBalanceRate")),
            "cfd_sell_balance_qty": num(cfd.get("sellBalanceQuantity")), "cfd_sell_balance_rate": num(cfd.get("sellBalanceRate"))})
        for k, cat in INST.items():
            b = ((r.get("institution") or {}).get("breakdown") or {}).get(k)
            if b:
                out["stock_institution_breakdown"].append({
                    "symbol": sym, "trade_date": r["date"], "category": cat, "buy_volume": num(b.get("buyVolume")),
                    "sell_volume": num(b.get("sellVolume")), "net_volume": num(b.get("netBuyVolume"))})
    for r in _pick(c.get(f"/api/v1/stocks/{sym}/program-trades", {"count": n})["records"], days):
        a, na = r.get("arbitrage") or {}, r.get("nonArbitrage") or {}
        out["stock_program_trades"].append({
            "symbol": sym, "trade_date": r["date"], "arbitrage_buy_volume": num(a.get("buyVolume")),
            "arbitrage_sell_volume": num(a.get("sellVolume")), "arbitrage_net_volume": num(a.get("netBuyVolume")),
            "non_arbitrage_buy_volume": num(na.get("buyVolume")), "non_arbitrage_sell_volume": num(na.get("sellVolume")),
            "non_arbitrage_net_volume": num(na.get("netBuyVolume"))})
    for r in _pick(c.get(f"/api/v1/stocks/{sym}/short-selling", {"count": n})["records"], days):
        out["stock_short_selling"].append({
            "symbol": sym, "trade_date": r["date"], "updated_at": r.get("updatedAt"),
            "short_volume": num(r.get("shortSellingVolume")), "short_amount": num(r.get("shortSellingAmount")),
            "short_volume_rate": num(r.get("shortSellingVolumeRate")), "short_amount_rate": num(r.get("shortSellingAmountRate"))})
    for r in _pick(c.get(f"/api/v1/stocks/{sym}/credit-trades", {"count": n})["records"], days):
        m, s = r.get("marginLoan") or {}, r.get("stockLoan") or {}
        out["stock_credit_trades"].append({
            "symbol": sym, "trade_date": r["date"], "updated_at": r.get("updatedAt"),
            "margin_new_qty": num(m.get("newQuantity")), "margin_return_qty": num(m.get("returnQuantity")),
            "margin_balance_qty": num(m.get("balanceQuantity")), "margin_balance_rate": num(m.get("balanceRate")),
            "margin_trading_rate": num(m.get("tradingRate")),
            "stockloan_new_qty": num(s.get("newQuantity")), "stockloan_return_qty": num(s.get("returnQuantity")),
            "stockloan_balance_qty": num(s.get("balanceQuantity")), "stockloan_balance_rate": num(s.get("balanceRate")),
            "stockloan_trading_rate": num(s.get("tradingRate"))})
    for r in _pick(c.get(f"/api/v1/stocks/{sym}/securities-lending", {"count": n})["records"], days):
        out["stock_securities_lending"].append({
            "symbol": sym, "trade_date": r["date"], "updated_at": r.get("updatedAt"),
            "execution_qty": num(r.get("executionQuantity")), "repayment_qty": num(r.get("repaymentQuantity")),
            "balance_qty": num(r.get("balanceQuantity")), "balance_amount": num(r.get("balanceAmount"))})
    return out


def toss_market(c: ML.TossClient, days: set) -> dict[str, list]:
    """지수·국채 일봉과 시장(KOSPI·KOSDAQ) 투자자별·기관 세부 매매대금."""
    n = min(100, len(days) + 10)
    out = {"market_indicator_daily_candles": [], "market_indicator_investor_trading": [], "market_institution_breakdown": []}
    for sym in INDICATORS:
        for cd in _pick(c.get(f"/api/v1/market-indicators/{sym}/candles", {"interval": "1d", "count": n})["candles"], days, "timestamp"):
            out["market_indicator_daily_candles"].append({
                "symbol": sym, "trade_date": cd["timestamp"][:10], "open_price": num(cd["openPrice"]), "high_price": num(cd["highPrice"]),
                "low_price": num(cd["lowPrice"]), "close_price": num(cd["closePrice"]), "volume": num(cd.get("volume"))})
    for sym in ["KOSPI", "KOSDAQ"]:
        for r in _pick(c.get(f"/api/v1/market-indicators/{sym}/investor-trading", {"interval": "1d", "count": n})["records"], days):
            g = lambda k, f: num((r.get(k) or {}).get(f))
            out["market_indicator_investor_trading"].append({
                "symbol": sym, "trade_date": r["date"], "updated_at": r.get("updatedAt"),
                **{f"{p}_{a}_amount": g(k, f) for p, k in [("individual", "individual"), ("foreigner", "foreigner"),
                                                             ("institution", "institution"), ("other_corp", "otherCorporation")]
                   for a, f in [("buy", "buyAmount"), ("sell", "sellAmount")]}})
            for k, cat in INST.items():
                b = ((r.get("institution") or {}).get("breakdown") or {}).get(k)
                if b:
                    out["market_institution_breakdown"].append({
                        "symbol": sym, "trade_date": r["date"], "category": cat,
                        "buy_amount": num(b.get("buyAmount")), "sell_amount": num(b.get("sellAmount"))})
    return out


# ---------------------------------------------------------------- 실행
def collect(days: list[date]) -> list[date]:
    """주어진 날짜들을 수집한다. 출력: 실제로 거래일이어서 수집한 날짜."""
    krx_symbols = set(read_sql("SELECT DISTINCT symbol FROM krx_daily_candles")["symbol"].astype(str))
    snap = read_sql("SELECT symbol FROM kospi200_membership WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM kospi200_membership)")
    kospi200 = set(snap["symbol"].astype(str))
    krx_symbols |= kospi200
    trading, krx_rows = [], []
    for d in days:
        raw = krx_day(d)
        if raw.empty:
            log.info("%s: KRX 데이터 없음(휴장일이거나 아직 게시 전)", d)
            continue
        trading.append(d)
        krx_rows.append(krx_to_table(raw, krx_symbols))
    if not trading:
        return []
    upsert("krx_daily_candles", pd.concat(krx_rows, ignore_index=True))
    tdays = {d.isoformat() for d in trading}
    c = ML.TossClient(min_interval=0.12)                     # 초당 약 8회(종목 수급 그룹 한도 10회 아래)
    acc: dict[str, list] = {}
    for part in [toss_market(c, tdays)] + [toss_symbol(c, s, tdays, kospi200) for s in sorted(kospi200)]:
        for t, rows in part.items():
            acc.setdefault(t, []).extend(rows)
    for t, rows in acc.items():
        n_after = upsert(t, pd.DataFrame(rows))
        got = {str(r["trade_date"])[:10] for r in rows}
        miss = sorted(tdays - got)
        log.info("%s: %d행 수집(누적 %d)%s", t, len(rows), n_after, f", 빠진 날짜 {miss}" if miss else "")
    return trading


def auto_range(today: date | None = None) -> list[date]:
    """마지막으로 수집한 KRX 날짜 다음 날(없으면 FIRST_LIVE_DAY) ~ 어제의 평일."""
    today = today or date.today()
    done = live_dates("krx_daily_candles")
    start = (max(done) + timedelta(days=1)) if done else date.fromisoformat(FIRST_LIVE_DAY)
    # 신용은 T+1 확정이라 마지막 날을 한 번 더 받는다(전날 수집 때 비었을 수 있음)
    if done:
        start = min(start, max(done))
    days, d = [], start
    while d < today:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def main() -> None:
    ap = argparse.ArgumentParser(description="KRX·토스증권 원천 데이터 일별 수집 → data/live/")
    ap.add_argument("--auto", action="store_true", help="마지막 수집일 다음 날 ~ 어제를 수집")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--no-score", action="store_true", help="수집 뒤 위험도 점수를 계산하지 않는다")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    if args.auto:
        days = auto_range()
    else:
        s, e = date.fromisoformat(args.start), date.fromisoformat(args.end or args.start)
        days = [s + timedelta(days=i) for i in range((e - s).days + 1) if (s + timedelta(days=i)).weekday() < 5]
    log.info("수집 대상 %d일: %s", len(days), [d.isoformat() for d in days])
    trading = collect(days)
    log.info("수집한 거래일: %s", [d.isoformat() for d in trading])
    if trading and not args.no_score:
        from src.daily_score import run as score_run
        for d in trading:
            score_run(d.isoformat())


if __name__ == "__main__":
    main()
