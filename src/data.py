"""DB 조회 결과를 Parquet으로 캐시해 읽는 함수."""
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from src.db import read_sql

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

# 테이블별로 읽을 컬럼(메타 컬럼 collected_at 제외)
_COLUMNS = {
    "stock_daily_candles": "symbol, trade_date, open_price, high_price, low_price, close_price, volume, adjusted",
    # KRX 정규장 일봉(2026-09 적재). 가격은 수정주가, raw_*는 원주가, listed_shares는 그날의 상장주식수
    "krx_daily_candles": ("symbol, trade_date, open_price, high_price, low_price, close_price, volume, trading_value, "
                          "raw_close, raw_volume, adj_factor, listed_shares, halted"),
    "market_indicator_daily_candles": "symbol, trade_date, open_price, high_price, low_price, close_price, volume",
    "stock_investor_trading": "*",
    "stock_program_trades": "*",
    "stock_short_selling": "*",
    "stock_credit_trades": "*",
    "stock_securities_lending": "*",
    "stock_institution_breakdown": "*",
    "market_indicator_investor_trading": "*",
    "market_institution_breakdown": "*",
}


def load_table(table: str, start: str, end: str, engine: Engine | None = None, refresh: bool = False,
               cache: bool = True, live: bool = False) -> pd.DataFrame:
    """일자 단위 테이블을 기간으로 잘라 읽고 `data/cache/`에 Parquet으로 저장해 재사용한다.

    입력: table(_COLUMNS의 키), start/end(YYYY-MM-DD, trade_date 양끝 포함), 선택적 엔진, refresh(캐시 무시),
          cache(False면 캐시를 읽지도 쓰지도 않는다. 일별 점수 산출용),
          live(True면 `data/live/`에 수집한 데이터를 합친다. 같은 키는 수집 값이 DB 값을 대체한다. src.collect 참고)
    출력: DataFrame. trade_date는 datetime64, decimal 컬럼은 float64, symbol은 문자열
    누수: 해당 없음(호출자가 기간을 통제한다). 검증 구간을 들여다보지 않으려면 end를 학습 구간 끝으로 둔다.
    """
    if live:
        return _merge_live(table, load_table(table, start, end, engine, refresh, cache=False), start, end)
    path = CACHE / f"{table}_{start}_{end}.parquet"
    if cache and path.exists() and not refresh:
        return pd.read_parquet(path)
    cols = _COLUMNS[table]
    df = read_sql(f"SELECT {cols} FROM {table} WHERE trade_date BETWEEN :s AND :e", engine, params={"s": start, "e": end})
    df = df.drop(columns=[c for c in ("collected_at",) if c in df.columns])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in df.columns:
        if df[c].dtype == object and c not in ("symbol", "category"):
            df[c] = pd.to_numeric(df[c], errors="coerce")   # MySQL decimal -> float
    df["symbol"] = df["symbol"].astype(str)
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


LIVE = ROOT / "data" / "live"
_KEYS = {"stock_institution_breakdown": ["symbol", "trade_date", "category"],
         "market_institution_breakdown": ["symbol", "trade_date", "category"]}


def _merge_live(table: str, db: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """DB 조회 결과에 data/live/<table>.parquet를 합친다(같은 키는 live 우선). 컬럼은 DB 조회와 같게 맞춘다.

    krx_daily_candles는 live 기간의 수정 이벤트(기준가 ≠ 전일 원종가)를 찾아 그 이전 모든 날짜의 수정주가·거래량·계수를
    다시 계산한다. 규칙은 기존 DB와 같다(2026-09-29 검증: 이벤트 31건 모두 계수 변화 = 기준가/전일 원종가).
    누수: 해당 없음(각 날짜 값은 그날 확정치). 단, 수정주가는 뒤 이벤트로 과거 값이 바뀌는 back-adjust 방식이다(DB와 동일)
    """
    path = LIVE / f"{table}.parquet"
    if not path.exists():
        return db
    lv = pd.read_parquet(path)
    lv = lv[(lv["trade_date"] >= pd.Timestamp(start)) & (lv["trade_date"] <= pd.Timestamp(end))].copy()
    if lv.empty:
        return db
    keys = _KEYS.get(table, ["symbol", "trade_date"])
    for c in ("updated_at",):
        if c in lv.columns and lv[c].dtype == object:
            lv[c] = pd.to_datetime(lv[c], utc=True, errors="coerce").dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    keep_base = table == "krx_daily_candles"
    cols = list(db.columns) + (["base_price"] if keep_base and "base_price" not in db.columns else [])
    lv = lv[[c for c in cols if c in lv.columns]]
    lv["_live"] = True
    k_db = pd.MultiIndex.from_frame(db[keys].astype({"symbol": str}))
    k_lv = pd.MultiIndex.from_frame(lv[keys].astype({"symbol": str}))
    out = pd.concat([db[~k_db.isin(k_lv)].assign(_live=False), lv], ignore_index=True)
    if keep_base:
        out = _readjust_krx(out)
        out = out.drop(columns=[c for c in ["base_price"] if c not in db.columns])
    out = out.drop(columns="_live")
    return out.sort_values(keys).reset_index(drop=True)


def _readjust_krx(d: pd.DataFrame) -> pd.DataFrame:
    """live 행 중 수정 이벤트가 있으면 그 이전 행들의 수정값에 비율을 곱한다(가격 ×r, 거래량 ÷r, 계수 ×r)."""
    d = d.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    prev_raw = d.groupby("symbol")["raw_close"].shift()
    ratio = (d["base_price"] / prev_raw).where(d["_live"] & prev_raw.notna() & (prev_raw > 0), 1.0).fillna(1.0)
    ratio = ratio.where((ratio - 1).abs() > 1e-9, 1.0)
    # 각 행의 배수 = 그 행보다 뒤(같은 종목)의 이벤트 비율 곱
    later = ratio.groupby(d["symbol"]).transform(lambda r: r[::-1].cumprod()[::-1].shift(-1, fill_value=1.0))
    if (later != 1.0).any():
        for c in ["open_price", "high_price", "low_price", "close_price"]:
            d[c] = d[c] * later
        d["volume"] = d["volume"] / later
        d["adj_factor"] = d["adj_factor"] * later
    return d
