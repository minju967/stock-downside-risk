"""KOSPI200 유니버스(point-in-time) 구성 함수."""
import pandas as pd
from sqlalchemy.engine import Engine

from src.db import read_sql

# 제외 종목. v1(기존 DB 일봉)에서는 가격이 없던 003410, 010620, 042670을 뺐으나,
# KRX 일봉에 세 종목이 모두 들어와 v2부터는 제외하지 않는다(2026-09-28 결정).
EXCLUDED_SYMBOLS: tuple[str, ...] = ()


def load_membership(end: str, engine: Engine | None = None) -> pd.DataFrame:
    """`end` 이하 기준일의 구성종목 스냅샷을 읽는다.

    입력: end(YYYY-MM-DD, 이 날짜 이하 스냅샷만), 선택적 엔진
    출력: DataFrame[snapshot_date(datetime64), symbol(str)]
    누수: 스냅샷 기준일 이후 정보를 포함하지 않는다. end를 분석 구간 끝으로 두면 이후 스냅샷은 읽지 않는다.
    """
    m = read_sql(
        "SELECT snapshot_date, symbol FROM kospi200_membership WHERE snapshot_date <= :end",
        engine, params={"end": end},
    )
    m["snapshot_date"] = pd.to_datetime(m["snapshot_date"])
    return m


def load_calendar(start: str, end: str, engine: Engine | None = None) -> pd.DatetimeIndex:
    """`stock_daily_candles`에 있는 거래일 캘린더를 읽는다.

    입력: start, end(YYYY-MM-DD, 양끝 포함), 선택적 엔진
    출력: 오름차순 DatetimeIndex
    누수: 해당 없음(날짜 목록만)
    """
    d = read_sql(
        "SELECT DISTINCT trade_date FROM stock_daily_candles WHERE trade_date BETWEEN :s AND :e ORDER BY 1",
        engine, params={"s": start, "e": end},
    )
    return pd.DatetimeIndex(pd.to_datetime(d["trade_date"]), name="trade_date")


def build_universe(calendar: pd.DatetimeIndex, membership: pd.DataFrame,
                   exclude: tuple[str, ...] = EXCLUDED_SYMBOLS) -> pd.DataFrame:
    """거래일마다 '그날 이전 가장 최근 스냅샷'의 구성종목을 붙여 (날짜, 종목) 유니버스를 만든다.

    입력: calendar(거래일), membership(load_membership 결과), exclude(제외 종목 코드)
    출력: DataFrame[trade_date, symbol, snapshot_date]. 제외 종목은 빠진다.
    누수: 각 거래일에는 그날 이하 기준일의 스냅샷만 쓴다(merge_asof backward).
          단, 스냅샷이 월 1회라 실제 편입·편출일보다 늦게 반영될 수 있다(02_universe 노트북 참고).
    """
    snaps = pd.DataFrame({"snapshot_date": sorted(membership["snapshot_date"].unique())})
    days = pd.DataFrame({"trade_date": calendar})
    days = pd.merge_asof(days, snaps, left_on="trade_date", right_on="snapshot_date", direction="backward")
    days = days.dropna(subset=["snapshot_date"])
    uni = days.merge(membership, on="snapshot_date", how="inner")
    uni = uni[~uni["symbol"].isin(exclude)]
    return uni[["trade_date", "symbol", "snapshot_date"]].sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def membership_events(membership: pd.DataFrame) -> pd.DataFrame:
    """연속한 스냅샷을 비교해 편입(IN)·편출(OUT) 이벤트를 만든다.

    입력: membership(load_membership 결과)
    출력: DataFrame[snapshot_date, prev_snapshot, symbol, event('IN'|'OUT')]
          snapshot_date는 변경이 처음 관측된 스냅샷이다(실제 효력일이 아니다).
    누수: 해당 없음(과거 스냅샷끼리 비교)
    """
    snaps = sorted(membership["snapshot_date"].unique())
    sets = {d: set(g["symbol"]) for d, g in membership.groupby("snapshot_date")}
    rows = []
    for prev, cur in zip(snaps, snaps[1:]):
        rows += [(cur, prev, s, "IN") for s in sorted(sets[cur] - sets[prev])]
        rows += [(cur, prev, s, "OUT") for s in sorted(sets[prev] - sets[cur])]
    return pd.DataFrame(rows, columns=["snapshot_date", "prev_snapshot", "symbol", "event"])
