"""고객 보유 종목 불러오기(고객정보 DB, 읽기 전용).

원천: `fintech_realistic_class` 데이터베이스(같은 MySQL 서버, `.env`의 DB 계정으로 조회 권한 부여됨, 2026-09-29)
  - customers: 고객 기본 정보(customer_id, name, ...)
  - customer_stock_positions: 고객 × 계좌 × 종목 보유 현황(quantity, purchase_date, company_name, market)
  - customer_stock_lots: 매수 로트. 수량 합계가 positions와 같다(2026-09-29 확인)라 여기서는 positions만 쓴다
데이터베이스 이름은 `.env`의 CUSTOMER_DB_NAME으로 바꿀 수 있다(기본 fintech_realistic_class).
교육용 데이터라 화면에 고객 이름을 표시해도 된다(사용자 확인, 2026-09-29). 그래도 리포트·로그로 내보내지는 않는다.
"""
import os
import re

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.engine import Engine

from src.db import get_engine, read_sql

load_dotenv()
CUSTOMER_DB = os.getenv("CUSTOMER_DB_NAME") or "fintech_realistic_class"
if not re.fullmatch(r"[A-Za-z0-9_]+", CUSTOMER_DB):          # 식별자는 파라미터로 못 넘기므로 형식을 검사한다
    raise ValueError("CUSTOMER_DB_NAME 형식이 올바르지 않습니다")


def get_customer(customer_id: int, engine: Engine | None = None) -> dict | None:
    """고객 1명의 기본 정보.

    입력: customer_id
    출력: {customer_id, name} 또는 없으면 None
    누수: 해당 없음(모델 입력이 아니다)
    """
    d = read_sql(f"SELECT customer_id, name FROM {CUSTOMER_DB}.customers WHERE customer_id = :c",
                 engine or get_engine(), params={"c": int(customer_id)})
    return None if d.empty else {"customer_id": int(d.customer_id[0]), "name": d.name[0]}


def load_holdings(customer_id: int, as_of: str, engine: Engine | None = None) -> pd.DataFrame:
    """고객의 기준일 보유 종목. 기준일 이후 매수분은 빼고, 여러 계좌의 같은 종목은 수량을 합친다.

    입력: customer_id, as_of(기준일 YYYY-MM-DD)
    출력: DataFrame[symbol(6자리 문자열), name, quantity, market]
    누수: 기준일 이후 매수한 종목은 넣지 않는다(과거 기준일 조회 시 그날 없던 보유가 섞이지 않게)
    """
    d = read_sql(
        f"""SELECT stock_code AS symbol, MAX(company_name) AS name, SUM(quantity) AS quantity, MAX(market) AS market
            FROM {CUSTOMER_DB}.customer_stock_positions
            WHERE customer_id = :c AND purchase_date <= :d AND quantity > 0
            GROUP BY stock_code ORDER BY stock_code""",
        engine or get_engine(), params={"c": int(customer_id), "d": as_of})
    d["symbol"] = d["symbol"].astype(str).str.zfill(6)
    d["quantity"] = pd.to_numeric(d["quantity"])
    return d


AGE_BANDS = [(0, 29, "20대 이하"), (30, 39, "30대"), (40, 49, "40대"), (50, 59, "50대"), (60, 69, "60대"), (70, 200, "70대 이상")]


def customers_by_age(as_of: str, kospi_symbols: list[str], per_band: int = 10, engine: Engine | None = None) -> pd.DataFrame:
    """연령대별 대표 고객 목록(선택용).

    기준일에 KOSPI200 구성종목을 1개 이상 보유한 고객 중, 연령대마다 KOSPI200 보유 종목 수가 많은 순(같으면 전체 보유
    종목 수, 고객 ID 순)으로 per_band명을 뽑는다. 같은 기준일이면 항상 같은 목록이다.
    입력: as_of(기준일), kospi_symbols(기준일 구성종목 코드), per_band(연령대별 인원)
    출력: DataFrame[age_band, customer_id, name, age, n_hold, n_kospi] (연령대 순, 연령대 안에서는 선정 순)
    누수: 해당 없음(모델 입력이 아니다). 나이와 보유는 기준일 기준
    """
    d = read_sql(
        f"""SELECT p.customer_id, c.name, c.birth_date, p.stock_code
            FROM {CUSTOMER_DB}.customer_stock_positions p JOIN {CUSTOMER_DB}.customers c USING (customer_id)
            WHERE p.purchase_date <= :d AND p.quantity > 0""",
        engine or get_engine(), params={"d": as_of})
    d["k"] = d["stock_code"].astype(str).str.zfill(6).isin(set(kospi_symbols))
    g = d.groupby(["customer_id", "name", "birth_date"], as_index=False).agg(
        n_hold=("stock_code", "nunique"), n_kospi=("k", "sum"))
    g = g[g["n_kospi"] > 0].copy()
    ref = pd.Timestamp(as_of)
    bd = pd.to_datetime(g["birth_date"])
    had_birthday = (bd.dt.month < ref.month) | ((bd.dt.month == ref.month) & (bd.dt.day <= ref.day))
    g["age"] = ref.year - bd.dt.year - (~had_birthday).astype(int)            # 기준일 만 나이
    labels = pd.Series(pd.NA, index=g.index, dtype=object)
    for lo, hi, lab in AGE_BANDS:
        labels[(g["age"] >= lo) & (g["age"] <= hi)] = lab
    g["age_band"] = labels
    g = g.sort_values(["n_kospi", "n_hold", "customer_id"], ascending=[False, False, True])
    out = g.groupby("age_band", sort=False).head(per_band)
    order = {lab: i for i, (_, _, lab) in enumerate(AGE_BANDS)}
    out = out.assign(_o=out["age_band"].map(order)).sort_values(["_o", "n_kospi", "n_hold", "customer_id"],
                                                                ascending=[True, False, False, True])
    return out[["age_band", "customer_id", "name", "age", "n_hold", "n_kospi"]].reset_index(drop=True)
