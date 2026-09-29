"""DB 연결 유틸리티 (읽기 전용).

접속 정보는 프로젝트 루트의 `.env`에서만 읽고, 값은 어디에도 출력하지 않는다.
"""
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from dotenv import dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

ROOT = Path(__file__).resolve().parents[1]

_READ_ONLY_PREFIXES = ("select", "show", "describe", "desc", "explain", "with")


def get_engine() -> Engine:
    """`.env`로 MySQL 엔진을 만든다.

    입력: 없음(`.env`의 DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)
    출력: SQLAlchemy Engine
    누수: 해당 없음
    """
    cfg = dotenv_values(ROOT / ".env")
    url = (
        f"mysql+pymysql://{quote_plus(cfg['DB_USER'])}:{quote_plus(cfg['DB_PASSWORD'])}"
        f"@{cfg['DB_HOST']}:{cfg.get('DB_PORT') or 3306}/{cfg['DB_NAME']}?charset=utf8mb4"
    )
    # hide_parameters: 오류 메시지에 파라미터 값이 찍히지 않게 한다
    return create_engine(url, pool_pre_ping=True, hide_parameters=True)


def read_sql(sql: str, engine: Engine | None = None, params: dict | None = None) -> pd.DataFrame:
    """읽기 전용 쿼리를 실행해 DataFrame으로 반환한다.

    입력: SQL 문자열(SELECT/SHOW/DESCRIBE/EXPLAIN만 허용), 선택적 엔진·파라미터
    출력: pandas DataFrame
    누수: 해당 없음(조회 기간은 호출자가 통제)
    """
    if not sql.lstrip().lower().startswith(_READ_ONLY_PREFIXES):
        raise ValueError("읽기 전용 쿼리만 허용된다.")
    engine = engine or get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)
