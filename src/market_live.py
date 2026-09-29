"""토스증권 Open API로 실시간 시장 정보(KOSPI 지수, 원/달러 환율)를 가져온다.

- 명세: https://openapi.tossinvest.com/openapi-docs/latest/openapi.json (v1.2.19 기준 작성, 2026-09-29)
- 인증: OAuth 2.0 Client Credentials. `.env`의 TOSS_CLIENT_ID, TOSS_CLIENT_SECRET을 쓴다(값은 출력하지 않는다).
- 호출 IP가 토스증권 개발자센터의 허용 IP 목록에 등록돼 있어야 한다.
- 토큰은 client당 1개만 유효하고, 재발급하면 이전 토큰이 즉시 무효가 된다. 그래서 TossClient 하나를 프로세스 안에서 공유한다.
- 이 모듈의 값은 화면 표시용이다. 모델 변수로 쓰지 않는다(누수 해당 없음).
"""
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()
BASE_URL = (os.getenv("TOSS_BASE_URL") or "https://openapi.tossinvest.com").rstrip("/")
KST = timezone(timedelta(hours=9))
TIMEOUT = 10


class TossAPIError(RuntimeError):
    """토스증권 API 오류(인증 실패, 허용 IP 미등록, 한도 초과 등)."""


def credentials_available() -> bool:
    """TOSS_CLIENT_ID, TOSS_CLIENT_SECRET이 설정돼 있는지(값은 보지 않는다)."""
    load_dotenv()
    return bool(os.getenv("TOSS_CLIENT_ID")) and bool(os.getenv("TOSS_CLIENT_SECRET"))


TOKEN_FILE = Path.home() / ".cache" / "stock_risk" / "toss_token.json"   # 저장소 밖, 권한 600


class TossClient:
    """토큰을 파일로 공유하는 최소 REST 클라이언트.

    토스 토큰은 client당 1개만 유효하고 재발급하면 이전 토큰이 즉시 무효가 된다. 대시보드와 수집기가 서로의 토큰을
    무효화하지 않도록 토큰을 TOKEN_FILE에 두고 공유한다. 401이면 먼저 파일을 다시 읽고(다른 프로세스가 갱신했을 수 있음),
    그래도 같은 토큰이면 새로 받는다(최대 3번). 429(한도 초과)는 잠시 쉬고 다시 시도한다.
    min_interval: 요청 사이 최소 간격(초). 대량 수집 때 한도(초당 5 ~ 20회)를 넘지 않게 한다.
    """

    def __init__(self, session: requests.Session | None = None, min_interval: float = 0.0,
                 token_file: Path | None = TOKEN_FILE):
        load_dotenv()
        self._id = os.getenv("TOSS_CLIENT_ID")
        self._secret = os.getenv("TOSS_CLIENT_SECRET")
        if not (self._id and self._secret):
            raise TossAPIError("TOSS_CLIENT_ID / TOSS_CLIENT_SECRET이 .env에 없습니다")
        self._s = session or requests.Session()
        self._token: str | None = None
        self._exp = 0.0
        self._lock = threading.Lock()
        self._file = token_file
        self._min_interval = min_interval
        self._last = 0.0

    # ---- 토큰 파일
    def _read_file(self) -> bool:
        if not self._file or not self._file.exists():
            return False
        try:
            j = json.loads(self._file.read_text())
        except (OSError, ValueError):
            return False
        if j.get("client_id") != self._id or time.time() >= j.get("exp", 0):
            return False
        self._token, self._exp = j["token"], j["exp"]
        return True

    def _write_file(self) -> None:
        if not self._file:
            return
        self._file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"client_id": self._id, "token": self._token, "exp": self._exp}))
        os.chmod(tmp, 0o600)
        tmp.replace(self._file)

    def _issue_token(self) -> None:
        r = self._s.post(f"{BASE_URL}/oauth2/token", timeout=TIMEOUT,
                         data={"grant_type": "client_credentials", "client_id": self._id, "client_secret": self._secret})
        if r.status_code != 200:
            raise TossAPIError(f"토큰 발급 실패 (HTTP {r.status_code}): {_error_code(r)}")
        j = r.json()
        self._token = j["access_token"]
        self._exp = time.time() + int(j.get("expires_in", 86400)) - 300
        self._write_file()

    def _ensure_token(self, stale: str | None = None) -> str:
        """유효한 토큰을 돌려준다. stale이 주어지면 그 토큰은 무효로 보고 파일 → 재발급 순으로 바꾼다."""
        with self._lock:
            if self._token and self._token != stale and time.time() < self._exp:
                return self._token
            if self._read_file() and self._token != stale:
                return self._token
            self._issue_token()
            return self._token

    def get(self, path: str, params: dict | None = None) -> dict | list:
        """GET 요청 후 `result`를 돌려준다."""
        token, auth_retries = self._ensure_token(), 0
        for attempt in range(6):
            if self._min_interval:
                wait = self._last + self._min_interval - time.time()
                if wait > 0:
                    time.sleep(wait)
                self._last = time.time()
            r = self._s.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT,
                            headers={"Authorization": f"Bearer {token}"})
            if r.status_code == 401 and auth_retries < 3:        # 다른 프로세스가 막 토큰을 바꾼 경우: 잠깐 쉬고 파일 → 재발급
                time.sleep(0.5 * auth_retries)
                token, auth_retries = self._ensure_token(stale=token), auth_retries + 1
                continue
            if r.status_code == 429 and attempt < 5:
                time.sleep(float(r.headers.get("Retry-After") or 1.0))
                continue
            if r.status_code != 200:
                hint = " (허용 IP 미등록일 수 있습니다)" if r.status_code == 403 else ""
                raise TossAPIError(f"{path} 실패 (HTTP {r.status_code}): {_error_code(r)}{hint}")
            return r.json()["result"]
        raise TossAPIError(f"{path} 실패: 재시도 한도 초과")


def _error_code(r: requests.Response) -> str:
    try:
        j = r.json()
        return str(j.get("error", {}).get("code") or j.get("error") or j)[:200]
    except ValueError:
        return r.text[:200]


def _ts(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s).astimezone(KST) if s else None


@dataclass
class IndexQuote:
    symbol: str
    price: float                 # 현재가(장 마감 뒤에는 종가)
    prev_close: float | None     # 직전 거래일 종가
    change: float | None         # 전일 대비(포인트)
    change_pct: float | None     # 전일 대비(%)
    as_of: datetime | None       # 데이터 시각(KST). API가 null을 주면 None


@dataclass
class FxQuote:
    mid: float                   # 매매기준율(1 USD = ? KRW)
    prev_mid: float | None       # 전일 같은 시각 매매기준율
    change: float | None
    change_pct: float | None
    as_of: datetime              # 유효 시작 시각(KST)


def get_index_quote(client: TossClient, symbol: str = "KOSPI") -> IndexQuote:
    """지수 현재가와 전일 대비 등락.

    현재가 API는 lastPrice만 준다. 전일 종가는 일봉 2개에서 현재가 날짜보다 앞선 봉의 종가로 구한다.
    입력: client, symbol(KOSPI 또는 KOSDAQ)
    출력: IndexQuote
    """
    p = client.get("/api/v1/market-indicators/prices", {"symbols": symbol})[0]
    price, as_of = float(p["lastPrice"]), _ts(p.get("timestamp"))
    candles = client.get(f"/api/v1/market-indicators/{symbol}/candles", {"interval": "1d", "count": 3})["candles"]
    # 현재가 timestamp가 null로 오는 경우가 있다(2026-09-29 확인). 그때는 가장 최근 일봉 날짜를 현재 거래일로 본다.
    # 장 시작 전에는 가장 최근 일봉이 직전 거래일이므로, 등락은 직전 거래일의 등락이 된다.
    session = as_of.date() if as_of else _ts(candles[0]["timestamp"]).date()
    prev = next((c for c in candles if _ts(c["timestamp"]).date() < session), None)   # 최신순 정렬
    prev_close = float(prev["closePrice"]) if prev else None
    chg = price - prev_close if prev_close else None
    return IndexQuote(symbol, price, prev_close, chg, chg / prev_close * 100 if prev_close else None, as_of)


def get_index_intraday(client: TossClient, symbol: str = "KOSPI", pages: int = 2) -> list[tuple[datetime, float]]:
    """지수 1분봉 종가(가장 최근 거래일 하루치, 시간 오름차순). 한 번에 200봉이라 2쪽(400분)이면 정규장 전체를 덮는다."""
    out, before = [], None
    for _ in range(pages):
        params = {"interval": "1m", "count": 200}
        if before:
            params["before"] = before
        page = client.get(f"/api/v1/market-indicators/{symbol}/candles", params)
        out += [(_ts(c["timestamp"]), float(c["closePrice"])) for c in page["candles"]]
        before = page.get("nextBefore")
        if not before:
            break
    if not out:
        return []
    last_day = max(t for t, _ in out).date()
    return sorted((t, v) for t, v in out if t.date() == last_day)


def get_usdkrw(client: TossClient) -> FxQuote:
    """원/달러 매매기준율과 전일 같은 시각 대비 변화(환율은 1분마다 갱신되는 참고용 표시 환율)."""
    q = {"baseCurrency": "USD", "quoteCurrency": "KRW"}
    now = client.get("/api/v1/exchange-rate", q)
    as_of = _ts(now["validFrom"])
    prev_t = as_of - timedelta(days=1)
    while prev_t.weekday() >= 5:                     # 주말이면 직전 금요일
        prev_t -= timedelta(days=1)
    try:
        prev = client.get("/api/v1/exchange-rate", {**q, "dateTime": prev_t.isoformat()})
        prev_mid = float(prev["midRate"])
    except TossAPIError:
        prev_mid = None
    mid = float(now["midRate"])
    chg = mid - prev_mid if prev_mid else None
    return FxQuote(mid, prev_mid, chg, chg / prev_mid * 100 if prev_mid else None, as_of)
