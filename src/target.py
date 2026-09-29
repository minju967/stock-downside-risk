"""타깃(향후 5거래일 최대 낙폭) 계산 함수.

주의: 이 모듈의 함수는 모두 **미래 가격**을 쓴다. 결과는 타깃으로만 쓰고 변수(feature)에 넣으면 안 된다.
"""
import numpy as np
import pandas as pd

H = 5  # 예측 구간(거래일)


def flag_low_spikes(c: pd.DataFrame, depth: float = 0.15, close_band: float = 0.05) -> pd.Series:
    """비정상 저가(스파이크) 행을 표시한다(Step 2 정의).

    정의: 거래량 > 0, |종가/전일종가 - 1| < close_band 이고 저가 < (1 - depth) * min(전일종가, 종가)
    입력: c(symbol, trade_date 정렬, open/high/low/close_price, volume 포함), depth, close_band
    출력: bool Series(c와 같은 인덱스)
    누수: t일 값과 t-1일 종가만 쓴다(누수 없음). 단, 이 플래그로 보정한 가격은 타깃 계산에만 쓴다.
    """
    prev = c.groupby("symbol")["close_price"].shift()
    r_close = c["close_price"] / prev - 1
    ref = np.minimum(prev, c["close_price"])
    return (c["volume"] > 0) & (r_close.abs() < close_band) & (c["low_price"] < (1 - depth) * ref)


def clean_low(c: pd.DataFrame, spike: pd.Series) -> pd.Series:
    """스파이크 행의 저가를 보정한다.

    규칙: 스파이크 행에서 시가 = 저가이면 시가도 스파이크로 보고 저가를 종가로 바꾼다.
          그렇지 않으면 저가를 min(시가, 종가)로 바꾼다. 실제 장중 저점보다 높게 잡힐 수 있다(보수적이지 않음).
    입력: c(open/low/close_price 포함), spike(flag_low_spikes 결과)
    출력: 보정 저가 Series
    누수: 해당 없음(같은 날 값만 사용). 타깃 계산 전용.
    """
    open_ok = c["open_price"] > c["low_price"]
    repl = np.where(open_ok, np.minimum(c["open_price"], c["close_price"]), c["close_price"])
    return pd.Series(np.where(spike, repl, c["low_price"]), index=c.index)


def _fwd_window(s: pd.Series, h: int = H) -> pd.DataFrame:
    """t+1..t+h 값을 열로 펼친다(미래 방향)."""
    return pd.concat([s.shift(-i) for i in range(1, h + 1)], axis=1)


def _halted(g: pd.DataFrame) -> pd.Series:
    """거래정지 여부: `halted` 컬럼(KRX 일봉)이 있으면 그것을, 없으면 거래량 0을 쓴다."""
    return (g["halted"] == 1).astype(float) if "halted" in g.columns else (g["volume"] == 0).astype(float)


def compute_targets(c: pd.DataFrame, h: int = H) -> pd.DataFrame:
    """네 가지 MDD_h 정의와 보조 플래그를 계산한다.

    정의(모두 0 이하, 작을수록 위험):
      - mdd_low:       min(Low[t+1..t+h]) / Close[t] - 1                  (기본 정의)
      - mdd_low_clean: min(보정 Low[t+1..t+h]) / Close[t] - 1             (스파이크 보정)
      - mdd_close:     min(Close[t+1..t+h]) / Close[t] - 1                (종가 기준)
      - mdd_path:      Close[t..t+h] 경로의 최대 고점 대비 낙폭 (peak-to-trough, 종가 기준)
    보조 컬럼:
      - spike_in_window: t+1..t+h에 저가 스파이크 행이 있음
      - halt_in_window:  t+1..t+h에 거래정지 행이 있음(`halted` 컬럼, 없으면 거래량 0)
      - n_fwd:           t+1..t+h 중 일봉이 있는 날 수(h 미만이면 타깃은 NaN)
    입력: c(종목별 연속 거래일 일봉. symbol, trade_date, OHLC, volume, 선택: halted). 기간 끝 h일은 타깃이 NaN이 된다.
    출력: DataFrame[symbol, trade_date, mdd_low, mdd_low_clean, mdd_close, mdd_path, spike_in_window, halt_in_window, n_fwd]
    누수: **미래 정보를 쓴다. 타깃 전용.** 변수 계산에 쓰면 안 된다.
    """
    c = c.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    spike = flag_low_spikes(c)
    low_c = clean_low(c, spike)
    out = c[["symbol", "trade_date"]].copy()
    parts = []
    for _, g in c.assign(_spike=spike, _low_c=low_c).groupby("symbol", sort=False):
        close = g["close_price"]
        lw = _fwd_window(g["low_price"], h)
        lwc = _fwd_window(g["_low_c"], h)
        cw = _fwd_window(close, h)
        n_fwd = cw.notna().sum(axis=1)
        full = n_fwd == h
        # peak-to-trough: t..t+h 종가 경로
        path = pd.concat([close, cw], axis=1).to_numpy()
        peak = np.fmax.accumulate(path, axis=1)
        dd = np.nanmin(path / peak - 1, axis=1)
        parts.append(pd.DataFrame({
            "mdd_low": np.where(full, lw.min(axis=1) / close - 1, np.nan),
            "mdd_low_clean": np.where(full, lwc.min(axis=1) / close - 1, np.nan),
            "mdd_close": np.where(full, cw.min(axis=1) / close - 1, np.nan),
            "mdd_path": np.where(full, dd, np.nan),
            "spike_in_window": _fwd_window(g["_spike"].astype(float), h).max(axis=1).fillna(0).astype(bool),
            "halt_in_window": _fwd_window(_halted(g), h).max(axis=1).fillna(0).astype(bool),
            "n_fwd": n_fwd,
        }, index=g.index))
    return pd.concat([out, pd.concat(parts).sort_index()], axis=1)
