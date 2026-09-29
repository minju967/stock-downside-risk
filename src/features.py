"""후보 변수(feature) 생성 함수.

공통 규칙(docs/eda_plan.md 2절):
- 모든 변수는 t일까지의 값만 쓴다(롤링은 과거 방향). t+1일 장 시작 전 예측이므로, t일 장 마감 후 공표된 수급 데이터도 lag 없이 t일 값으로 쓴다.
- 롤링 윈도우는 최대 60거래일, t일 이전에 필요한 거래일은 61일 이하.
- 가격 수준(원)은 소급 조정 영향을 받으므로 변수로 쓰지 않고 비율만 쓴다.
- 고가·저가는 비정상 꼬리(스파이크)를 같은 날 정보로 보정한 값을 쓴다(`prepare_prices`).
"""
import numpy as np
import pandas as pd

from src.target import clean_low, flag_low_spikes

# 변수 명세: 이름 -> (계열, 계산식, t일 이전에 필요한 거래일 수, 비고)
SPEC: dict[str, tuple[str, str, int, str]] = {}


def _spec(name, family, formula, lookback, note=""):
    SPEC[name] = (family, formula, lookback, note)


# ---------------------------------------------------------------- 가격 준비
def flag_high_spikes(c: pd.DataFrame, depth: float = 0.15, close_band: float = 0.05) -> pd.Series:
    """비정상 고가(스파이크) 행을 표시한다. 저가 스파이크의 대칭 정의.

    정의: 거래량 > 0, |종가/전일종가 - 1| < close_band 이고 고가 > (1 + depth) * max(전일종가, 종가)
    입력: c(symbol, trade_date 정렬된 일봉)
    출력: bool Series
    누수: t일 값과 t-1일 종가만 사용(누수 없음)
    """
    prev = c.groupby("symbol")["close_price"].shift()
    r_close = c["close_price"] / prev - 1
    ref = np.maximum(prev, c["close_price"])
    return (c["volume"] > 0) & (r_close.abs() < close_band) & (c["high_price"] > (1 + depth) * ref)


def prepare_prices(c: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    """변수 계산용 가격표를 만든다: 정렬, 전일 종가, (선택) 보정 고가·저가, 거래대금, 정지 여부, 원주가·상장주식수.

    입력: c(krx_daily_candles 또는 stock_daily_candles 형식), clean(스파이크 보정 여부. KRX 정규장 시세면 False)
    출력: DataFrame[symbol, trade_date, open, high, low, close, volume, prev_close, value, halted,
                    raw_close, raw_volume, listed_shares, lo_spike, hi_spike] (원주가·상장주식수는 KRX 일봉에만 있음)
    누수: 같은 날과 전일 값만 사용(누수 없음)
    """
    c = c.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    p = pd.DataFrame({"symbol": c.symbol, "trade_date": c.trade_date, "open": c.open_price,
                      "close": c.close_price, "volume": c.volume.astype(float)})
    p["prev_close"] = p.groupby("symbol")["close"].shift()
    lo_sp = flag_low_spikes(c) if clean else pd.Series(False, index=c.index)
    hi_sp = flag_high_spikes(c) if clean else pd.Series(False, index=c.index)
    p["low"] = clean_low(c, lo_sp) if clean else c.low_price
    open_ok = c.open_price < c.high_price
    hi_repl = np.where(open_ok, np.maximum(c.open_price, c.close_price), c.close_price)
    p["high"] = np.where(hi_sp, hi_repl, c.high_price)
    # 거래대금: KRX 일봉의 실제 거래대금, 없으면 종가×거래량 근사
    p["value"] = c["trading_value"].astype(float) if "trading_value" in c else p["close"] * p["volume"]
    p["halted"] = (c["halted"] == 1) if "halted" in c else (c["volume"] == 0)
    for col in ("raw_close", "raw_volume", "listed_shares"):
        p[col] = c[col].astype(float) if col in c else np.nan
    p["lo_spike"], p["hi_spike"] = lo_sp.values, hi_sp.values
    # 거래정지일(거래량 0)은 수익률 0으로 계산된다. 별도 플래그로 남긴다
    return p


def _roll(g, col, n, fn, **kw):
    return g[col].transform(lambda s: getattr(s.rolling(n, **kw), fn)())


# ---------------------------------------------------------------- 종목 가격 변수
def price_features(p: pd.DataFrame) -> pd.DataFrame:
    """수익률·모멘텀, 변동성, 하방 특성, 가격 위치, 거래·유동성, 캔들 변수를 만든다.

    입력: p(prepare_prices 결과)
    출력: DataFrame[symbol, trade_date, <변수들>]
    누수: 모든 롤링은 t일 포함 과거 방향만 쓴다(shift 음수 없음)
    """
    f = p[["symbol", "trade_date"]].copy()
    g = p.groupby("symbol")
    r = p["close"] / p["prev_close"] - 1
    lr = np.log1p(r)
    p = p.assign(r=r, lr=lr)
    g = p.groupby("symbol")

    # 수익률·모멘텀
    for n in (1, 5, 20, 60):
        f[f"ret_{n}"] = p["close"] / g["close"].shift(n) - 1
        _spec(f"ret_{n}", "수익률·모멘텀", f"Close[t] / Close[t-{n}] - 1", n)
    for n in (5, 20, 60):
        f[f"ma_gap_{n}"] = p["close"] / _roll(g, "close", n, "mean") - 1
        _spec(f"ma_gap_{n}", "수익률·모멘텀", f"Close[t] / mean(Close[t-{n-1}..t]) - 1", n - 1)
    f["ret_20_skip5"] = g["close"].shift(5) / g["close"].shift(20) - 1
    _spec("ret_20_skip5", "수익률·모멘텀", "Close[t-5] / Close[t-20] - 1 (최근 5일 제외 모멘텀)", 20)

    # 변동성 (일간 로그수익률 표준편차, 연율화하지 않음)
    for n in (5, 20, 60):
        f[f"vol_{n}"] = _roll(g, "lr", n, "std")
        _spec(f"vol_{n}", "변동성", f"std(log수익률[t-{n-1}..t])", n)
    hl = np.log(p["high"] / p["low"]) ** 2
    co = np.log(p["close"] / p["open"]) ** 2
    p = p.assign(pk=hl / (4 * np.log(2)), gk=0.5 * hl - (2 * np.log(2) - 1) * co)
    g = p.groupby("symbol")
    for n in (5, 20):
        f[f"parkinson_{n}"] = np.sqrt(_roll(g, "pk", n, "mean"))
        _spec(f"parkinson_{n}", "변동성", f"sqrt(mean(ln(H/L)^2 / 4ln2)), {n}일", n - 1, "고가·저가 보정값 사용")
        f[f"gk_{n}"] = np.sqrt(_roll(g, "gk", n, "mean").clip(lower=0))
        _spec(f"gk_{n}", "변동성", f"Garman-Klass: sqrt(mean(0.5 ln(H/L)^2 - (2ln2-1) ln(C/O)^2)), {n}일", n - 1, "고가·저가 보정값 사용")
    f["vol_ratio_5_20"] = f["vol_5"] / f["vol_20"]
    _spec("vol_ratio_5_20", "변동성", "vol_5 / vol_20 (변동성 변화)", 20)
    f["vol_ratio_20_60"] = f["vol_20"] / f["vol_60"]
    _spec("vol_ratio_20_60", "변동성", "vol_20 / vol_60 (변동성 변화)", 60)
    f["vol_chg_20"] = f["vol_20"] / f.groupby(p["symbol"])["vol_20"].shift(20) - 1
    _spec("vol_chg_20", "변동성", "vol_20[t] / vol_20[t-20] - 1", 40)

    # 하방 특성
    neg = p["lr"].clip(upper=0)
    p = p.assign(neg2=neg ** 2)
    g = p.groupby("symbol")
    for n in (20, 60):
        f[f"semidev_{n}"] = np.sqrt(_roll(g, "neg2", n, "mean"))
        _spec(f"semidev_{n}", "하방 특성", f"sqrt(mean(min(log수익률,0)^2)), {n}일", n)
        f[f"downside_ratio_{n}"] = f[f"semidev_{n}"] / f[f"vol_{n}"]
        _spec(f"downside_ratio_{n}", "하방 특성", f"semidev_{n} / vol_{n}", n)
        f[f"skew_{n}"] = _roll(g, "lr", n, "skew")
        _spec(f"skew_{n}", "하방 특성", f"왜도(log수익률, {n}일)", n)
        f[f"kurt_{n}"] = _roll(g, "lr", n, "kurt")
        _spec(f"kurt_{n}", "하방 특성", f"초과 첨도(log수익률, {n}일)", n)
        f[f"min_ret_{n}"] = _roll(g, "r", n, "min")
        _spec(f"min_ret_{n}", "하방 특성", f"min(일간 수익률[t-{n-1}..t])", n)
        f[f"n_down3_{n}"] = g["r"].transform(lambda s: (s <= -0.03).astype(float).rolling(n).mean())
        _spec(f"n_down3_{n}", "하방 특성", f"최근 {n}일 중 일간 수익률 ≤ -3%인 날의 비율", n)
    f["past_mdd_5"] = g["low"].transform(lambda s: s.rolling(5).min()) / g["close"].shift(5) - 1
    _spec("past_mdd_5", "하방 특성", "min(Low[t-4..t]) / Close[t-5] - 1 (= t-5일의 타깃)", 5, "저가 보정값 사용")
    for n in (20, 60):
        f[f"past_mdd_{n}"] = g["close"].transform(lambda s: _roll_mdd(s, n))
        _spec(f"past_mdd_{n}", "하방 특성", f"Close[t-{n-1}..t] 경로의 고점 대비 최대 낙폭", n - 1)
    p = p.assign(intra_drop=p["low"] / p["prev_close"] - 1)
    g = p.groupby("symbol")
    f["min_intra_drop_20"] = _roll(g, "intra_drop", 20, "min")
    _spec("min_intra_drop_20", "하방 특성", "min(Low / 전일 Close - 1), 20일", 20, "저가 보정값 사용")

    # 가격 위치
    hi60 = _roll(g, "high", 60, "max")
    lo60 = _roll(g, "low", 60, "min")
    f["pos_60"] = (p["close"] - lo60) / (hi60 - lo60)
    _spec("pos_60", "가격 위치", "(Close - min Low60) / (max High60 - min Low60)", 59, "52주 대신 60일")
    f["dist_high_60"] = p["close"] / hi60 - 1
    _spec("dist_high_60", "가격 위치", "Close / max(High[t-59..t]) - 1", 59)
    f["dist_low_60"] = p["close"] / lo60 - 1
    _spec("dist_low_60", "가격 위치", "Close / min(Low[t-59..t]) - 1", 59)
    f["days_since_high_60"] = g["close"].transform(lambda s: s.rolling(60).apply(lambda a: 59 - np.argmax(a), raw=True))
    _spec("days_since_high_60", "가격 위치", "60일 종가 최고점 이후 경과 거래일(0~59)", 59)
    f["pos_20"] = (p["close"] - _roll(g, "low", 20, "min")) / (_roll(g, "high", 20, "max") - _roll(g, "low", 20, "min"))
    _spec("pos_20", "가격 위치", "(Close - min Low20) / (max High20 - min Low20)", 19)

    # 거래·유동성
    lv = np.log(p["value"].where(p["value"] > 0))
    p = p.assign(lv=lv, amihud=(p["r"].abs() / p["value"].where(p["value"] > 0)) * 1e9)
    g = p.groupby("symbol")
    f["log_value_20"] = _roll(g, "lv", 20, "mean")
    _spec("log_value_20", "거래·유동성", "mean(ln(거래대금)), 20일 (규모·유동성)", 19, "KRX 실제 거래대금")
    # 시가총액·회전율: 그날의 원주가와 상장주식수(point-in-time)로 계산
    f["log_mcap"] = np.log(p["raw_close"] * p["listed_shares"])
    _spec("log_mcap", "거래·유동성", "ln(원주가 종가 × 그날 상장주식수)", 0, "KRX 일봉의 listed_shares(시점별 값)")
    p = p.assign(turn=p["raw_volume"] / p["listed_shares"])
    g = p.groupby("symbol")
    f["turnover_20"] = _roll(g, "turn", 20, "mean")
    _spec("turnover_20", "거래·유동성", "mean(원 거래량 / 상장주식수), 20일", 19)
    f["turnover_ratio_5_60"] = _roll(g, "turn", 5, "mean") / _roll(g, "turn", 60, "mean")
    _spec("turnover_ratio_5_60", "거래·유동성", "mean(회전율, 5일) / mean(회전율, 60일)", 59)
    for a, b in ((1, 20), (5, 20), (20, 60)):
        f[f"value_ratio_{a}_{b}"] = _roll(g, "value", a, "mean") / _roll(g, "value", b, "mean")
        _spec(f"value_ratio_{a}_{b}", "거래·유동성", f"mean(거래대금, {a}일) / mean(거래대금, {b}일)", b - 1)
    f["volume_z_20"] = (p["volume"] - _roll(g, "volume", 20, "mean")) / _roll(g, "volume", 20, "std")
    _spec("volume_z_20", "거래·유동성", "(Volume - mean20) / std20", 19)
    f["amihud_20"] = _roll(g, "amihud", 20, "mean")
    _spec("amihud_20", "거래·유동성", "mean(|수익률| / 거래대금 × 1e9), 20일", 20)
    f["zero_vol_20"] = g["halted"].transform(lambda s: s.astype(float).rolling(20).mean())
    _spec("zero_vol_20", "거래·유동성", "최근 20일 중 거래정지일의 비율", 19)

    # 캔들
    rng = (p["high"] - p["low"]).replace(0, np.nan)
    f["range_1"] = (p["high"] - p["low"]) / p["close"]
    _spec("range_1", "캔들", "(High - Low) / Close", 0, "고가·저가 보정값 사용")
    p = p.assign(range_1=f["range_1"])
    g = p.groupby("symbol")
    for n in (5, 20):
        f[f"range_{n}"] = _roll(g, "range_1", n, "mean")
        _spec(f"range_{n}", "캔들", f"mean((High - Low) / Close), {n}일", n - 1)
    f["gap_1"] = p["open"] / p["prev_close"] - 1
    _spec("gap_1", "캔들", "Open / 전일 Close - 1", 1)
    f["upper_shadow_1"] = (p["high"] - np.maximum(p["open"], p["close"])) / rng
    _spec("upper_shadow_1", "캔들", "(High - max(O,C)) / (High - Low)", 0)
    f["lower_shadow_1"] = (np.minimum(p["open"], p["close"]) - p["low"]) / rng
    _spec("lower_shadow_1", "캔들", "(min(O,C) - Low) / (High - Low)", 0)
    f["body_1"] = (p["close"] - p["open"]) / rng
    _spec("body_1", "캔들", "(Close - Open) / (High - Low)", 0)
    f["close_loc_1"] = (p["close"] - p["low"]) / rng
    _spec("close_loc_1", "캔들", "(Close - Low) / (High - Low) (일중 종가 위치)", 0)
    p = p.assign(ls=f["lower_shadow_1"], us=f["upper_shadow_1"])
    g = p.groupby("symbol")
    f["lower_shadow_5"] = _roll(g, "ls", 5, "mean")
    _spec("lower_shadow_5", "캔들", "mean(lower_shadow_1), 5일", 4)
    f["upper_shadow_5"] = _roll(g, "us", 5, "mean")
    _spec("upper_shadow_5", "캔들", "mean(upper_shadow_1), 5일", 4)
    return f


def _roll_mdd(s: pd.Series, n: int) -> pd.Series:
    """과거 n일 종가 경로의 고점 대비 최대 낙폭(t 포함, 과거 방향)."""
    a = s.to_numpy(dtype=float)
    out = np.full(len(a), np.nan)
    for i in range(n - 1, len(a)):
        w = a[i - n + 1: i + 1]
        out[i] = np.min(w / np.maximum.accumulate(w) - 1)
    return pd.Series(out, index=s.index)


# ---------------------------------------------------------------- 시장·상대 변수
def market_features(p: pd.DataFrame, idx: pd.DataFrame, universe: pd.DataFrame,
                    mkt_flow: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시장 공통 변수(날짜 단위)와 시장 대비 종목 변수(종목·날짜 단위)를 만든다.

    입력: p(prepare_prices), idx(market_indicator_daily_candles), universe(trade_date, symbol: 그날 구성종목),
          mkt_flow(market_indicator_investor_trading)
    출력: (mkt: 날짜 단위 DataFrame[trade_date, ...], rel: DataFrame[symbol, trade_date, ...])
    누수: 모든 값은 t일까지. 폭(breadth)은 t일 구성종목의 t일 이전 값으로만 계산
    """
    k = idx[idx.symbol == "KOSPI"].set_index("trade_date").sort_index()
    m = pd.DataFrame(index=k.index)
    kr = k["close_price"].pct_change()
    klr = np.log1p(kr)
    for n in (1, 5, 20, 60):
        m[f"mkt_ret_{n}"] = k["close_price"] / k["close_price"].shift(n) - 1
        _spec(f"mkt_ret_{n}", "시장", f"KOSPI Close[t] / Close[t-{n}] - 1", n, "날짜 공통")
    for n in (5, 20, 60):
        m[f"mkt_vol_{n}"] = klr.rolling(n).std()
        _spec(f"mkt_vol_{n}", "시장", f"std(KOSPI log수익률), {n}일", n, "날짜 공통")
    m["mkt_vol_ratio_5_60"] = m["mkt_vol_5"] / m["mkt_vol_60"]
    _spec("mkt_vol_ratio_5_60", "시장", "mkt_vol_5 / mkt_vol_60", 60, "날짜 공통")
    m["mkt_range_1"] = (k["high_price"] - k["low_price"]) / k["close_price"]
    _spec("mkt_range_1", "시장", "KOSPI (High - Low) / Close", 0, "날짜 공통")
    m["mkt_range_5"] = m["mkt_range_1"].rolling(5).mean()
    _spec("mkt_range_5", "시장", "mean(mkt_range_1), 5일", 4, "날짜 공통")
    m["mkt_mdd_20"] = _roll_mdd(k["close_price"], 20)
    _spec("mkt_mdd_20", "시장", "KOSPI 20일 종가 경로 최대 낙폭", 19, "날짜 공통")
    m["mkt_dist_high_60"] = k["close_price"] / k["high_price"].rolling(60).max() - 1
    _spec("mkt_dist_high_60", "시장", "KOSPI Close / max(High, 60일) - 1", 59, "날짜 공통")
    m["mkt_ma_gap_20"] = k["close_price"] / k["close_price"].rolling(20).mean() - 1
    _spec("mkt_ma_gap_20", "시장", "KOSPI Close / MA20 - 1", 19, "날짜 공통")
    m["mkt_gap_1"] = k["open_price"] / k["close_price"].shift() - 1
    _spec("mkt_gap_1", "시장", "KOSPI Open / 전일 Close - 1", 1, "날짜 공통")
    kq = idx[idx.symbol == "KOSDAQ"].set_index("trade_date").sort_index()["close_price"]
    m["kosdaq_ret_5"] = kq / kq.shift(5) - 1
    _spec("kosdaq_ret_5", "시장", "KOSDAQ 5일 수익률", 5, "날짜 공통")
    # 금리
    b3 = idx[idx.symbol == "KR_BOND_3Y"].set_index("trade_date")["close_price"]
    b10 = idx[idx.symbol == "KR_BOND_10Y"].set_index("trade_date")["close_price"]
    m["bond3y_chg_20"] = (b3 - b3.shift(20)).reindex(m.index)
    _spec("bond3y_chg_20", "시장", "국고채 3년 금리 20일 변화(%p)", 20, "날짜 공통")
    m["term_spread"] = (b10 - b3).reindex(m.index)
    _spec("term_spread", "시장", "국고채 10년 - 3년 금리(%p)", 0, "날짜 공통")
    # 시장 수급 (KOSPI, 금액)
    fl = mkt_flow[mkt_flow.symbol == "KOSPI"].set_index("trade_date").sort_index()
    tot = fl[[c for c in fl.columns if c.endswith("_buy_amount")]].sum(axis=1)
    for who in ("foreigner", "institution", "individual"):
        net = fl[f"{who}_buy_amount"] - fl[f"{who}_sell_amount"]
        for n in (1, 5, 20):
            m[f"mkt_{who}_net_{n}"] = (net.rolling(n).sum() / tot.rolling(n).sum()).reindex(m.index)
            _spec(f"mkt_{who}_net_{n}", "시장 수급", f"KOSPI {who} 순매수 금액 / 전체 매수 금액, {n}일 합", n - 1, "날짜 공통")

    # 폭(breadth)·분산: t일 구성종목 기준
    r = p.assign(r=p["close"] / p["prev_close"] - 1)
    r["ma20"] = r.groupby("symbol")["close"].transform(lambda s: s.rolling(20).mean())
    u = universe[["trade_date", "symbol"]].merge(r[["symbol", "trade_date", "r", "close", "ma20"]], on=["symbol", "trade_date"])
    br = u.groupby("trade_date").agg(adv=("r", lambda s: (s > 0).mean()), dec=("r", lambda s: (s < 0).mean()),
                                     below_ma20=("close", lambda s: np.nan), disp=("r", "std"))
    br["below_ma20"] = u.assign(b=u.close < u.ma20).groupby("trade_date")["b"].mean()
    m["breadth_adv_1"] = br["adv"]
    _spec("breadth_adv_1", "시장", "구성종목 중 당일 상승 종목 비율", 1, "날짜 공통")
    m["breadth_adv_5"] = br["adv"].reindex(m.index).rolling(5).mean()
    _spec("breadth_adv_5", "시장", "breadth_adv_1의 5일 평균", 5, "날짜 공통")
    m["breadth_below_ma20"] = br["below_ma20"]
    _spec("breadth_below_ma20", "시장", "구성종목 중 Close < MA20 비율", 19, "날짜 공통")
    m["cs_dispersion_1"] = br["disp"]
    _spec("cs_dispersion_1", "시장", "구성종목 당일 수익률의 횡단면 표준편차", 1, "날짜 공통")
    m["cs_dispersion_5"] = br["disp"].reindex(m.index).rolling(5).mean()
    _spec("cs_dispersion_5", "시장", "cs_dispersion_1의 5일 평균", 5, "날짜 공통")
    m = m.reset_index()

    # 시장 대비 종목 변수
    rel = p[["symbol", "trade_date"]].copy()
    x = p[["symbol", "trade_date", "close", "prev_close"]].merge(
        pd.DataFrame({"trade_date": k.index, "kr": kr.values}), on="trade_date", how="left")
    x["r"] = x["close"] / x["prev_close"] - 1
    gx = x.groupby("symbol")
    for n in (5, 20):
        cr = gx["close"].transform(lambda s: s / s.shift(n) - 1)
        kn = x["trade_date"].map(m.set_index("trade_date")[f"mkt_ret_{n}"])
        rel[f"excess_ret_{n}"] = (cr - kn).values
        _spec(f"excess_ret_{n}", "시장 대비", f"ret_{n} - mkt_ret_{n}", n)
    x["rk"] = x["r"] * x["kr"]
    x["kk"] = x["kr"] ** 2
    gx = x.groupby("symbol")
    mean_r = gx["r"].transform(lambda s: s.rolling(60).mean())
    mean_k = gx["kr"].transform(lambda s: s.rolling(60).mean())
    cov = gx["rk"].transform(lambda s: s.rolling(60).mean()) - mean_r * mean_k
    var = gx["kk"].transform(lambda s: s.rolling(60).mean()) - mean_k ** 2
    beta = cov / var
    rel["beta_60"] = beta.values
    _spec("beta_60", "시장 대비", "cov(r, r_KOSPI) / var(r_KOSPI), 60일", 60)
    var_r = gx["r"].transform(lambda s: s.rolling(60).var(ddof=0))
    rel["idio_vol_60"] = np.sqrt((var_r - beta ** 2 * var).clip(lower=0)).values
    _spec("idio_vol_60", "시장 대비", "sqrt(var(r) - beta_60^2·var(r_KOSPI)), 60일 (시장모형 잔차 표준편차)", 60)
    x["kdown"] = (x["kr"] < 0).astype(float)
    x["r_down"] = x["r"].where(x["kr"] < 0)
    rel["down_beta_ret_60"] = x.groupby("symbol")["r_down"].transform(lambda s: s.rolling(60, min_periods=15).mean()).values
    _spec("down_beta_ret_60", "시장 대비", "KOSPI 하락일의 종목 평균 수익률, 60일(최소 15일)", 60)
    return m, rel


# ---------------------------------------------------------------- 수급 변수
def flow_features(p: pd.DataFrame, inv: pd.DataFrame, inst: pd.DataFrame, prog: pd.DataFrame,
                  short: pd.DataFrame, credit: pd.DataFrame, lend: pd.DataFrame,
                  vol_all: pd.DataFrame | None = None) -> pd.DataFrame:
    """투자자별·기관 세부·프로그램 매매, 공매도, 신용, 대차 변수를 만든다.

    분모 거래량: 투자자별·기관 세부 매매는 NXT 거래까지 합친 값이라 통합 거래량(vol_all)으로 나누고,
                공매도·프로그램 매매는 KRX 기준이라 KRX 거래량(p.volume)으로 나눈다. 신용·대차 잔고 일수도 통합 거래량을 쓴다.
    입력: p(prepare_prices), 각 수급 테이블(load_table 결과, 학습 구간),
          vol_all(symbol, trade_date, volume: KRX+NXT 통합 거래량. 없으면 p.volume을 쓴다)
    출력: DataFrame[symbol, trade_date, <변수들>, late_update_20]
    누수: t일 장 마감 후 공표분을 t일 값으로 쓴다(결정 사항). updated_at이 t+1일 09:00 이후인 행이 최근 20일 안에
          있으면 late_update_20 = True(사후 수정 가능성). 휴장일(12-31) 행은 거래일 캘린더에 맞춰 버린다.
    """
    base = p[["symbol", "trade_date", "volume", "close"]].copy()
    if vol_all is not None:
        base = base.merge(vol_all[["symbol", "trade_date", "volume"]].rename(columns={"volume": "volume_all"}),
                          on=["symbol", "trade_date"], how="left")
    else:
        base["volume_all"] = base["volume"]
    cal = pd.DatetimeIndex(sorted(p.trade_date.unique()))
    nxt = pd.Series(cal[1:].append(pd.DatetimeIndex([cal[-1] + pd.Timedelta(days=1)])), index=cal)

    def late(df):
        d = df[["symbol", "trade_date", "updated_at"]].copy()
        d = d[d.trade_date.isin(cal)]
        d["late"] = pd.to_datetime(d["updated_at"]) > d["trade_date"].map(nxt) + pd.Timedelta(hours=9)
        return d[["symbol", "trade_date", "late"]]

    lates = [late(t).rename(columns={"late": f"late_{i}"}) for i, t in enumerate((inv, short, credit, lend))]
    x = base.merge(inv, on=["symbol", "trade_date"], how="left", suffixes=("", "_inv"))
    ins = inst.pivot_table(index=["symbol", "trade_date"], columns="category", values="net_volume").add_prefix("inst_").reset_index()
    x = x.merge(ins, on=["symbol", "trade_date"], how="left")
    x = x.merge(prog[["symbol", "trade_date", "arbitrage_net_volume", "non_arbitrage_net_volume"]], on=["symbol", "trade_date"], how="left")
    x = x.merge(short[["symbol", "trade_date", "short_volume"]], on=["symbol", "trade_date"], how="left")
    x = x.merge(credit[["symbol", "trade_date", "margin_balance_qty", "margin_balance_rate", "margin_new_qty"]], on=["symbol", "trade_date"], how="left")
    x = x.merge(lend[["symbol", "trade_date", "balance_qty"]].rename(columns={"balance_qty": "lend_balance_qty"}), on=["symbol", "trade_date"], how="left")
    for l_ in lates:
        x = x.merge(l_, on=["symbol", "trade_date"], how="left")
    x = x.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    g = x.groupby("symbol")
    f = x[["symbol", "trade_date"]].copy()
    vol = x["volume"].replace(0, np.nan)

    def net_ratio(col, n, den_col="volume_all"):
        num = g[col].transform(lambda s: s.rolling(n).sum())
        den = g[den_col].transform(lambda s: s.rolling(n).sum()).replace(0, np.nan)
        return num / den

    # 거래정지일에 프로그램 매매 행이 없으므로 0으로 채운다(거래가 없었음)
    for col in ("arbitrage_net_volume", "non_arbitrage_net_volume"):
        x[col] = x[col].fillna(0)
    g = x.groupby("symbol")
    for who, col in (("foreign", "foreigner_net_volume"), ("inst", "institution_net_volume"),
                     ("indiv", "individual_net_volume"), ("pension", "inst_pension_fund"),
                     ("fininv", "inst_financial_investment"), ("prog_nonarb", "non_arbitrage_net_volume"),
                     ("prog_arb", "arbitrage_net_volume")):
        den_col = "volume" if who.startswith("prog") else "volume_all"
        den_name = "KRX 거래량" if den_col == "volume" else "통합 거래량(KRX+NXT)"
        for n in (1, 5, 20):
            f[f"{who}_net_{n}"] = net_ratio(col, n, den_col)
            _spec(f"{who}_net_{n}", "수급", f"sum({col}, {n}일) / sum({den_name}, {n}일)", n - 1, "t일 장 마감 후 공표분 사용")
    # 외국인 보유
    f["foreign_hold"] = x["foreigner_holding_rate"]
    _spec("foreign_hold", "수급", "외국인 보유율(0~1)", 0)
    f["foreign_hold_chg_20"] = x["foreigner_holding_rate"] - g["foreigner_holding_rate"].shift(20)
    _spec("foreign_hold_chg_20", "수급", "외국인 보유율 20일 변화", 20)
    # CFD 잔고 (결측 약 18%)
    f["cfd_buy_rate"] = x["cfd_buy_balance_rate"]
    _spec("cfd_buy_rate", "수급", "CFD 매수 잔고율", 0, "결측 약 18%")
    # 공매도: 2025-03-31 이전은 전면 금지 기간(체제 변화)
    # 공매도 행이 없는 날은 공매도 0으로 본다(가정). 결측은 대부분 금지 기간(2.7%)이고 재개 후는 0.08%
    x["short_volume"] = x["short_volume"].fillna(0)
    g = x.groupby("symbol")
    for n in (1, 5, 20):
        f[f"short_share_{n}"] = net_ratio("short_volume", n, "volume")
        _spec(f"short_share_{n}", "공매도·대차", f"sum(공매도 거래량, {n}일) / sum(KRX 거래량, {n}일)", n - 1,
              "2025-03-31 전 공매도 금지 기간과 체제가 다름")
    # 신용
    f["margin_rate"] = x["margin_balance_rate"]
    _spec("margin_rate", "신용", "신용잔고율(0~1)", 0)
    for n in (5, 20):
        f[f"margin_chg_{n}"] = x["margin_balance_qty"] / g["margin_balance_qty"].shift(n) - 1
        _spec(f"margin_chg_{n}", "신용", f"신용잔고 수량 {n}일 변화율", n)
    f["margin_days_20"] = x["margin_balance_qty"] / g["volume_all"].transform(lambda s: s.rolling(20).mean()).replace(0, np.nan)
    _spec("margin_days_20", "신용", "신용잔고 수량 / 20일 평균 통합 거래량", 19)
    # 대차
    for n in (5, 20):
        f[f"lend_chg_{n}"] = x["lend_balance_qty"] / g["lend_balance_qty"].shift(n) - 1
        _spec(f"lend_chg_{n}", "공매도·대차", f"대차잔고 수량 {n}일 변화율", n, "2024-01~07은 2024-08-08 일괄 재갱신")
    f["lend_days_20"] = x["lend_balance_qty"] / g["volume_all"].transform(lambda s: s.rolling(20).mean()).replace(0, np.nan)
    _spec("lend_days_20", "공매도·대차", "대차잔고 수량 / 20일 평균 통합 거래량", 19)
    # 늦은 갱신 플래그
    lt = x[[f"late_{i}" for i in range(4)]].fillna(False).astype(float).max(axis=1)
    f["late_update_20"] = lt.groupby(x["symbol"]).transform(lambda s: s.rolling(20, min_periods=1).max()).astype(bool)
    return f


# ---------------------------------------------------------------- 업종·횡단면 순위
def peer_features(f: pd.DataFrame, industry: pd.Series, cols: list[str]) -> pd.DataFrame:
    """같은 날 같은 업종(구성종목 기준) 평균을 만든다(자기 자신 포함).

    입력: f(symbol, trade_date, cols. 그날 구성종목 행만), industry(symbol -> 업종), cols
    출력: DataFrame[symbol, trade_date, ind_<col>]
    누수: 같은 날 값만 사용. 업종은 stocks 테이블의 현재값(정적 속성으로 가정)
    """
    x = f[["symbol", "trade_date"] + cols].copy()
    x["industry"] = x["symbol"].map(industry).fillna("(없음)")
    out = x[["symbol", "trade_date"]].copy()
    for c in cols:
        out[f"ind_{c}"] = x.groupby(["trade_date", "industry"])[c].transform("mean")
        _spec(f"ind_{c}", "업종", f"같은 날 같은 업종 구성종목의 {c} 평균", SPEC[c][2], "업종은 현재값")
    return out


def cs_rank(f: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """날짜별 횡단면 백분위 순위(0~1)를 만든다. 입력 행은 그날 구성종목만이어야 한다.

    입력: f(symbol, trade_date, cols)
    출력: DataFrame[symbol, trade_date, <col>_rk]
    누수: 같은 날 값만 사용
    """
    out = f[["symbol", "trade_date"]].copy()
    rk = f.groupby("trade_date")[cols].rank(pct=True)
    rk.columns = [f"{c}_rk" for c in cols]
    return pd.concat([out, rk], axis=1)


# ---------------------------------------------------------------- 전체 조립
PEER_COLS = ["ret_5", "ret_20", "vol_20", "foreign_net_5"]


def build_all(c: pd.DataFrame, idx: pd.DataFrame, universe: pd.DataFrame, flows: dict[str, pd.DataFrame],
              industry: pd.Series, clean: bool = True,
              vol_all: pd.DataFrame | None = None) -> tuple[pd.DataFrame, list[str], list[str]]:
    """모든 후보 변수를 만들어 유니버스(구성종목) 행에 붙인다.

    입력: c(일봉), idx(지수·금리), universe(trade_date, symbol, ...), flows(테이블명 -> DataFrame),
          industry(symbol -> 업종), clean(고가·저가 스파이크 보정), vol_all(수급 분모용 통합 거래량)
    출력: (features: 유니버스 행 × [원 변수, 업종 변수, 시장 변수, <종목 변수>_rk, late_update_20],
           stock_cols: 종목 단위 변수 이름, market_cols: 날짜 공통 변수 이름)
    누수: 모든 변수는 t일까지의 정보. 순위·업종 평균은 같은 날 구성종목끼리만 계산
    """
    p = prepare_prices(c, clean=clean)
    fp = price_features(p)
    mkt, rel = market_features(p, idx, universe, flows["market_indicator_investor_trading"])
    fl = flow_features(p, flows["stock_investor_trading"], flows["stock_institution_breakdown"],
                       flows["stock_program_trades"], flows["stock_short_selling"],
                       flows["stock_credit_trades"], flows["stock_securities_lending"], vol_all=vol_all)
    f = fp.merge(rel, on=["symbol", "trade_date"]).merge(fl, on=["symbol", "trade_date"])
    f = universe[["trade_date", "symbol"]].merge(f, on=["symbol", "trade_date"], how="left")
    stock_cols = [k for k in f.columns if k in SPEC]
    num = f.select_dtypes("number").columns
    f[num] = f[num].replace([np.inf, -np.inf], np.nan)
    peer = peer_features(f, industry, PEER_COLS)
    rk = cs_rank(f, stock_cols)
    f = f.merge(peer, on=["symbol", "trade_date"]).merge(rk, on=["symbol", "trade_date"])
    mkt = mkt.replace([np.inf, -np.inf], np.nan)
    f = f.merge(mkt, on="trade_date", how="left")
    market_cols = [k for k in mkt.columns if k in SPEC]
    return f, stock_cols + [f"ind_{k}" for k in PEER_COLS], market_cols
