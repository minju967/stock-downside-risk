"""변수-타깃 관계 평가 함수 (Step 5)."""
import numpy as np
import pandas as pd
from scipy import stats


def daily_ic(df: pd.DataFrame, feats: list[str], target: str, date: str = "trade_date", min_n: int = 30) -> pd.DataFrame:
    """날짜별 횡단면 Spearman 상관(IC)을 계산한다.

    입력: df(날짜·종목 행, feats와 target 컬럼), feats, target, min_n(날짜별 최소 유효 종목 수)
    출력: DataFrame[index=날짜, columns=feats] (유효 종목이 min_n 미만인 날은 NaN)
    누수: 해당 없음(평가 전용. target은 미래 값이므로 변수로 쓰지 않는다)
    """
    out = {}
    for f in feats:
        ok = df[f].notna() & df[target].notna()
        x = df[f].where(ok)
        yy = df[target].where(ok)
        rx = x.groupby(df[date]).rank()
        ry = yy.groupby(df[date]).rank()
        rx = rx - rx.groupby(df[date]).transform("mean")
        ry = ry - ry.groupby(df[date]).transform("mean")
        num = (rx * ry).groupby(df[date]).sum()
        den = np.sqrt((rx ** 2).groupby(df[date]).sum() * (ry ** 2).groupby(df[date]).sum())
        n = ok.groupby(df[date]).sum()
        out[f] = (num / den).where(n >= min_n)
    return pd.DataFrame(out)


def ic_summary(ic: pd.DataFrame, nonoverlap_days: pd.DatetimeIndex, split: str) -> pd.DataFrame:
    """IC 시계열을 요약한다: 평균, 표준편차, IR, 양수 비율, 월별 부호 일치율, 겹치지 않는 표본 t값, 전·후반 평균.

    입력: ic(daily_ic 결과), nonoverlap_days(5거래일 간격 날짜), split(전·후반 경계일)
    출력: DataFrame[index=변수]
    누수: 해당 없음(평가 전용)
    """
    m = ic.mean()
    s = ic.std()
    icn = ic[ic.index.isin(nonoverlap_days)]
    monthly = ic.groupby(ic.index.to_period("M")).mean()
    same_sign = (np.sign(monthly) == np.sign(m)).mean()
    first, second = ic[ic.index < split].mean(), ic[ic.index >= split].mean()
    return pd.DataFrame({
        "mean_ic": m, "ic_std": s, "ic_ir": m / s, "pos_share": (ic > 0).mean(),
        "month_sign_consistency": same_sign,
        "t_nonoverlap": icn.mean() / icn.std() * np.sqrt(icn.notna().sum()),
        "ic_first": first, "ic_second": second,
        "n_days": ic.notna().sum(),
    })


def decile_means(df: pd.DataFrame, feat: str, target: str, date: str = "trade_date", q: int = 10) -> pd.Series:
    """날짜별 분위(1~q)로 나눈 뒤, 분위별 '날짜 평균 대비 초과 타깃'의 평균을 구한다.

    입력: df, feat, target, q
    출력: Series(index=1..q) — 시장 공통 효과를 뺀 횡단면 관계
    누수: 해당 없음(평가 전용)
    """
    d = df[[date, feat, target]].dropna()
    d["bucket"] = np.ceil(d.groupby(date)[feat].rank(pct=True) * q).clip(1, q)
    d["excess"] = d[target] - d.groupby(date)[target].transform("mean")
    return d.groupby("bucket")["excess"].mean()


def shape_of(dm: pd.Series) -> str:
    """분위별 평균의 모양을 분류한다: 단조 증가/감소, U자, 역U자, 약함."""
    rho = stats.spearmanr(dm.index, dm.values)[0]
    mid = dm.iloc[3:7].mean()
    ends = (dm.iloc[0], dm.iloc[-1])
    spread = dm.max() - dm.min()
    if abs(rho) >= 0.8:
        return "단조 증가" if rho > 0 else "단조 감소"
    if min(ends) - mid > 0.25 * spread:
        return "역U자(양 끝이 안전)"
    if mid - max(ends) > 0.25 * spread:
        return "U자(양 끝이 위험)"
    return "비단조·약함"


def ts_corr(y: pd.Series, x: pd.Series, nonoverlap_days: pd.DatetimeIndex, split: str) -> dict:
    """날짜 단위 시계열 Spearman 상관(전체, 겹치지 않는 표본, 전·후반).

    입력: y(날짜별 평균 타깃), x(날짜별 변수), nonoverlap_days, split
    출력: dict
    누수: 해당 없음(평가 전용)
    """
    d = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    dn = d[d.index.isin(nonoverlap_days)]
    r_all = stats.spearmanr(d.x, d.y)[0]
    r_n, p_n = stats.spearmanr(dn.x, dn.y)
    f, s = d[d.index < split], d[d.index >= split]
    return {"ts_rho": r_all, "ts_rho_nonoverlap": r_n, "p_nonoverlap": p_n, "n_nonoverlap": len(dn),
            "ts_first": stats.spearmanr(f.x, f.y)[0], "ts_second": stats.spearmanr(s.x, s.y)[0]}
