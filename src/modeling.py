"""모델링 데이터셋과 시간 순서 교차검증 도구 (docs/modeling_plan.md)."""
import numpy as np
import pandas as pd

TARGET = "mdd_low"
HORIZON = 5  # 타깃 구간(거래일). purge 길이로도 쓴다

# 초기 변수 세트(modeling_plan.md 3절). 순위 버전이 아니라 원래 값을 쓴다.
STOCK_FEATURES = [
    "gk_5", "idio_vol_60",                                                     # 변동성
    "skew_60", "past_mdd_60", "n_down3_60", "downside_ratio_20", "min_intra_drop_20",  # 하방 특성
    "dist_low_60", "pos_20", "ret_20", "abs_ret_20",                           # 가격 위치·모멘텀
    "turnover_20", "log_mcap", "vol_ratio_20_60",                              # 유동성·규모
    "beta_60",                                                                 # 시장 민감도
    "margin_rate", "lend_days_20", "pension_net_20", "indiv_net_20", "prog_nonarb_net_20", "foreign_hold",  # 신용·대차·수급
    "ind_vol_20",                                                              # 업종
]
MARKET_FEATURES = ["mkt_vol_20", "mkt_ret_20", "mkt_ret_60", "cs_dispersion_1", "breadth_adv_5", "mkt_institution_net_20"]
FEATURES = STOCK_FEATURES + MARKET_FEATURES


def build_dataset(feat: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    """변수와 타깃을 합쳐 모델링 표본을 만든다.

    입력: feat(features_train.parquet), tg(target_train.parquet: use, mdd_low, mdd_close 등)
    출력: DataFrame[trade_date, symbol, FEATURES..., mdd_low, mdd_close] — tg.use인 행만
    누수: 변수는 t일까지의 정보(Step 4 절단 검사 통과), 타깃은 미래 값(학습·평가 전용).
          파생 변수 abs_ret_20은 같은 행의 ret_20만 쓴다.
    """
    f = feat.copy()
    f["abs_ret_20"] = f["ret_20"].abs()
    cols = ["trade_date", "symbol"] + FEATURES
    t = tg.loc[tg["use"], ["trade_date", "symbol", "mdd_low", "mdd_close"]]
    return t.merge(f[cols], on=["trade_date", "symbol"], how="left").sort_values(["trade_date", "symbol"]).reset_index(drop=True)


def make_blocks(dates: pd.Series, n_blocks: int = 5) -> pd.Series:
    """거래일을 시간 순서대로 n개 블록(1..n)으로 나눈다(블록마다 거래일 수가 거의 같다).

    입력: dates(표본의 trade_date), n_blocks
    출력: 각 행의 블록 번호 Series
    누수: 해당 없음(날짜만 사용)
    """
    days = np.sort(dates.unique())
    edges = np.array_split(days, n_blocks)
    block_of = {d: i + 1 for i, chunk in enumerate(edges) for d in chunk}
    return dates.map(block_of)


def expanding_folds(df: pd.DataFrame, block_col: str = "block", purge: int = HORIZON) -> list[dict]:
    """확장 창 교차검증 분할을 만든다: 블록 k를 검증하고, 블록 1..k-1로 학습한다.

    검증 블록 시작 직전 `purge`거래일의 학습 행은 뺀다. 그 행들의 타깃 구간(t+1..t+5)이 검증 블록과 겹치기 때문이다.
    입력: df(trade_date, block 포함), block_col, purge(거래일 수)
    출력: [{"fold": k, "train_idx": ndarray, "test_idx": ndarray, "purged": int}, ...] (k = 2..n)
    누수: 학습 행의 타깃 구간이 검증 기간과 겹치지 않도록 보장한다
    """
    days = np.sort(df["trade_date"].unique())
    pos = {d: i for i, d in enumerate(days)}
    dpos = df["trade_date"].map(pos).to_numpy()
    folds = []
    for k in sorted(df[block_col].unique())[1:]:
        test = df[block_col].to_numpy() == k
        start = dpos[test].min()
        train = (df[block_col].to_numpy() < k) & (dpos < start - purge)
        purged = int(((df[block_col].to_numpy() < k) & (dpos >= start - purge)).sum())
        folds.append({"fold": int(k), "train_idx": np.where(train)[0], "test_idx": np.where(test)[0], "purged": purged})
    return folds


# ---------------------------------------------------------------- 전처리·모델
def linear_matrix(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """선형 모델 입력을 만든다: 결측은 학습 구간 중앙값으로 채우고 결측 플래그를 붙인 뒤, 학습 구간 기준으로 표준화한다.

    양 끝 1%는 학습 구간 분위수로 자른다(극단값 완화).
    입력: train, test(DataFrame), feats(변수 이름)
    출력: (X_train, X_test) numpy 배열
    누수: 중앙값·분위수·평균·표준편차는 모두 학습 행에서만 계산한다
    """
    lo, hi = train[feats].quantile(0.01), train[feats].quantile(0.99)
    med = train[feats].median()
    out = []
    for d in (train, test):
        x = d[feats].clip(lo, hi, axis=1)
        flags = x.isna().astype(float).add_suffix("_na")
        out.append(pd.concat([x.fillna(med), flags], axis=1))
    keep = [c for c in out[0].columns if not c.endswith("_na") or out[0][c].sum() > 0]
    tr, te = out[0][keep], out[1][keep]
    mu, sd = tr.mean(), tr.std().replace(0, 1)
    return ((tr - mu) / sd).to_numpy(), ((te - mu) / sd).to_numpy()


# ---------------------------------------------------------------- 평가
def evaluate_predictions(df: pd.DataFrame, pred: str = "pred", target: str = TARGET) -> dict:
    """예측을 평가한다: 순위(날짜별 IC), 수준(날짜 평균 상관·편향), 오차(MAE·RMSE), 위험 상위 10% 적중.

    입력: df(trade_date, pred, target 포함)
    출력: dict
    누수: 해당 없음(평가 전용)
    """
    from src.evaluate import daily_ic
    ic = daily_ic(df, [pred], target)[pred].dropna()
    dm = df.groupby("trade_date")[[pred, target]].mean()
    err = df[pred] - df[target]
    top = df[pred] <= df[pred].quantile(0.10)          # 예측 낙폭이 가장 큰 10%
    event = df[target] <= -0.10
    return {
        "IC": ic.mean(), "IC_IR": ic.mean() / ic.std(),
        "level_corr": dm[pred].corr(dm[target]), "level_bias": (dm[pred] - dm[target]).mean(),
        "MAE": err.abs().mean(), "RMSE": np.sqrt((err ** 2).mean()),
        "top10_event_rate": event[top].mean(), "top10_event_capture": (event & top).sum() / max(event.sum(), 1),
        "n": len(df),
    }


def calibration_table(df: pd.DataFrame, pred: str = "pred", target: str = TARGET, q: int = 10) -> pd.DataFrame:
    """예측값 분위(전체 풀링)별 실제 평균 낙폭과 -10% 이상 하락 비율. 분위 10 = 예측 위험 최상위.

    입력: df(pred, target)
    출력: DataFrame[bucket, pred_mean, actual_mean, event_rate, n]
    누수: 해당 없음(평가 전용)
    """
    b = pd.qcut((-df[pred]).rank(method="first"), q, labels=range(1, q + 1))
    return df.assign(bucket=b).groupby("bucket", observed=True).agg(
        pred_mean=(pred, "mean"), actual_mean=(target, "mean"),
        event_rate=(target, lambda s: (s <= -0.10).mean()), n=(target, "size")).reset_index()


# ---------------------------------------------------------------- 최종 후보 모델: 변동성 수준 + 횡단면 잔차
LEVEL_FEATURES = ["gk_5", "idio_vol_60", "ind_vol_20"]


class VolResidModel:
    """두 부분으로 된 예측 모델.

    1) 수준: 변동성 변수(LEVEL_FEATURES, 원래 값)로 MDD_5를 선형 회귀한다. 시장 전체가 흔들리면 변동성이 함께 올라
       예측 수준도 올라간다.
    2) 횡단면 보정: 1)의 잔차에서 날짜 평균을 뺀 값을, 종목 변수의 날짜별 백분위 순위로 LightGBM 회귀한다.
       예측 보정값도 날짜 평균을 0으로 맞춰서, 날짜 수준은 1)만 결정한다.
    입력 변수는 모두 t일까지의 정보다(누수 없음). fit은 학습 행만, predict는 같은 날 구성종목끼리의 순위를 쓴다.
    """

    def __init__(self, level_features=None, cs_features=None, lgb_params=None):
        self.level_features = level_features or LEVEL_FEATURES
        self.cs_features = cs_features or STOCK_FEATURES
        self.lgb_params = dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=300,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5, verbose=-1)
        self.lgb_params.update(lgb_params or {})

    def _level_X(self, d):
        x = d[self.level_features].clip(self.lo_, self.hi_, axis=1).fillna(self.med_)
        return ((x - self.mu_) / self.sd_).to_numpy()

    def _ranks(self, d):
        return d.groupby("trade_date")[self.cs_features].rank(pct=True)

    def fit(self, d: pd.DataFrame) -> "VolResidModel":
        """입력: d(trade_date, 변수, mdd_low). 출력: self. 누수: 학습 행에서만 통계를 계산한다."""
        import lightgbm as lgb
        from sklearn.linear_model import Ridge
        f = self.level_features
        self.lo_, self.hi_ = d[f].quantile(0.01), d[f].quantile(0.99)
        self.med_ = d[f].median()
        x = d[f].clip(self.lo_, self.hi_, axis=1).fillna(self.med_)
        self.mu_, self.sd_ = x.mean(), x.std().replace(0, 1)
        self.level_ = Ridge(alpha=1.0).fit(self._level_X(d), d[TARGET])
        resid = d[TARGET] - self.level_.predict(self._level_X(d))
        resid = resid - resid.groupby(d["trade_date"]).transform("mean")
        self.cs_ = lgb.LGBMRegressor(**self.lgb_params).fit(self._ranks(d), resid)
        return self

    def predict_parts(self, d: pd.DataFrame) -> pd.DataFrame:
        """출력: DataFrame[level, cs_adj, pred] (d와 같은 인덱스). 누수: 같은 날 구성종목 순위만 쓴다."""
        level = self.level_.predict(self._level_X(d))
        adj = pd.Series(self.cs_.predict(self._ranks(d)), index=d.index)
        adj = adj - adj.groupby(d["trade_date"]).transform("mean")
        return pd.DataFrame({"level": level, "cs_adj": adj.to_numpy(), "pred": level + adj.to_numpy()}, index=d.index)

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        return self.predict_parts(d)["pred"].to_numpy()
