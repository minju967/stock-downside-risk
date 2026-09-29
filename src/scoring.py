"""위험도 점수(0~100, D 방식)와 등급 변환."""
import numpy as np
import pandas as pd

# 기본 등급: 점수 20점 단위 5등급(과거 예측 분포의 5분위). 경계는 grade_table.csv로 함께 저장한다.
GRADE_EDGES = [0, 20, 40, 60, 80, 100]
GRADE_LABELS = ["매우 낮음", "낮음", "보통", "높음", "매우 높음"]


def build_reference(pred: np.ndarray, n: int = 1001) -> np.ndarray:
    """점수 환산표를 만든다: 예측 위험(-pred)의 분위수 n개(0 ~ 100%).

    입력: pred(학습 기간 예측 MDD_5 배열), n(분위수 개수)
    출력: 오름차순 위험 분위수 배열(길이 n). 위험 = -예측 낙폭(클수록 위험)
    누수: 학습 기간 예측만 쓴다. 검증·운영에서는 이 표를 고정해서 쓴다
    """
    return np.quantile(-np.asarray(pred, dtype=float), np.linspace(0, 1, n))


def to_score(pred: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """예측 MDD_5를 0 ~ 100 위험도 점수로 바꾼다(환산표 안에서 선형 보간, 범위 밖은 0 또는 100).

    입력: pred(예측 MDD_5), reference(build_reference 결과)
    출력: 점수 배열(높을수록 위험)
    누수: 해당 없음(고정된 환산표만 사용)
    """
    risk = -np.asarray(pred, dtype=float)
    grid = np.linspace(0, 100, len(reference))
    return np.interp(risk, reference, grid, left=0.0, right=100.0)


def to_grade(score: np.ndarray, edges=GRADE_EDGES, labels=GRADE_LABELS) -> pd.Categorical:
    """점수를 등급으로 바꾼다. 마지막 구간은 100점을 포함한다.

    입력: score(0 ~ 100), edges(등급 경계), labels(등급 이름)
    출력: 순서 있는 Categorical
    누수: 해당 없음
    """
    e = list(edges[:-1]) + [edges[-1] + 1e-9]
    return pd.cut(np.asarray(score, dtype=float), e, right=False, labels=labels)
