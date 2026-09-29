"""EDA 그림 공통 스타일."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"

# 범주형 색상은 이 순서로만 배정한다(순환 금지)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

_KO_BOLD = ["/mnt/c/Windows/Fonts/malgunbd.ttf"]
_KO_FONTS = ["/mnt/c/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]


def setup() -> None:
    """한글 글꼴과 차분한 축·격자 스타일을 설정한다.

    입력: 없음
    출력: 없음(matplotlib rcParams 변경)
    누수: 해당 없음
    """
    for extra in _KO_BOLD:
        if Path(extra).exists():
            font_manager.fontManager.addfont(extra)
    for f in _KO_FONTS:
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            mpl.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
            break
    mpl.rcParams.update({
        "axes.unicode_minus": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "text.color": TEXT, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    })


def save(fig: plt.Figure, name: str) -> Path:
    """그림을 outputs/figures/<name>.png로 저장한다.

    입력: fig, name(확장자 제외)
    출력: 저장 경로
    누수: 해당 없음
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    return path
