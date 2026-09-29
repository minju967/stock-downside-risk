"""outputs/eda_report.md를 그림을 넣은 PDF로 변환한다.

사용: python -m src.report_pdf
방법: Markdown -> HTML(인쇄용 CSS) -> Windows Chrome/Edge 헤드리스 인쇄
"""
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "eda_report.md"
HTML = ROOT / "outputs" / "eda_report_print.html"
PDF = ROOT / "outputs" / "eda_report.pdf"

BROWSERS = [
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
]

# 섹션 제목(앞부분) -> 그 섹션 끝에 넣을 그림 [(파일, 설명)]
FIGURES = {
    "2. 데이터 문제": [("03b_krx_spikes.png", "그림 1. 월별 저가 스파이크 건수: 기존 DB 일봉(KRX+NXT 통합) vs KRX 정규장 일봉")],
    "3. 타깃 정의 의견": [
        ("04_target_ecdf.png", "그림 2. 네 가지 MDD_5 정의의 누적분포와 왼쪽 꼬리"),
        ("04_target_monthly.png", "그림 3. 월별 MDD_5 분위. 2026년 상반기 고변동 국면"),
        ("04_target_crash_event.png", "그림 4. KOSPI 급락일 전후 평균 MDD_5"),
    ],
    "4. 추천 변수 Top 20": [
        ("07_top20.png", "그림 5. Top 20 변수의 IC IR과 변동성 통제 후 IC IR"),
        ("06_deciles_top12.png", "그림 6. 분위별 초과 MDD_5 (|IC IR| 상위 12개)"),
        ("06_ic_halves.png", "그림 7. 전·후반 IC 비교"),
        ("06_market_ts.png", "그림 8. 시장 변수와 날짜별 평균 MDD_5의 시계열 상관"),
    ],
}

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { word-break: keep-all; overflow-wrap: anywhere; font-family: 'Malgun Gothic', '맑은 고딕', sans-serif; font-size: 9.6pt; line-height: 1.55; color: #1a1a19; }
h1 { font-size: 17pt; margin: 0 0 8px; padding-bottom: 6px; border-bottom: 2px solid #2a78d6; }
h2 { font-size: 13pt; margin: 22px 0 8px; padding-bottom: 3px; border-bottom: 1px solid #d9d8d4; break-after: avoid; }
h3 { font-size: 11pt; margin: 14px 0 6px; break-after: avoid; }
p, li { margin: 3px 0; }
ul, ol { padding-left: 20px; margin: 4px 0; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f1f0ec; padding: 0 3px; border-radius: 3px; }
pre { background: #f1f0ec; padding: 8px 10px; border-radius: 4px; overflow-x: auto; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 10px; font-size: 8.2pt; break-inside: auto; }
th, td { border: 1px solid #d9d8d4; padding: 3px 5px; vertical-align: top; word-break: keep-all; overflow-wrap: normal; }
td code { word-break: break-all; }
td:nth-child(n+3) { overflow-wrap: anywhere; }
table { table-layout: auto; max-width: 100%; }
th { background: #eef3fb; font-weight: bold; }
tr { break-inside: avoid; }
figure { margin: 10px 0 14px; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 118mm; }
figcaption { font-size: 8.4pt; color: #52514e; margin-top: 3px; }
hr { border: none; border-top: 1px solid #e4e3df; margin: 14px 0; }
.page-break { break-before: page; }
"""


def _win_path(p: Path) -> str:
    """/mnt/c/... 경로를 C:\\... 로 바꾼다."""
    s = str(p)
    m = re.match(r"^/mnt/([a-z])/(.*)$", s)
    return f"{m.group(1).upper()}:\\" + m.group(2).replace("/", "\\") if m else s


_LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s")


def _normalize_lists(text: str) -> str:
    """Python-Markdown 목록 규칙에 맞춘다: 목록 앞 빈 줄, 중첩 들여쓰기 2칸 -> 4칸."""
    out, prev = [], ""
    for line in text.splitlines():
        m = _LIST.match(line)
        if m:
            indent = len(m.group(1))
            line = " " * (indent * 2) + line.lstrip()
            if prev.strip() and not _LIST.match(prev) and not prev.lstrip().startswith("|"):
                out.append("")
        out.append(line)
        prev = line
    return "\n".join(out)


def build_html() -> str:
    """Markdown을 인쇄용 HTML로 바꾸고, 섹션 끝에 그림을 넣는다.

    입력: outputs/eda_report.md, outputs/figures/*.png
    출력: HTML 문자열
    누수: 해당 없음
    """
    md_text = _normalize_lists(SRC.read_text(encoding="utf-8"))
    # 섹션별로 나눠 그림 삽입
    parts = re.split(r"(?m)^(?=## )", md_text)
    out = []
    for part in parts:
        head = part.splitlines()[0][3:].strip() if part.startswith("## ") else ""
        figs = next((v for k, v in FIGURES.items() if head.startswith(k)), [])
        body = part.rstrip()
        if body.endswith("---"):
            body = body[:-3].rstrip()
        for fname, cap in figs:
            body += f'\n\n<figure><img src="figures/{fname}"><figcaption>{cap}</figcaption></figure>\n'
        out.append(body + "\n\n")
    html_body = markdown.markdown("".join(out), extensions=["tables", "fenced_code", "sane_lists"])
    # 4절(Top 20 표)은 새 페이지에서 시작
    for key in ("4. 추천 변수 Top 20",):
        html_body = html_body.replace(f"<h2>{key}", f'<h2 class="page-break">{key}', 1)
    return f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><title>EDA 리포트 v2</title><style>{CSS}</style></head><body>{html_body}</body></html>"


def main() -> None:
    """HTML을 저장하고 헤드리스 브라우저로 PDF를 만든다."""
    HTML.write_text(build_html(), encoding="utf-8")
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        raise SystemExit("Chrome/Edge를 찾을 수 없다.")
    cmd = [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={_win_path(PDF)}", "file:///" + _win_path(HTML).replace("\\", "/")]
    try:
        subprocess.run(cmd, check=True, timeout=180, capture_output=True)
    finally:
        HTML.unlink(missing_ok=True)   # 중간 HTML은 남기지 않는다
    print("저장:", PDF)


if __name__ == "__main__":
    main()
