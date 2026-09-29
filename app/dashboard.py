"""보유 종목 위험도 대시보드 (Streamlit).

실행: streamlit run app/dashboard.py
데이터: `python -m src.daily_score --date YYYY-MM-DD`가 만든 outputs/daily/ 파일
표시 원칙(2026-09-29 결정)
  - 종목별 예상 낙폭은 보여 주지 않는다(개별 정확도가 낮다).
  - 종목별로는 위험도 5단계, KOSPI200 안 오늘 순위, 전일 대비 변화, 최근 등급 추이를 보여 준다.
  - 등급별 낙폭은 학습 기간에 그 등급이었던 종목들의 실제 결과(고정표)로 설명한다.
  - 원천 데이터가 덜 적재된 날짜(usable = False)는 결과를 보여 주지 않는다.
"""
import html
import math
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.dashboard_data import grade_reference, holdings_view, list_dates, load_check, load_scores, market_grade_share  # noqa: E402
from src import holdings as H  # noqa: E402
from src import market_live as ML  # noqa: E402
from src.scoring import GRADE_LABELS  # noqa: E402

st.set_page_config(page_title="보유 종목 위험도", page_icon="📉", layout="wide")

# 등급 색: 파랑 - 초록 - 노랑 - 주황 - 빨강(사용자 지정, 2026-09-29). 인접 등급 정상 시력 ΔE ≥ 16.9,
# 색각 이상 시뮬레이션 ΔE ≥ 13.5 확인. 노랑은 밝은 배경 대비가 낮아 색만으로 구분하지 않도록 항상 등급 글자를 함께 쓴다.
GRADE_COLORS = ["#1f5fbf", "#34a853", "#fbd13a", "#f58a1f", "#b71c1c"]
PALETTE = {
    "light": {"grades": GRADE_COLORS,
              "text": "#0b0b0b", "text2": "#52514e", "muted": "#898781", "line": "#e1e0d9", "surface": "#fcfcfb", "hover": "#f3f2ee", "track": "#dddcd6"},
    "dark": {"grades": GRADE_COLORS,
             "text": "#ffffff", "text2": "#c3c2b7", "muted": "#898781", "line": "#2c2c2a", "surface": "#1a1a19", "hover": "#242422", "track": "#3a3a37"},
}
try:
    MODE = "dark" if st.context.theme.type == "dark" else "light"
except Exception:
    MODE = "light"
C = PALETTE[MODE]
GCOLOR = dict(zip(GRADE_LABELS, C["grades"]))

st.markdown(f"""
<style>
.rk-wrap {{ overflow-x: auto; width: 100%; }}
.rk-table {{ width: 100%; min-width: 760px; border-collapse: collapse; font-size: 0.95rem; color: {C['text']}; }}
.rk-table th {{ text-align: left; font-weight: 600; color: {C['text2']}; border-bottom: 1px solid {C['line']}; padding: 8px 10px; white-space: nowrap; }}
.rk-table td:first-child {{ white-space: nowrap; }}
.rk-table td {{ border-bottom: 1px solid {C['line']}; padding: 9px 10px; vertical-align: middle; }}
/* 종목별 위험도: 표가 넘치면 종목당 두 줄로 바꾼다(container query) */
.rk-hwrap {{ container-type: inline-size; width: 100%; overflow-x: auto; margin-bottom: 10px; }}
.rk-grid {{ display: grid; font-size: 0.95rem; color: {C['text']};
  grid-template-columns: minmax(95px, auto) minmax(147px, auto) minmax(77px, auto) minmax(104px, auto) minmax(205px, 1fr) minmax(138px, auto) minmax(147px, auto) minmax(147px, auto); }}
.rk-row {{ display: grid; grid-column: 1 / -1; grid-template-columns: subgrid; border-bottom: 1px solid {C['line']}; }}
.rk-grid > .rk-row:not(.h):hover {{ background: {C['hover']}; }}
.rk-row > div {{ display: flex; align-items: center; padding: 9px 10px; }}
.rk-row.h > div {{ justify-content: center; text-align: center; font-weight: 600; color: {C['text2']}; padding: 8px 10px; }}
.rk-row.h > .pos {{ flex-direction: column; align-items: stretch; }}
.rk-row > .num {{ justify-content: flex-end; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.rk-row > .nw {{ white-space: nowrap; }}
.rk-row > .chg {{ flex-direction: column; align-items: flex-start; justify-content: center; }}
.rk-row > .pos {{ gap: 10px; }}
.rk-sc {{ font-variant-numeric: tabular-nums; white-space: nowrap; min-width: 34px; text-align: right; }}
/* 표가 넘치면 종목당 두 줄: 위 = 코드·종목·수량·위험도·순위, 아래 = 점수 막대·전일 대비·추이 */
@container (max-width: 1059px) {{
  .rk-grid {{ grid-template-columns: minmax(95px, auto) minmax(147px, 2fr) minmax(77px, 1fr) minmax(104px, auto) minmax(138px, auto); }}
  .rk-row > .pos {{ grid-column: span 3; order: 2; padding-top: 4px; }}
  .rk-row > .chg, .rk-row > .trd {{ order: 2; padding-top: 4px; }}
  .rk-row > .top {{ padding-bottom: 2px; }}
}}
.rk-trend {{ white-space: nowrap; }}
.rk-table th, .rk-table th.num {{ text-align: center; }}
.rk-table tr:hover td {{ background: {C['hover']}; }}
.rk-table .num {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.rk-sub {{ color: {C['muted']}; font-size: 0.82rem; }}
.rk-dot {{ display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 7px; vertical-align: -1px; }}
.rk-grade {{ white-space: nowrap; font-weight: 600; }}
.rk-trend span {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin-right: 2px; vertical-align: middle; }}
.rk-trend span.none {{ border: 1px dashed {C['muted']}; }}
.rk-pos {{ position: relative; width: 100%; height: 12px; background: {C['track']}; border-radius: 4px; overflow: hidden; }}
.rk-pos .fill {{ position: absolute; top: 0; bottom: 0; left: 0; border-radius: 4px;
  background-image: linear-gradient(to right, {GRADE_COLORS[0]} 0 20%, {GRADE_COLORS[1]} 20% 40%, {GRADE_COLORS[2]} 40% 60%, {GRADE_COLORS[3]} 60% 80%, {GRADE_COLORS[4]} 80% 100%);
  background-repeat: no-repeat; }}
.rk-posh {{ display: flex; justify-content: space-between; width: 100%; font-weight: 400; font-size: 0.78rem; color: {C['muted']}; }}
.rk-donut {{ display: flex; flex-direction: column; align-items: center; color: {C['text']}; }}
.rk-dtitle {{ font-weight: 600; color: {C['text2']}; margin-bottom: 8px; }}
.rk-help {{ position: relative; display: inline-flex; align-items: center; justify-content: center; width: 16px; height: 16px;
  margin-left: 6px; border: 1px solid {C['muted']}; border-radius: 50%; font-size: 11px; font-weight: 600; color: {C['muted']};
  cursor: help; vertical-align: 1px; }}
.rk-help .tip {{ visibility: hidden; opacity: 0; transition: opacity .12s; position: absolute; top: 24px; left: 50%; transform: translateX(-50%);
  width: 340px; padding: 8px 12px; border-radius: 8px; background: {C['surface']}; color: {C['text']}; border: 1px solid {C['line']};
  box-shadow: 0 4px 14px rgba(0,0,0,.15); font-size: 0.85rem; font-weight: 400; line-height: 1.5; text-align: left; z-index: 20; }}
.rk-help:hover .tip, .rk-help:focus .tip {{ visibility: visible; opacity: 1; }}
.rk-mcard {{ border: 1px solid {C['line']}; border-radius: 8px; padding: 14px 16px; min-height: 180px; box-sizing: border-box;
  display: flex; flex-direction: column; justify-content: center; }}
.rk-mcard .lbl {{ color: {C['text2']}; font-size: 0.9rem; font-weight: 600; }}
.rk-mcard .val {{ font-size: 2rem; font-weight: 600; color: {C['text']}; font-variant-numeric: tabular-nums; line-height: 1.3; }}
.rk-mcard .chg {{ font-size: 0.95rem; font-variant-numeric: tabular-nums; }}
.rk-mcard .ts {{ color: {C['muted']}; font-size: 0.78rem; margin-top: 4px; }}
.rk-shead {{ display: flex; align-items: baseline; flex-wrap: wrap; gap: 6px 18px; margin: 1.2rem 0 0.8rem; }}
.rk-shead h3 {{ margin: 0; padding: 0; }}
.rk-shead h3 a, .rk-shead h3 [data-testid='stHeaderActionElements'] {{ display: none !important; }}
.rk-hleg {{ display: flex; flex-wrap: wrap; gap: 12px; font-size: 0.82rem; color: {C['text2']}; }}
.rk-hleg .rk-dot {{ width: 9px; height: 9px; margin-right: 5px; }}
</style>
""", unsafe_allow_html=True)


def dot(g: str) -> str:
    return f'<span class="rk-dot" style="background:{GCOLOR[g]}"></span>'


def grade_cell(g: str) -> str:
    return f'<span class="rk-grade">{dot(g)}{html.escape(g)}</span>'


def change_cell(chg, prev) -> str:
    if pd.isna(chg):
        return '<span class="rk-sub">–</span>'
    if chg == 0:
        return '<span class="rk-sub" style="white-space:nowrap">변화 없음</span>'
    arrow = "▲ 상승" if chg > 0 else "▼ 하락"
    return f'<b style="white-space:nowrap">{arrow}</b><span class="rk-sub" style="white-space:nowrap">(전일 {html.escape(prev)})</span>'


def trend_cell(trend) -> str:
    boxes = []
    for d, g in trend:
        tip = f"{d[5:]} {g if g else '구성종목 아님'}"
        boxes.append(f'<span title="{html.escape(tip)}" style="background:{GCOLOR[g]}"></span>' if g
                     else f'<span class="none" title="{html.escape(tip)}"></span>')
    return f'<span class="rk-trend">{"".join(boxes)}</span>'


def score_cell(score: float) -> str:
    """위험도 점수 막대(0 ~ 100): 왼쪽부터 점수만큼 색을 채우고 나머지는 회색.

    색 구간은 20점 단위로 등급 경계와 같다(파랑 → 빨강). 막대 끝 색 = 그 종목의 등급 색.
    """
    x = max(float(score) / 100, 0.01)                  # 0점도 보이도록 최소 길이
    bar = (f'<div class="rk-pos" title="위험도 점수 {score:.0f}점 (과거 대비 백분위)">'
           f'<div class="fill" style="width:{x * 100:.1f}%; background-size:{100 / x:.1f}% 100%"></div></div>')
    return f'{bar}<span class="rk-sc">{score:.0f}</span>'


def donut(title: str, counts: pd.Series, help: str | None = None) -> str:
    """등급별 종목 수 도넛(SVG). 조각 사이 2px 틈, 가운데 종목 수. 조각에 마우스를 올리면 등급·종목 수·비중."""
    counts = counts.reindex(GRADE_LABELS).fillna(0).astype(int)
    total = int(counts.sum())
    r, sw, size = 100, 36, 260
    circ = 2 * math.pi * r
    gap = 2.0 if (counts > 0).sum() > 1 else 0.0
    segs, start = [], 0.0
    for g in GRADE_LABELS:
        n = int(counts[g])
        if n == 0:
            continue
        length = circ * n / total
        tip = f"{g}: {n}종목 ({n / total:.0%})"
        segs.append(
            f'<circle cx="{size / 2}" cy="{size / 2}" r="{r}" fill="none" stroke="{GCOLOR[g]}" stroke-width="{sw}" '
            f'stroke-dasharray="{max(length - gap, 0.5):.2f} {circ:.2f}" stroke-dashoffset="{-start:.2f}" '
            f'transform="rotate(-90 {size / 2} {size / 2})"><title>{html.escape(tip)}</title></circle>')
        start += length
    center = (f'<text x="50%" y="48%" text-anchor="middle" font-size="36" font-weight="600" fill="{C["text"]}">{total}</text>'
              f'<text x="50%" y="60%" text-anchor="middle" font-size="14" fill="{C["muted"]}">종목</text>')
    svg = (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" aria-label="{html.escape(title)} 등급 분포">'
           f'<circle cx="{size / 2}" cy="{size / 2}" r="{r}" fill="none" stroke="{C["track"]}" stroke-width="{sw}"/>'
           f'{"".join(segs)}{center}</svg>')
    tip = (f'<span class="rk-help" tabindex="0" aria-label="{html.escape(help)}">?<span class="tip">{html.escape(help)}</span></span>'
           if help else "")
    return f'<div class="rk-donut"><div class="rk-dtitle">{html.escape(title)}{tip}</div>{svg}</div>'


# ---------------------------------------------------------------- 사이드바: 기준일, 보유 종목 입력
dates = list_dates()
if dates.empty:
    st.error("점수 파일이 없습니다. `python -m src.daily_score --date YYYY-MM-DD`로 먼저 만드세요.")
    st.stop()

with st.sidebar:
    st.header("설정")
    usable_idx = dates.index[dates.usable]
    date = st.selectbox(
        "기준일 (이 날 종가 기준)", dates.date.tolist(),
        index=int(usable_idx[0]) if len(usable_idx) else 0,
        format_func=lambda d: d if bool(dates.set_index("date").loc[d, "usable"]) else f"{d}  ⚠ 데이터 미완성")
    st.caption("다음 거래일 장 시작 전에 참고하는 값입니다.")
    st.divider()
    source = st.radio("보유 종목 입력", ["고객 조회", "직접 입력"])

scores = load_scores(date)


@st.cache_data(ttl=600, show_spinner=False)
def cached_customer_list(as_of: str):
    return H.customers_by_age(as_of, load_scores(as_of)["종목코드"].tolist())


@st.cache_data(ttl=600, show_spinner=False)
def cached_holdings(cid: int, as_of: str):
    return H.load_holdings(cid, as_of)


options = [f"{n} ({s})" for n, s in zip(scores["종목명"], scores["종목코드"])]
options_sorted = sorted(options)

with st.sidebar:
    if source == "고객 조회":
        holdings = pd.DataFrame(columns=["symbol", "name", "quantity", "market"])
        try:
            clist = cached_customer_list(date)
        except Exception as e:                                   # DB 접속·권한 오류
            clist = None
            st.error(f"고객 목록을 불러오지 못했습니다: {e}")
        if clist is not None and not clist.empty:
            bands = list(dict.fromkeys(clist["age_band"]))
            band = st.selectbox("연령대", bands)
            sub = clist[clist["age_band"] == band].set_index("customer_id")
            cid = st.selectbox("고객", sub.index.tolist(),
                               format_func=lambda c: f"{sub.loc[c, 'name']} ({sub.loc[c, 'age']}세, ID {c})")
            try:
                holdings = cached_holdings(int(cid), date)
                n_k = int(holdings["symbol"].isin(scores["종목코드"]).sum())
                st.success(f"**{sub.loc[cid, 'name']}** ({sub.loc[cid, 'age']}세, ID {cid})  \n"
                           f"보유 {len(holdings)}종목 · KOSPI200 {n_k}종목")
            except Exception as e:
                st.error(f"보유 종목을 불러오지 못했습니다: {e}")
        st.caption("연령대별로 기준일에 KOSPI200 종목을 가장 많이 보유한 고객 10명입니다(나이는 기준일 만 나이). "
                   "보유 종목은 기준일까지 매수분이며, 여러 계좌의 같은 종목은 합산합니다.")
    else:
        if "manual" not in st.session_state:
            st.session_state.manual = pd.DataFrame({"종목": ["삼성전자 (005930)", "SK하이닉스 (000660)"], "수량": [10, 5]})
        edited = st.data_editor(
            st.session_state.manual, num_rows="dynamic", width="stretch", hide_index=True,
            column_config={"종목": st.column_config.SelectboxColumn("종목", options=options_sorted, required=True),
                           "수량": st.column_config.NumberColumn("수량", min_value=0, step=1)})
        edited = edited.dropna(subset=["종목"])
        holdings = pd.DataFrame({"symbol": edited["종목"].str.extract(r"\((\w{6})\)$")[0],
                                 "name": edited["종목"].str.replace(r"\s*\(\w{6}\)$", "", regex=True),
                                 "quantity": edited["수량"]})
        st.caption("KOSPI200 구성종목만 고를 수 있습니다. 행을 추가·삭제할 수 있습니다.")

# ---------------------------------------------------------------- 본문


# ---------------------------------------------------------------- 시장 현황(토스증권 Open API, 5분마다 자동 갱신)
UP, DOWN = "#d03b3b", "#2a78d6"          # 국내 관례: 상승 빨강, 하락 파랑(항상 ▲▼ 기호와 함께)


@st.cache_resource
def toss_client():
    """토큰을 공유하는 클라이언트 1개(토큰을 다시 받으면 이전 토큰이 무효가 되므로 프로세스당 하나만 쓴다)."""
    return ML.TossClient()


@st.cache_resource
def last_market() -> dict:
    """마지막으로 성공한 시장 정보(갱신 실패 시 대신 보여 준다)."""
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market():
    c = toss_client()
    return {"kospi": ML.get_index_quote(c, "KOSPI"), "kosdaq": ML.get_index_quote(c, "KOSDAQ"),
            "intraday": ML.get_index_intraday(c, "KOSPI"),
            "fx": ML.get_usdkrw(c), "fetched": pd.Timestamp.now(tz="Asia/Seoul")}


def change_html(chg, pct, digits=2) -> str:
    if chg is None:
        return f'<span class="rk-sub">전일 대비 정보 없음</span>'
    color, mark = (UP, "▲") if chg > 0 else (DOWN, "▼") if chg < 0 else (C["text2"], "–")
    return f'<span style="color:{color}">{mark} {abs(chg):,.{digits}f} ({pct:+.2f}%)</span>'


def market_card(label, value, chg_html, ts) -> str:
    return (f'<div class="rk-mcard"><div class="lbl">{label}</div><div class="val">{value}</div>'
            f'<div class="chg">{chg_html}</div><div class="ts">{ts}</div></div>')


@st.fragment(run_every="5m")
def market_section():
    help_txt = "토스증권 Open API · 5분마다 자동 갱신. 환율은 참고용 표시 환율이며 실제 거래 환율과 다를 수 있습니다."
    if not ML.credentials_available():
        st.subheader("오늘의 시장 현황", anchor=False, help=help_txt)
        st.info("실시간 KOSPI·KOSDAQ·환율을 보려면 `.env`에 `TOSS_CLIENT_ID`, `TOSS_CLIENT_SECRET`을 넣고, "
                "토스증권 개발자센터에 이 서버의 IP를 허용 IP로 등록하세요.")
        return
    last = last_market()                                     # 마지막 성공 값(서버 프로세스 공유)
    err = None
    try:
        m = fetch_market()
        last["m"] = m
    except Exception as e:                                   # 네트워크·인증·한도 오류: 마지막 성공 값을 계속 보여 준다
        err, m = e, last.get("m")
    st.subheader("오늘의 시장 현황", anchor=False,
                 help=f"{help_txt} 마지막 조회 {m['fetched']:%H:%M:%S}." if m else help_txt)
    if m is None:
        st.warning(f"실시간 시장 정보를 불러오지 못했습니다: {err}")
        return
    if err is not None:
        st.caption(f"⚠️ 최신 시세 갱신에 실패해 {m['fetched']:%H:%M} 조회 값을 보여 줍니다. 5분 뒤 다시 시도합니다. ({err})")
    k, kq, fx = m["kospi"], m["kosdaq"], m["fx"]
    # 지수 현재가 시각이 없으면(API null) 마지막 1분봉 시각을 기준 시각으로 쓴다
    idx_ts = k.as_of or (m["intraday"][-1][0] if m["intraday"] else None)
    idx_ts_txt = f"{idx_ts:%m-%d %H:%M} 기준" if idx_ts else ""
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2.2])
    c1.markdown(market_card("KOSPI", f"{k.price:,.2f}", change_html(k.change, k.change_pct), idx_ts_txt),
                unsafe_allow_html=True)
    c2.markdown(market_card("KOSDAQ", f"{kq.price:,.2f}", change_html(kq.change, kq.change_pct), idx_ts_txt),
                unsafe_allow_html=True)
    c3.markdown(market_card("원/달러 환율 (매매기준율)", f"{fx.mid:,.2f}", change_html(fx.change, fx.change_pct),
                            f"{fx.as_of:%m-%d %H:%M} 기준 · 전일 동시각 대비"), unsafe_allow_html=True)
    if m["intraday"]:
        d = pd.DataFrame(m["intraday"], columns=["시각", "KOSPI"])
        d["시각"] = d["시각"].dt.tz_localize(None)
        line_color = UP if (k.prev_close and d["KOSPI"].iloc[-1] >= k.prev_close) else DOWN
        base = alt.Chart(d).encode(x=alt.X("시각:T", axis=alt.Axis(format="%H:%M", title=None, grid=False)))
        line = base.mark_line(strokeWidth=2, color=line_color).encode(
            y=alt.Y("KOSPI:Q", scale=alt.Scale(zero=False), axis=alt.Axis(title=None, format=",.0f")),
            tooltip=[alt.Tooltip("시각:T", format="%H:%M"), alt.Tooltip("KOSPI:Q", format=",.2f")])
        layers = [line]
        if k.prev_close:
            layers.append(alt.Chart(pd.DataFrame({"전일 종가": [k.prev_close]})).mark_rule(
                strokeDash=[4, 4], color=C["muted"]).encode(y="전일 종가:Q", tooltip=[alt.Tooltip("전일 종가:Q", format=",.2f")]))
        c4.markdown(f'<div class="rk-dtitle">KOSPI 장중 ({d["시각"].iloc[-1]:%m-%d}, 점선 = 전일 종가)</div>',
                    unsafe_allow_html=True)
        c4.altair_chart(alt.layer(*layers).properties(height=146), width="stretch")   # 카드 높이(180px) = 제목 + 간격 + 차트


market_section()
st.divider()
st.subheader("보유 종목 위험도", anchor=False)
st.caption(f"기준일 {date} 종가 기준 · 향후 5거래일 안의 하락 위험 · KOSPI200 구성종목 대상")
# KOSPI200 구성종목 수 설명(위험도 분포의 'KOSPI200 전체' 물음표에 표시)
mem = load_check(date).get("구성종목", {})
kospi_note = None
if mem:
    kospi_note = f"기준일 KOSPI200 구성종목은 {mem['구성종목수']}개입니다({mem['스냅샷_기준일']} 구성종목 스냅샷 기준)."
    extra = mem.get("초과_편입", [])
    if mem["구성종목수"] != 200 and extra:
        who = ", ".join(f"{e['name']}({e['list_date']} 상장)" if e.get("list_date") else e["name"] for e in extra)
        kospi_note += f" {who}이(가) 편입되고 편출된 종목이 없어 200개를 넘습니다(신규 상장 종목 임시 편입)."

row = dates.set_index("date").loc[date]
if not row.usable:
    st.error(f"{date} 결과는 원천 데이터가 모두 적재되지 않아 사용할 수 없습니다. 적재가 끝난 뒤 다시 계산합니다.\n\n"
             f"비어 있는 변수: {', '.join(row.warn.keys())}")
    st.stop()

if holdings.empty:
    st.info("보유 종목이 없습니다. 고객을 선택하거나 종목을 직접 입력하세요.")
    st.stop()

view, excluded = holdings_view(holdings, date)
mkt = market_grade_share(date)

if view.empty:                                               # 보유 종목이 모두 KOSPI200 밖이면 평가할 것이 없다
    names = ", ".join(f"{n} ({s_})" for n, s_ in zip(excluded["name"], excluded["symbol"]))
    st.info(f"보유 종목 중 기준일 KOSPI200 구성종목이 없어 위험도를 평가할 수 없습니다.  \n보유 종목: {names}")
    st.stop()

# 요약
c1, c2, c3, c4 = st.columns(4)
c1.metric("평가 종목", f"{len(view)}개", help="KOSPI200 구성종목인 보유 종목 수")
c2.metric("높음 이상", f"{int(view.등급.isin(['높음', '매우 높음']).sum())}개")
up = int((view.변화 > 0).sum())
c3.metric("전일보다 위험 상승", f"{up}개", help="전 거래일보다 등급이 올라간 종목 수")
c4.metric("시장 전체 '매우 높음' 비중", f"{mkt['매우 높음']:.0%}",
          help="KOSPI200 구성종목 중 '매우 높음' 비중. 평소(학습 기간)는 약 20%")

if mkt["매우 높음"] >= 0.4:
    st.warning(f"오늘은 KOSPI200 종목의 {mkt['매우 높음']:.0%}가 '매우 높음'입니다. 시장 전체가 불안한 시기라 등급보다 "
               "**KOSPI200 내 순위**로 종목을 비교하세요.")

# 등급 분포: 보유 vs 시장
if len(view):
    legend = "".join(f"<span>{dot(g)}{g}</span>" for g in GRADE_LABELS)
    st.markdown(f"<div class='rk-shead'><h3>위험도 분포</h3><div class='rk-hleg'>{legend}</div></div>",
                unsafe_allow_html=True)
    left, right = st.columns(2)
    left.markdown(donut("내 보유 종목", view.등급.value_counts()), unsafe_allow_html=True)
    right.markdown(donut("KOSPI200 전체", scores["등급"].value_counts(), help=kospi_note), unsafe_allow_html=True)

# 보유 종목 표
st.subheader("보유 종목 하락 위험도", anchor=False)
HEAD = ("<div class='rk-row h'><div class='top'>코드</div><div class='top'>종목</div><div class='top'>수량</div>"
        "<div class='top'>위험도</div>"
        "<div class='pos'>위험도 점수<div class='rk-posh'><span>0 안전</span><span>위험 100</span></div></div>"
        "<div class='top'>KOSPI200<br>내 순위</div><div class='chg'>전일 대비</div><div class='trd'>최근 등급 추이</div></div>")
rows = []
for r in view.itertuples():
    qty = "" if pd.isna(r.quantity) else f"{int(r.quantity):,}"
    rows.append(
        "<div class='rk-row'>"
        f"<div class='top nw rk-sub'>{r.symbol}</div>"
        f"<div class='top nw'><b>{html.escape(r.종목명)}</b></div>"
        f"<div class='top num'>{qty}</div>"
        f"<div class='top nw'>{grade_cell(r.등급)}</div>"
        f"<div class='pos'>{score_cell(r.점수)}</div>"
        f"<div class='top num'>{int(r.순위)}위&nbsp;<span class='rk-sub'>/ {int(r.종목수)}</span></div>"
        f"<div class='chg'>{change_cell(r.변화, r.전일_등급)}</div>"
        f"<div class='trd'>{trend_cell(r.추이)}</div></div>")
st.markdown(f"<div class='rk-hwrap'><div class='rk-grid'>{HEAD}{''.join(rows)}</div></div>", unsafe_allow_html=True)
st.caption("위험도 점수는 과거(2024-04 ~ 2026-06) 예측 분포 대비 백분위(0 ~ 100)이고, 막대 색 구간은 등급 경계(20점 단위)와 같습니다. "
           "KOSPI200 내 순위 1위 = 기준일 KOSPI200 구성종목 중 가장 위험(점수와 같이 최근 3거래일 예측 평균 기준).")

if len(excluded):
    names = ", ".join(f"{n} ({s})" for n, s in zip(excluded["name"], excluded["symbol"]))
    st.info(f"KOSPI200 구성종목이 아니라 평가하지 않은 종목: {names}")

# 등급 설명
st.subheader("위험도 등급 설명", anchor=False)
ref = grade_reference()
held = set(view.등급)
ref_rows = "".join(
    f"<tr><td>{grade_cell(r.등급)}{' <span class=rk-sub>· 보유</span>' if r.등급 in held else ''}</td>"
    f"<td class='num'>{r.평균_낙폭:.1f}%</td><td class='num'>{r.하위10_낙폭:.1f}%</td>"
    f"<td class='num'>약 {r.하락10_확률 * 100:.0f}%</td></tr>" for r in ref.itertuples())
st.markdown(
    "<div class='rk-wrap'><table class='rk-table' style='min-width:520px'><thead><tr><th>등급</th><th class='num'>5일 안 평균 최대 낙폭</th>"
    "<th class='num'>10번 중 1번은 이보다 더 하락</th><th class='num'>-10% 이상 하락 확률</th></tr></thead>"
    f"<tbody>{ref_rows}</tbody></table></div>", unsafe_allow_html=True)
st.caption("2024-04 ~ 2026-06에 각 등급을 받은 종목들의 실제 결과입니다(기준일 종가 대비 5거래일 안의 최저가). "
           "앞으로의 결과를 보장하지 않습니다.")

with st.expander("이 위험도를 읽는 법"):
    st.markdown("""
- **위험도(5단계)** 는 과거(2024-04 ~ 2026-06)와 비교한 이 종목의 하락 위험 수준입니다. 시장 전체가 불안하면 대부분 종목의 등급이 함께 올라갑니다.
- **KOSPI200 내 순위** 는 같은 날 구성종목끼리 비교한 순위입니다. 위험 상위 20%와 하위 20%를 비교하면 약 80%의 경우 순서가 맞았습니다. 순위가 비슷한 종목끼리의 차이는 큰 의미가 없습니다.
- 종목별 "몇 % 하락" 수치는 보여 주지 않습니다. 개별 종목의 낙폭은 오차가 커서(검증 기간 평균 ±4%p) 참고하기 어렵습니다.
- 특정 날의 시장 급락은 미리 알려 주지 못합니다. 변동성이 커진 뒤에 위험도가 따라 올라갑니다.
- 등급은 최근 3거래일 예측의 평균으로 매겨 하루하루 크게 흔들리지 않게 했습니다.
""")
