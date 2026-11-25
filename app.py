from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
import streamlit as st

st.set_page_config(page_title="올림픽 개최국 효과", page_icon="🏅", layout="wide")

HERE = Path(__file__).parent
SEASON_KO = {"Summer": "하계", "Winter": "동계"}
SEASON_EN = {ko: en for en, ko in SEASON_KO.items()}
NO_FOCUS = "(없음)"
SEASON_COLORS = {"하계": "#D55E00", "동계": "#0072B2"}  # 색맹 안전 두 색
BOYCOTT = {(1980, "Summer"), (1984, "Summer")}
INK = "#7B61FF"                    # 주 선·강조 표식 (보라: 하계 주황·동계 파랑과 구분)
MUTED = "#8A8F98"                  # 기준선·오차 막대·지시선
EDGE = "#FFFFFF"                   # 점 테두리
LABEL_BG = "rgba(128,128,128,0.22)" 
LOG_TICKS = [0.5, 0.7, 1, 1.5, 2, 3, 5, 10, 20]     # 배수 축 (차트 ①·②)
SHARE_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]  # 점유율 축 (차트 ④)
CHART_H, MARGIN_T, MARGIN_B = 560, 60, 70
PLOT_PX = CHART_H - MARGIN_T - MARGIN_B   # 그림 영역 높이(px). 가로가 더 좁으면 ④ 정사각형이 줄어듦
# 출전하지 않은 대회: 0%(실력)가 아니라 '자료 없음'으로 처리해 평소 수준 평균에서 뺀다
ABSENT = {
    (1920, "Summer"): ["GER"],                                         # 1차 대전 패전국 초청 제외
    (1924, "Summer"): ["GER"], (1924, "Winter"): ["GER"],
    (1948, "Summer"): ["GER", "JPN"], (1948, "Winter"): ["GER", "JPN"],  # 2차 대전 패전국 초청 제외
    (1980, "Summer"): ["USA", "GER", "CAN", "JPN", "KOR", "NOR", "CHN"],  # 모스크바 보이콧
    (1984, "Summer"): ["RUS"],                                         # LA 보이콧(소련)
    (2024, "Summer"): ["RUS"],                                         # 러시아 출전 정지
    (2026, "Winter"): ["RUS"],                                         # 러시아 출전 정지
}
# 아직 열리지 않은 개최 확정 대회 (hosts.csv에는 이미 열린 대회만 있음)
UPCOMING = [
    (2028, "Summer", "로스앤젤레스", "USA", "미국"),
    (2030, "Winter", "프렌치 알프스", "FRA", "프랑스"),
    (2032, "Summer", "브리즈번", "AUS", "호주"),
    (2034, "Winter", "솔트레이크시티", "USA", "미국"),
]
PRED_LEVEL = 0.80   # 예측구간 수준: 한 나라 한 대회의 예측이라 폭이 넓어 80%를 쓴다

# ── 0. 화면 단계와 애니메이션 (CSS) ──────────────────────────
# 단계마다 st.container(key="step-…")를 쓰면 Streamlit이 'st-key-step-…' 클래스를 붙인다 → 한 화면 높이로 만든다.
# 애니메이션: 첫 화면은 열릴 때 차례로 떠오르고, 그 아래는 스크롤해서 화면에 들어올 때 제목·차트·카드만 떠오른다.
# 스크롤 연동(animation-timeline)을 지원하지 않는 브라우저는 애니메이션 없이 그대로 보이고,
# '동작 줄이기'를 켠 사용자에게는 모두 끈다.
st.html(f"""
<style>
:root {{ --ink: {INK}; --muted: {MUTED}; --ease: cubic-bezier(.2,.7,.2,1); }}
[data-testid="stMainBlockContainer"] {{ padding-top: 1rem; padding-bottom: 2rem; }}
/* 한글을 낱말 중간에서 줄바꿈하지 않음 (Streamlit이 글자 요소마다 break-word를 직접 걸어 두어 요소에 덮어씀) */
:is([data-testid="stMarkdownContainer"], [data-testid="stHeading"], [data-testid="stCaptionContainer"],
    [data-testid="stMetric"], [data-testid="stWidgetLabel"]) :is(p, li, h1, h2, h3, h4, span, label, div) {{
  word-break: keep-all !important; overflow-wrap: anywhere; }}

[class*="st-key-step-"] {{ min-height: calc(100vh - 3.75rem); justify-content: center; padding: 3rem 0; }}
[class*="st-key-step-"]:not(.st-key-step-hero) {{ border-top: 1px solid rgba(128,128,128,.2); }}
.eyebrow {{ color: var(--ink); font-weight: 700; font-size: .85rem; letter-spacing: .08em; margin: 0 0 -.75rem; }}

.hero-eyebrow {{ color: var(--ink); font-weight: 700; letter-spacing: .12em; font-size: .9rem; margin: 0; }}
.hero-title {{ font-size: clamp(2.4rem, 6vw, 4.2rem) !important; line-height: 1.15 !important;
              font-weight: 800 !important; padding: .4rem 0 1rem !important; }}
.hero-lead {{ font-size: 1.2rem; line-height: 1.7; max-width: 46rem; margin: 0; }}
.scroll-hint {{ display: flex; flex-direction: column; align-items: center; gap: .25rem;
               margin-top: 2.5rem; color: var(--muted); font-size: .9rem; }}
.scroll-hint span {{ font-size: 1.5rem; animation: bob 1.8s ease-in-out infinite; }}
.st-key-step-hero > * {{ animation: rise .8s var(--ease) both; }}
.st-key-step-hero > *:nth-child(2) {{ animation-delay: .12s; }}
.st-key-step-hero > *:nth-child(3) {{ animation-delay: .24s; }}
.st-key-step-hero > *:nth-child(4) {{ animation-delay: .36s; }}
.st-key-step-hero > *:nth-child(5) {{ animation-delay: .48s; }}
.st-key-step-hero > *:nth-child(n+6) {{ animation-delay: .7s; }}

.finding-num {{ font-size: 2.2rem; font-weight: 800; color: var(--ink); line-height: 1; margin: 0; }}
[class*="st-key-finding-"] {{ border-left: 4px solid var(--ink) !important; }}

@keyframes rise {{ from {{ opacity: 0; transform: translateY(28px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes rise-scale {{ from {{ opacity: 0; transform: translateY(40px) scale(.97); }} to {{ opacity: 1; transform: none; }} }}
@keyframes bob {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(7px); }} }}
@keyframes fade {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}

@supports (animation-timeline: view()) {{
  /* 차트 상자가 떠오른 뒤, 그 안의 점·선·막대가 이어서 서서히 나타남 */
  [data-testid="stPlotlyChart"] {{ view-timeline-name: --chart; }}
  [data-testid="stPlotlyChart"] :is(.scatterlayer, .barlayer) {{
    animation: fade linear both; animation-timeline: --chart; animation-range: entry 60% cover 40%;
  }}
  [class*="st-key-step-"]:not(.st-key-step-hero) :is(h2, .eyebrow),
  [data-testid="stMetric"], [class*="st-key-finding-"] {{
    animation: rise linear both; animation-timeline: view(); animation-range: entry 0% cover 22%;
  }}
  [data-testid="stPlotlyChart"] {{
    animation: rise-scale linear both; animation-timeline: view(); animation-range: entry 0% cover 28%;
  }}
  /* 옆으로 놓인 카드는 조금씩 늦게 → 왼쪽부터 차례로 떠오름 */
  [data-testid="stColumn"]:nth-child(2) :is([data-testid="stMetric"], [class*="st-key-finding-"]) {{ animation-range: entry 6% cover 28%; }}
  [data-testid="stColumn"]:nth-child(3) [data-testid="stMetric"] {{ animation-range: entry 12% cover 34%; }}
  [data-testid="stColumn"]:nth-child(4) [data-testid="stMetric"] {{ animation-range: entry 18% cover 40%; }}
}}
@media (prefers-reduced-motion: reduce) {{ *, *::before, *::after {{ animation: none !important; }} }}
</style>
""")


def step_head(eyebrow, title):
    """단계 제목: 작은 안내 문구 + 큰 제목 (제목 옆 링크 아이콘은 뺌)."""
    st.markdown(f"<p class='eyebrow'>{eyebrow}</p>", unsafe_allow_html=True)
    st.header(title, anchor=False)


def finish_layout(fig, height=CHART_H, legend=True):
    """차트 공통: 높이·여백 고정, 범례는 그림 위 가로 한 줄."""
    fig.update_layout(height=height, margin=dict(t=MARGIN_T, b=MARGIN_B))
    if legend:
        fig.update_layout(legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom", title=None))
    return fig


# ── 1. 자료와 캐시 ───────────────────────────────────────────
@st.cache_data
def load_data():
    medals = pd.read_csv(HERE / "medals.csv", encoding="utf-8-sig")
    hosts = pd.read_csv(HERE / "hosts.csv", encoding="utf-8-sig")
    return medals, hosts


@st.cache_data
def share_table(medals, hosts, col):
    """대회마다 나라별 메달 점유율(%). 출전했는데 메달이 없으면 0, 불참이면 빈 값(NaN)."""
    held = hosts[["year", "season"]]
    total = medals.groupby(["year", "season"])[col].sum().rename("all")
    m = medals.merge(total, on=["year", "season"])
    m["share"] = m[col] / m["all"] * 100
    teams = hosts["team"].unique()
    grid = held.merge(pd.DataFrame({"team": teams}), how="cross")
    out = grid.merge(m[["year", "season", "team", col, "share"]], on=["year", "season", "team"], how="left")
    out = out.fillna({col: 0, "share": 0.0})
    for (y, season), teams_out in ABSENT.items():
        hit = (out["year"] == y) & (out["season"] == season) & out["team"].isin(teams_out)
        out.loc[hit, [col, "share"]] = np.nan
    return out


@st.cache_data
def host_effects(shares, hosts, k, drop_boycott):
    """개최 대회 점유율과 '평소 수준'(같은 계절 직전 k개 + 직후 k개 대회 평균)을 비교."""
    h = hosts.copy()
    if drop_boycott:
        h = h[[(y, s) not in BOYCOTT for y, s in zip(h["year"], h["season"])]]
    s = shares.set_index(["year", "season", "team"])["share"]
    rows = []
    for season, g in h.groupby("season"):
        years = sorted(g["year"])
        for _, r in g.iterrows():
            i = years.index(r["year"])
            before, after = years[max(0, i - k):i], years[i + 1:i + 1 + k]
            b_vals = [s[(y, season, r["team"])] for y in before]
            a_vals = [s[(y, season, r["team"])] for y in after]
            b_vals = [v for v in b_vals if pd.notna(v)]      # 불참 대회는 평균에서 뺌
            a_vals = [v for v in a_vals if pd.notna(v)]
            base = b_vals + a_vals
            rows.append({
                "year": r["year"], "season": SEASON_KO[season], "city": r["city"],
                "team": r["team"], "country_ko": r["country_ko"],
                "host_share": s[(r["year"], season, r["team"])],
                "base_share": np.mean(base) if base else np.nan,
                "n_before": len(b_vals), "n_after": len(a_vals),
            })
    e = pd.DataFrame(rows)
    e["diff"] = e["host_share"] - e["base_share"]
    e["ratio"] = np.where(e["base_share"] > 0, e["host_share"] / e["base_share"], np.nan)
    e["label"] = e["year"].astype(str) + " " + e["city"] + " · " + e["country_ko"]
    return e


# 다시 읽기 버튼은 자료를 읽기 전에 처리해야 바로 반영된다. 사이드바에서는 다른 위젯 아래에 보이도록 자리를 먼저 잡아 둔다.
controls = st.sidebar.container()
if st.sidebar.button("CSV 다시 읽기", help="medals.csv·hosts.csv를 고친 뒤 누르면 캐시를 비우고 새로 읽습니다."):
    st.cache_data.clear()
medals, hosts = load_data()

# ── 2. 위젯 ─────────────────────────────────────────────────
with controls:
    st.header("조건 고르기")
    years = st.slider("기간 (개최 연도)", 1896, 2026, (1948, 2026), step=2)
    season_pick = st.selectbox("대회 종류", ["하계 + 동계", "하계", "동계"])
    medal_pick = st.selectbox("메달 기준", ["전체 메달", "금메달"])
    col = "total" if medal_pick == "전체 메달" else "gold"
    k = st.slider("‘평소 수준’으로 쓸 앞뒤 대회 수", 1, 3, 2,
                  help="2이면 직전 2개 + 직후 2개 대회 점유율의 평균과 비교합니다.")
    drop_boycott = st.checkbox("보이콧 대회(1980 모스크바·1984 LA) 빼기", value=False,
                               help="두 대회를 개최 대회와 비교 대상 모두에서 뺍니다. 경쟁국이 빠져 개최국 점유율이 부풀었을 수 있기 때문입니다.")
    country_names = sorted(hosts["country_ko"].unique())
    focus = st.selectbox("중심 국가", [NO_FOCUS] + country_names, index=1 + country_names.index("대한민국"),
                         help="‘(없음)’을 고르면 차트 ①이 전체 개최 대회의 연도별 효과로 바뀝니다.")
    has_focus = focus != NO_FOCUS
    effect_view = st.selectbox("효과 표시 방식", ["배수 (개최 ÷ 평소)", "차이 (개최 − 평소, %p)"],
                               help="중심 국가를 ‘(없음)’으로 고르면 차트 ①(연도별 개최국 효과)에도 같은 방식이 적용됩니다.")

# ── 3. 첫 화면: 제목·질문·출처·집계 기준 ─────────────────────
with st.container(key="step-hero"):
    st.markdown("<p class='hero-eyebrow'>OLYMPIC HOST EFFECT · 1896–2026</p>", unsafe_allow_html=True)
    st.markdown("<h1 class='hero-title'>올림픽 개최국은<br>메달을 얼마나 더 딸까?</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='hero-lead'>올림픽을 개최한 국가는 <b>평소보다 메달을 더 많이</b> 땄을까? ",
        unsafe_allow_html=True,
    )
    st.caption(
        "출처: Olympedia 메달표(olympicAthletes R 패키지, CC BY 4.0) · 1896~2026년 하계 30회·동계 25회 · "
        "왼쪽 사이드바에서 조건을 바꾸면 아래 모든 화면이 함께 바뀝니다."
    )
    with st.expander("데이터와 집계 기준"):
        st.markdown(
            "**출처**\n"
            "- 메달표: `medal_table_summary.csv` — 대회×국가(NOC)별 금·은·동 수\n"
            "- 대회 정보: `edition_metadata.csv` — 개최 도시, 개최국, 메달 종목 수\n"
            "- 저장소: github.com/moderndive/olympicAthletes 의 `data-raw` 폴더에서 CSV를 직접 내려받음 (라이선스 CC BY 4.0)\n"
            "**집계 기준** (make_data.py)\n"
            "1. 1906년 중간 대회(IOC 비공인) 제외, 1916·1940·1944년 취소 대회 제외\n"
            "2. 1956년 스톡홀름 승마(호주 검역 문제로 분리 개최)는 1956년 하계에 합치고, 개최국은 호주로 봄\n"
            "3. 나라 계보 통합: 독일 = 독일+서독, 러시아 = 소련+독립국가연합(1992)+러시아+ROC (동독과 중립 선수단은 따로)\n"
            "4. 출전했는데 메달을 못 딴 대회는 점유율 0%. 불참한 대회는 0이 아니라 ‘자료 없음’으로 처리해 평균에서 뺌: "
            "1920·1924·1948년 독일, 1948년 일본(전쟁 패전국으로 초청 제외), 1980 모스크바(미국·서독·캐나다·일본·한국·노르웨이·중국 보이콧), "
            "1984 LA(소련 보이콧), 2024 파리·2026 밀라노(러시아 출전 정지). 메달표에는 메달을 딴 나라만 있으므로, 이 목록 밖의 빈칸은 ‘출전했지만 메달 0개’로 봄\n"
            "5. **평소 수준** = 같은 계절 올림픽 중 직전 k개와 직후 k개 대회의 점유율 평균. 그중 불참한 대회는 빼고 평균 (2024·2026은 직후 대회가 아직 없어 직전 대회만 씀)\n"
            "6. **개최국 효과(배수)** = 개최 대회 점유율 ÷ 평소 수준, **차이** = 개최 대회 점유율 − 평소 수준 (%p)"
        )
    st.markdown("<div class='scroll-hint'>아래로 내려 보기<span>↓</span></div>", unsafe_allow_html=True)

# ── 4. 필터와 빈 선택 ────────────────────────────────────────
shares = share_table(medals, hosts, col)
effects = host_effects(shares, hosts, k, drop_boycott)
view = effects[effects["year"].between(*years)]
if season_pick != "하계 + 동계":
    view = view[view["season"] == season_pick]

if view.empty:
    st.warning("조건에 맞는 개최 대회가 없습니다. 기간을 넓히거나 대회 종류를 바꿔 보세요.")
    st.stop()

# ── 5. 요약: 지표 카드 + 통계 ────────────────────────────────
valid = view[view["ratio"] > 0].copy()
valid["log_ratio"] = np.log(valid["ratio"])
n_up = int((view["diff"] > 0).sum())
med_diff = view["diff"].median()
best = view.loc[view["diff"].idxmax()]      # 점유율이 가장 많이 오른 대회
geo, ci_lo, ci_hi = np.nan, np.nan, np.nan
if len(valid) >= 3:
    model = smf.ols("log_ratio ~ 1", data=valid).fit()   # 로그 배수의 평균 = 기하평균 배수
    geo = np.exp(model.params["Intercept"])
    ci_lo, ci_hi = np.exp(model.conf_int().loc["Intercept"])
mine = view[view["country_ko"] == focus]

with st.container(key="step-summary"):
    step_head("한눈에 보기", f"{years[0]}~{years[1]}년 {season_pick} 개최 {len(view)}회 요약")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("평소보다 점유율이 오른 개최국", f"{n_up} / {len(view)}회", border=True)
    c2.metric("개최국 효과 (기하평균)", f"{geo:.2f}배" if np.isfinite(geo) else "계산 불가", border=True,
              help="개최 대회 점유율 ÷ 평소 수준을 로그로 평균한 뒤 되돌린 값. 소수의 극단값에 덜 흔들립니다.")
    c3.metric("점유율 차이 (중앙값)", f"{med_diff:+.2f}%p", border=True)
    if not has_focus:
        c4.metric("점유율이 가장 많이 오른 대회", f"{best['city']} {best['year']}", border=True,
                  delta=f"{best['diff']:+.2f}%p ({best['country_ko']})")
    elif mine.empty:
        c4.metric(f"{focus} 개최 효과", "기간 안에 개최 없음", border=True)
    else:
        my_geo = np.exp(np.log(mine.loc[mine["ratio"] > 0, "ratio"]).mean())
        c4.metric(f"{focus} 개최 효과", f"{my_geo:.2f}배", border=True, delta=f"{mine['diff'].mean():+.2f}%p",
                  help=f"{focus}가 기간 안에 개최한 {len(mine)}회의 기하평균 배수와 평균 점유율 차이")
    if np.isfinite(geo):
        st.caption(f"기하평균 배수의 95% 신뢰구간: {ci_lo:.2f}~{ci_hi:.2f}배 (n = {len(valid)}, statsmodels OLS로 로그 배수의 평균을 추정). "
                   "1배는 ‘개최해도 평소와 같다’는 뜻입니다.")

# ── 6. 차트 ① 중심 국가 추이 (또는 연도별 효과) ──────────────
is_ratio = effect_view.startswith("배수")
xcol = "ratio" if is_ratio else "diff"
ref = 1 if is_ratio else 0          # '효과 없음' 기준선

with st.container(key="step-chart1"):
    if not has_focus:
        # 중심 국가가 없으면: 개최 대회마다의 효과를 연도 순으로 → 개최국 효과가 시대에 따라 달라졌는지
        tl = view.dropna(subset=[xcol])
        if is_ratio:
            tl = tl[tl["ratio"] > 0]       # 로그 눈금에 0배는 그릴 수 없음
        step_head("시대에 따라 효과가 달라졌을까?", f"① 연도별 개최국 효과 ({medal_pick}, {len(tl)}회)")
        fig1 = px.scatter(
            tl, x="year", y=xcol, color="season", color_discrete_map=SEASON_COLORS,
            hover_name="label", hover_data={"year": False, "season": False, xcol: ":.2f"},
            labels={"year": "개최 연도", "season": "대회",
                    xcol: "개최 대회 점유율 ÷ 평소 수준 (배, 로그 눈금)" if is_ratio else "개최 대회 점유율 − 평소 수준 (%p)"},
        )
        fig1.update_traces(marker=dict(size=11, line=dict(width=1, color=EDGE)))
        # add_hline의 글자는 로그 축에서 위치가 틀어지므로(Plotly 버그) 글자를 따로 단다. 로그 축 좌표는 log10 값
        fig1.add_hline(y=ref, line_dash="dash", line_color=MUTED)
        fig1.add_annotation(x=1, xref="paper", xanchor="right", y=np.log10(ref) if is_ratio else ref,
                            yanchor="top", text="효과 없음", showarrow=False, font=dict(color=MUTED))
        if is_ratio:
            fig1.update_yaxes(type="log", tickvals=LOG_TICKS)
        st.plotly_chart(finish_layout(fig1), width="stretch")
        st.caption("점 하나가 개최 대회 하나입니다. 기준선보다 위면 평소보다 많이 땄다는 뜻이고, "
                   "오른쪽(최근)으로 갈수록 점이 기준선에 가까워지면 개최국 효과가 줄어든 것입니다. 효과 표시 방식은 차트 ②와 같습니다.")
    else:
        step_head("중심 국가의 메달 추이", f"① {focus}의 {medal_pick} 점유율 추이 (★ = 직접 개최)")
        team = hosts.loc[hosts["country_ko"] == focus, "team"].iloc[0]
        line = shares[(shares["team"] == team) & shares["year"].between(*years)].copy()
        line["season"] = line["season"].map(SEASON_KO)
        if season_pick != "하계 + 동계":
            line = line[line["season"] == season_pick]
        line = line.sort_values("year")
        fig1 = px.line(
            line, x="year", y="share", color="season", markers=True, color_discrete_map=SEASON_COLORS,
            labels={"year": "개최 연도", "share": f"{medal_pick} 점유율 (%)", "season": "대회"},
        )
        fig1.update_traces(line_width=2, marker_size=7)
        hosted = line.merge(hosts.assign(season=hosts["season"].map(SEASON_KO))[["year", "season", "team", "city"]],
                            on=["year", "season", "team"])
        fig1.add_trace(go.Scatter(
            x=hosted["year"], y=hosted["share"], mode="markers+text", name="개최 대회",
            marker=dict(symbol="star", size=15, color=INK, line=dict(width=1, color=EDGE)),
            text=hosted["city"], textposition="top center", hovertemplate="%{text} %{x}<br>%{y:.2f}%",
        ))
        fig1.update_yaxes(rangemode="tozero")
        fig1.update_layout(hovermode="closest")
        st.plotly_chart(finish_layout(fig1), width="stretch")
        st.caption("선이 끊긴 곳은 불참한 대회입니다(예: 한국 1980 하계)")

# ── 7. 차트 ② 개최 대회별 효과 ───────────────────────────────
with st.container(key="step-chart2"):
    bar = view.dropna(subset=[xcol]).sort_values(xcol).copy()   # 막대는 '효과 없음' 선(ref)에서 출발
    step_head("개최 대회마다 얼마나 더 땄나", f"② 개최 대회별 개최국 효과 ({len(bar)}회)")
    # 중심 국가는 축 이름을 굵게 + 강조색(보라, 라이트·다크 모두에서 보이는 중간 밝기)으로 표시
    bar["name"] = np.where(bar["country_ko"] == focus,
                           f"<b><span style='color:{INK}'>▶ " + bar["label"] + "</span></b>", bar["label"])
    fig2 = go.Figure()
    for season, color in SEASON_COLORS.items():
        part = bar[bar["season"] == season]
        if part.empty:
            continue
        fig2.add_trace(go.Bar(
            y=part["name"], x=part[xcol] - ref, base=ref, orientation="h", name=season,
            # 중심 국가가 있으면 나머지를 흐리게, 없으면 모두 진하게
            marker=dict(color=color, opacity=np.where(part["country_ko"] == focus, 1.0, 0.55 if has_focus else 0.9),
                        line=dict(width=np.where(part["country_ko"] == focus, 2, 0), color=INK)),
            customdata=np.stack([part["host_share"], part["base_share"], part[xcol]], axis=1),
            hovertemplate="%{y}<br>개최 대회 %{customdata[0]:.2f}% · 평소 %{customdata[1]:.2f}%"
                          + ("<br>%{customdata[2]:.2f}배" if is_ratio else "<br>%{customdata[2]:+.2f}%p")
                          + "<extra></extra>",
        ))
    # add_vline의 글자는 로그 축에서 위치가 틀어지므로(Plotly 버그) 글자를 따로 단다. 로그 축 좌표는 log10 값
    fig2.add_vline(x=ref, line_dash="dash", line_color=MUTED)
    fig2.add_annotation(x=np.log10(ref) if is_ratio else ref, y=1, yref="paper", yanchor="bottom",
                        text="효과 없음", showarrow=False, font=dict(color=MUTED))
    fig2.update_layout(
        barmode="overlay", bargap=0.25,
        xaxis=dict(title="개최 대회 점유율 ÷ 평소 수준 (배, 로그 눈금)" if is_ratio
                   else "개최 대회 점유율 − 평소 수준 (%p)", type="log" if is_ratio else "linear",
                   tickvals=LOG_TICKS if is_ratio else None),
        yaxis=dict(categoryorder="array", categoryarray=bar["name"].tolist(), tickfont=dict(size=11)),
    )
    st.plotly_chart(finish_layout(fig2, height=max(CHART_H, 20 * len(bar))), width="stretch")
    st.caption("평소 점유율이 작은 나라는 배수가 크게 나오므로(예: 1972 삿포로·일본), ‘차이(%p)’ 보기와 함께 확인하세요.")

# ── 8. 차트 ③ 개최 전후 패턴 ─────────────────────────────────
rel_rows = []
sidx = shares.set_index(["year", "season", "team"])["share"]
for _, r in view.iterrows():
    season_en = SEASON_EN[r["season"]]
    yrs = sorted(hosts.loc[hosts["season"] == season_en, "year"])
    i = yrs.index(r["year"])
    around = {rel: yrs[i + rel] for rel in range(-3, 4) if 0 <= i + rel < len(yrs)}
    others = [sidx[(y, season_en, r["team"])] for rel, y in around.items() if rel != 0]
    others = [v for v in others if pd.notna(v)]
    norm = np.mean(others) if others else np.nan
    if not norm or not np.isfinite(norm):
        continue
    for rel, y in around.items():
        v = sidx[(y, season_en, r["team"])]
        if pd.notna(v):
            rel_rows.append({"rel": rel, "value": v / norm})
stat = pd.DataFrame(rel_rows).groupby("rel")["value"].agg(["mean", "median", "std", "count"]).reset_index()
# 위치마다 사례 수가 달라 자유도(n−1)에 맞는 t 임계값을 쓴다 (n=1이면 구간 없음)
tcrit = scipy_stats.t.ppf(0.975, (stat["count"] - 1).clip(lower=1))
stat["ci"] = np.where(stat["count"] > 1, tcrit * stat["std"] / np.sqrt(stat["count"]), 0)

with st.container(key="step-chart3"):
    step_head("효과는 개최 전후에도 이어질까?", "③ 개최 전후 대회의 점유율 (산술평균·중앙값)")
    fig3 = go.Figure(go.Scatter(
        x=stat["rel"], y=stat["mean"], mode="lines+markers", name="산술평균 (95% 신뢰구간)",
        error_y=dict(type="data", array=stat["ci"], color=MUTED, thickness=1.5, width=6),
        line=dict(color=INK, width=2.5), marker=dict(size=10, color=INK, line=dict(width=1.5, color=EDGE)),
        customdata=stat["count"], hovertemplate="%{x}번째 대회 산술평균: %{y:.2f}배 (n=%{customdata})<extra></extra>",
    ))
    fig3.add_trace(go.Scatter(
        x=stat["rel"], y=stat["median"], mode="lines+markers", name="중앙값",
        line=dict(color=MUTED, width=2, dash="dot"), marker=dict(size=8, color=MUTED, symbol="diamond"),
        customdata=stat["count"], hovertemplate="%{x}번째 대회 중앙값: %{y:.2f}배 (n=%{customdata})<extra></extra>",
    ))
    fig3.add_hline(y=1, line_dash="dash", line_color=MUTED, annotation_text="앞뒤 대회 평균",
                   annotation_position="bottom right", annotation_font_color=MUTED)
    fig3.update_layout(
        xaxis=dict(title="개최 대회 기준 몇 번째 대회인가 (개최 대회 = 0)", tickmode="linear", dtick=1),
        yaxis=dict(title="앞뒤 3개 대회 평균 대비 (배)", rangemode="tozero"),
    )
    st.plotly_chart(finish_layout(fig3), width="stretch")
    st.caption("산술평균은 평소 점유율이 작은 나라의 큰 배수에 끌려 올라가므로, "
               "두 선의 간격이 크면 소수의 극단값이 평균을 키운 것입니다. "
               "오차 막대는 산술평균의 95% 신뢰구간입니다.")

# ── 9. 차트 ④ 평소 수준 대 개최 대회 ────────────────────────
with st.container(key="step-chart4"):
    step_head("평소 실력과 상관없이 오를까?", "④ 평소 수준 대비 개최 대회 점유율")
    chart_col, text_col = st.columns([3, 2], gap="large", vertical_alignment="center")
    with chart_col:
        fig4 = px.scatter(
            view, x="base_share", y="host_share", color="season", color_discrete_map=SEASON_COLORS,
            hover_name="label", hover_data={"base_share": ":.2f", "host_share": ":.2f", "season": False},
            labels={"base_share": "평소 수준 점유율 (%, 로그 눈금)", "host_share": "개최 대회 점유율 (%, 로그 눈금)", "season": "대회"},
        )
        fig4.update_traces(marker=dict(size=11, line=dict(width=1, color=EDGE)))
        pos = view[(view["base_share"] > 0) & (view["host_share"] > 0)]
        lo = float(min(pos["base_share"].min(), pos["host_share"].min())) * 0.7
        hi = float(max(pos["base_share"].max(), pos["host_share"].max())) * 1.4
        fig4.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="평소와 같음",
                                  line=dict(color=MUTED, dash="dash"), hoverinfo="skip"))
        pts = mine[(mine["base_share"] > 0) & (mine["host_share"] > 0)].sort_values("host_share")
        if not pts.empty:
            # 선택 국가 점에 고리를 둘러 강조
            fig4.add_trace(go.Scatter(x=pts["base_share"], y=pts["host_share"], mode="markers", showlegend=False,
                                      marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(width=2, color=INK)),
                                      hoverinfo="skip"))
            # 이름표는 점 왼쪽(점선 위 빈 공간)에 두고, 각 이름표를 자기 점과 거의 같은 높이에 놓는다.
            # → 지시선이 짧고 거의 수평이라 어느 점을 가리키는지 바로 보인다.
            # 높이가 겹치면 최소 간격만큼 위아래로 밀어 이름표끼리 겹치지 않게 하고, 그림 안에 가둔다.
            L, H = np.log10(lo), np.log10(hi)
            span = H - L
            # 이름표 간격을 픽셀로 정한 뒤 로그 단위로 바꾼다: 이름표 높이(글자 11px + 안쪽 여백 + 테두리 ≈ 22px) + 여유 6px
            gap = 28 / PLOT_PX * span
            px_ = np.log10(pts["base_share"].to_numpy())
            py_ = np.log10(pts["host_share"].to_numpy())         # 이미 낮은 점 → 높은 점 순
            ys = py_.copy()
            for i in range(1, len(ys)):                          # 아래에서 위로 밀어 올리기
                ys[i] = max(ys[i], ys[i - 1] + gap)
            top_y = H - 0.6 * gap
            if len(ys) and ys[-1] > top_y:                       # 위로 넘치면 위에서 아래로 다시 밀기
                ys[-1] = top_y
                for i in range(len(ys) - 2, -1, -1):
                    ys[i] = min(ys[i], ys[i + 1] - gap)
            # 이름표 오른쪽 끝을 한 줄로 맞추되, 이름표 높이대에 있는 모든 점(다른 나라 포함)보다 왼쪽에 둔다
            ax_all, ay_all = np.log10(pos["base_share"].to_numpy()), np.log10(pos["host_share"].to_numpy())
            band = (ay_all > ys.min() - gap) & (ay_all < ys.max() + gap)
            label_x = min(px_.min(), ax_all[band].min()) - 0.05 * span
            label_x = max(label_x, L + 0.4 * span)               # 왼쪽 밖으로 나가지 않게
            for (_, r), x0, y0, yl in zip(pts.iterrows(), px_, py_, ys):
                fig4.add_annotation(
                    x=x0, y=y0, ax=label_x, ay=yl, axref="x", ayref="y",
                    text=f"{r['city']} {r['year']}", xanchor="right", font=dict(size=11),
                    bgcolor=LABEL_BG, bordercolor=INK, borderwidth=1, borderpad=3,
                    showarrow=True, arrowhead=0, arrowwidth=1.2, arrowcolor=INK, standoff=11,
                )
        fig4.update_xaxes(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=SHARE_TICKS)
        fig4.update_yaxes(type="log", range=[np.log10(lo), np.log10(hi)], tickvals=SHARE_TICKS, scaleanchor="x", scaleratio=1)
        st.plotly_chart(finish_layout(fig4), width="stretch")
    with text_col:
        st.markdown(
            "**점선 위 = 평소보다 많이 땀.**\n\n"
            "두 축 모두 같은 로그 눈금이라 점선이 정확히 45도입니다. 점선 위로 떨어진 거리가 배수이므로, "
            "점유율 0.2%인 나라와 20%인 나라를 같은 그림에서 비교할 수 있습니다."
        )
        st.caption("점유율 0%인 경우는 로그 눈금에 그릴 수 없어 뺐습니다.")

# ── 10. 다음 개최국 예측 ────────────────────────────────────
n_base = 2 * k   # '평소 수준'과 같은 개수(앞뒤 k개씩 = 2k개)의 최근 대회를 쓴다
totals = medals.groupby(["year", "season"])[col].sum()
eff_pool = effects[effects["year"].between(*years) & (effects["ratio"] > 0)]
pred_rows = []
for p_year, p_season, p_city, p_team, p_ko in UPCOMING:
    logs = np.log(eff_pool.loc[eff_pool["season"] == SEASON_KO[p_season], "ratio"])
    hosted_yrs = hosts.loc[(hosts["team"] == p_team) & (hosts["season"] == p_season), "year"]
    past = shares[(shares["team"] == p_team) & (shares["season"] == p_season) & ~shares["year"].isin(hosted_yrs)]
    past = past.dropna(subset=["share"]).sort_values("year").tail(n_base)
    if len(logs) < 3 or past.empty:
        continue
    last_year = totals.loc[(slice(None), p_season)].index.max()
    base_cnt = past["share"].mean() / 100 * totals.loc[(last_year, p_season)]
    # 한 대회의 예측구간: 로그 배수의 평균 ± t × 표준편차 × √(1 + 1/n)
    n_e, m_e, s_e = len(logs), logs.mean(), logs.std(ddof=1)
    half = scipy_stats.t.ppf(0.5 + PRED_LEVEL / 2, n_e - 1) * s_e * np.sqrt(1 + 1 / n_e)
    pred_rows.append({
        "label": f"{p_year} {p_city} · {p_ko}", "season": SEASON_KO[p_season],
        "base": base_cnt, "pred": base_cnt * np.exp(m_e),
        "lo": base_cnt * np.exp(m_e - half), "hi": base_cnt * np.exp(m_e + half),
        "effect": np.exp(m_e), "n_eff": n_e, "base_from": f"{past['year'].min()}~{past['year'].max()}",
    })

with st.container(key="step-predict"):
    step_head("다음 개최국", "다음 개최국은 메달을 몇 개 딸까?")
    st.markdown(
        f"**예측 방법.** 예상 메달 수 = **최근 평소 점유율** × **개최국 효과** × **대회 전체 메달 수**\n"
        f"- 최근 평소 점유율: 같은 계절에서 그 나라가 개최하지 않은 최근 {n_base}개 대회의 {medal_pick} 점유율 평균 (사이드바 k의 2배)\n"
        f"- 개최국 효과: 선택한 기간({years[0]}~{years[1]}년)의 같은 계절 개최 대회 배수의 기하평균 (위 요약 카드와 같은 방식)\n"
        f"- 대회 전체 메달 수: 같은 계절 가장 최근 대회의 값을 그대로 씀"
    )
    if not pred_rows:
        st.info("선택한 기간에 같은 계절 개최 대회가 3회 미만이라 예측할 수 없습니다. 기간을 넓혀 보세요.")
    else:
        pred = pd.DataFrame(pred_rows)
        for c, (_, r) in zip(st.columns(len(pred)), pred.iterrows()):
            c.metric(r["label"], f"약 {r['pred']:.0f}개", border=True, delta=f"{r['pred'] - r['base']:+.0f}개 (개최 안 할 때 대비)",
                     help=f"평소 수준 {r['base']:.0f}개 ({r['base_from']}년 대회 기준) × 개최국 효과 {r['effect']:.2f}배 "
                          f"({r['season']} {r['n_eff']}회 기하평균)")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=pred["base"], y=pred["label"], mode="markers", name="개최하지 않을 때 (평소 수준)",
            marker=dict(size=11, symbol="circle-open", color=MUTED, line=dict(width=2)),
            hovertemplate="%{y}<br>평소 수준 %{x:.0f}개<extra></extra>",
        ))
        fig5.add_trace(go.Scatter(
            x=pred["pred"], y=pred["label"], mode="markers", name=f"개최국 예측 ({PRED_LEVEL:.0%} 예측구간)",
            marker=dict(size=12, color=INK, line=dict(width=1.5, color=EDGE)),
            error_x=dict(type="data", symmetric=False, array=pred["hi"] - pred["pred"],
                         arrayminus=pred["pred"] - pred["lo"], color=INK, thickness=1.5, width=6),
            customdata=np.stack([pred["lo"], pred["hi"]], axis=1),
            hovertemplate="%{y}<br>예측 %{x:.0f}개 (%{customdata[0]:.0f}~%{customdata[1]:.0f}개)<extra></extra>",
        ))
        fig5.update_layout(
            xaxis=dict(title=f"{medal_pick} 수 (개)", rangemode="tozero"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(finish_layout(fig5, height=140 + 70 * len(pred)), width="stretch")
        st.caption(
            f"⑤ 막대는 {PRED_LEVEL:.0%} 예측구간입니다. 평균 효과의 신뢰구간이 아니라 ‘한 나라가 한 대회에서 얻을 효과’의 범위라 넓습니다. "
            "종목 수 변화(예: LA 2028의 신규 종목), 선수단 규모, 경쟁국 출전 여부는 반영하지 않은 단순 추정입니다. "
            "기간 슬라이더를 최근으로 좁히면 최근 개최국 효과로 예측이 바뀝니다."
        )

# ── 11. 결론: 발견 하나 = 카드 하나 ──────────────────────────
pre = stat.loc[stat["rel"] == -1, "mean"]
post = stat.loc[stat["rel"] == 1, "mean"]
peak = stat.loc[stat["rel"] == 0, "mean"]
peak_med = stat.loc[stat["rel"] == 0, "median"]
findings = [(
    "개최국은 대부분 평소보다 메달을 더 땄다",
    f"{years[0]}~{years[1]}년 {season_pick} 개최 {len(view)}회 중 **{n_up}회**({n_up / len(view) * 100:.0f}%)에서 "
    f"{medal_pick} 점유율이 평소 수준보다 높았다.",
)]
if np.isfinite(geo):
    findings.append((
        f"크기는 평균 약 {geo:.1f}배다",
        f"개최 대회 점유율은 평소 수준의 기하평균 **{geo:.2f}배**였다(95% 신뢰구간 {ci_lo:.2f}~{ci_hi:.2f}배"
        + ("로 1배를 넘는다" if ci_lo > 1 else "로 1배를 포함한다")
        + f"). 점유율로는 중앙값 **{med_diff:+.2f}%p**에 해당한다.",
    ))
if len(peak) and len(pre) and len(post):
    findings.append((
        "효과는 개최 당해에 가장 크고, 앞뒤 대회에는 일부만 남는다",
        f"앞뒤 3개 대회 평균을 1로 놓으면 (산술평균 기준) 직전 대회 {pre.iloc[0]:.2f}배 → **개최 대회 {peak.iloc[0]:.2f}배** → "
        f"직후 대회 {post.iloc[0]:.2f}배였다. 중앙값으로는 개최 대회 {peak_med.iloc[0]:.2f}배다.",
    ))
findings.append((
    "가장 크게 오른 대회",
    f"**{best['label']}** 대회로, 평소 {best['base_share']:.2f}%에서 {best['host_share']:.2f}%로 "
    f"**{best['diff']:+.2f}%p** 올랐다.",
))
if not mine.empty:
    findings.append((
        f"{focus}의 개최 효과",
        "  \n".join(f"{r['city']} {r['year']}: {r['base_share']:.2f}% → **{r['host_share']:.2f}%** ({r['ratio']:.2f}배)"
                    for _, r in mine.sort_values("year").iterrows()),
    ))

with st.container(key="step-conclusion"):
    step_head("결론", "그래서, 개최국은 메달을 더 딸까?")
    for row_start in range(0, len(findings), 2):
        cols = st.columns(2, gap="medium")
        for j, (title, body) in enumerate(findings[row_start:row_start + 2]):
            with cols[j], st.container(border=True, key=f"finding-{row_start + j}"):
                st.markdown(f"<p class='finding-num'>{row_start + j + 1:02d}</p>", unsafe_allow_html=True)
                st.markdown(f"#### {title}")
                st.markdown(body)

# ── 12. 한계 ────────────────────────────────────────────────
with st.container(key="step-limits"):
    step_head("한계", "이 결과를 읽을 때 주의할 점")
    st.markdown(
        "- **인과를 단정할 수 없다.** 개최국은 애초에 경제력과 스포츠 투자가 큰 나라가 뽑히는 경향이 있고, "
        "개최 준비 기간의 집중 투자도 함께 작용한다. 이 대시보드는 ‘평소보다 많이 땄다’까지만 보여 준다.\n"
        "- **이유를 나누지 못한다.** 홈 관중, 이동·시차가 없는 점, 심판 판정, **개최국 자동 출전권(선수 수 증가)**, "
        "개최국이 고른 신규 종목 등이 섞여 있다. 선수단 규모 자료는 이 데이터에 없다.\n"
        "- **초기 대회는 비교가 어렵다.** 1896~1936년에는 참가국이 적고 개최국 선수 비중이 매우 컸다(예: 1904년 미국 점유율 89%). "
        "메달을 못 딴 해가 ‘출전’인지 ‘불참’인지 확인되지 않은 경우도 있다(예: 1904년 세인트루이스에 유럽 나라 상당수 불참). "
        "그래서 기본 기간을 1948년 이후로 두었고, 기간 슬라이더로 직접 바꿔 볼 수 있다.\n"
        "- **보이콧**: 1980년에는 미국 등 60여 개국, 1984년에는 소련 등 동구권이 불참해 개최국 점유율이 부풀었다. 불참한 나라의 그해 값은 평균에서 뺐고, 두 대회 자체도 체크박스로 빼고 비교할 수 있다.\n"
        "- **평소 수준의 정의에 따라 값이 달라진다.** 앞뒤 대회 수(k)를 바꾸면 배수가 바뀌며, 2024 파리·2026 밀라노-코르티나는 직후 대회가 없어 직전 대회만으로 비교했다.\n"
        "- **나라 계보 통합**(서독→독일, 소련→러시아)은 분석을 위한 선택이며, 동독의 메달은 독일에 넣지 않았다. "
        "다만 1956~1964년 독일은 동·서독 단일팀이라, 1972 뮌헨(서독)의 평소 수준이 높게 잡혀 효과가 1배 아래로 나온다."
    )

# ── 13. 표와 내려받기 ───────────────────────────────────────
with st.container(key="step-download"):
    step_head("자료", "표와 내려받기")
    table = view.rename(columns={
        "year": "연도", "season": "대회", "city": "개최 도시", "country_ko": "개최국",
        "host_share": "개최 대회 점유율(%)", "base_share": "평소 수준(%)", "diff": "차이(%p)", "ratio": "배수",
        "n_before": "비교한 앞 대회 수", "n_after": "비교한 뒤 대회 수",
    }).drop(columns=["team", "label"]).round(2).sort_values("배수", ascending=False)
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "현재 선택 CSV 저장", data=table.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"host_effect_{years[0]}_{years[1]}.csv", mime="text/csv",
    )
