import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025       # 수업 기준 기간의 마지막 해
MIN_DAYS = 300         # 이보다 관측일이 적은 해는 제외

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")
st.title("🌡️ 기온 예측기")
st.caption("서울 연평균기온으로 회귀 직선을 만들고, 고른 연도의 기온을 예측합니다.")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_annual_mean(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8")
    df.columns = [str(c).strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])

    # 기준 기간(2025년)까지만 사용
    df = df[df["날짜"].dt.year <= LAST_YEAR].copy()
    df["연도"] = df["날짜"].dt.year

    annual = (
        df.groupby("연도")
        .agg(연평균기온=("평균기온", "mean"), 관측일=("날짜", "nunique"))
        .reset_index()
    )
    # 관측일이 300일 미만인 해 제외
    annual = annual[annual["관측일"] >= MIN_DAYS].reset_index(drop=True)
    return annual


try:
    annual = load_annual_mean(DATA_URL)
except Exception as e:  # 네트워크/열 이름 문제 등
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()

if len(annual) < 2:
    st.error("회귀 직선을 만들기에 데이터가 부족합니다.")
    st.stop()

x = annual["연도"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
r = float(np.corrcoef(x, y)[0, 1])

# 최근 20년(끝 연도 기준)만으로 구한 기울기
RECENT_YEARS = 20
recent = annual[annual["연도"] > x.max() - RECENT_YEARS]
if len(recent) >= 2:
    xr = recent["연도"].to_numpy(dtype=float)
    yr = recent["연평균기온"].to_numpy(dtype=float)
    slope_recent, intercept_recent = np.polyfit(xr, yr, 1)
else:
    xr = yr = None
    slope_recent = intercept_recent = None

# --- 직선을 만든 데이터 요약 ---
c1, c2, c3 = st.columns(3)
c1.metric("사용한 해의 개수", f"{len(annual)}개")
c2.metric("시작 연도", f"{int(x.min())}년")
c3.metric("끝 연도", f"{int(x.max())}년")
st.caption(
    f"{LAST_YEAR}년까지의 자료 중 관측일이 {MIN_DAYS}일 이상인 해만 사용했습니다."
)

# --- 슬라이더와 예상 기온 ---
year = st.slider("예측할 연도", min_value=1900, max_value=2100, value=2030, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
      <div style="font-size:1.1rem; opacity:0.7;">{year}년 예상 연평균기온</div>
      <div style="font-size:4.5rem; font-weight:800; line-height:1.1;">{pred:.1f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if year < x.min() or year > x.max():
    st.info("선택한 연도가 데이터 범위 밖이라, 직선을 연장한 외삽 값입니다.")

# --- 산점도 + 회귀 직선 ---
x_line = np.array([min(x.min(), year), max(x.max(), year)])
y_line = slope * x_line + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=7, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f}°C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="crimson"),
    )
)
if slope_recent is not None:
    xr_line = np.array([xr.min(), xr.max()])
    fig.add_trace(
        go.Scatter(
            x=xr_line,
            y=slope_recent * xr_line + intercept_recent,
            mode="lines",
            name=f"최근 {RECENT_YEARS}년 직선",
            line=dict(width=3, color="royalblue", dash="dash"),
        )
    )
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예상",
        marker=dict(size=16, symbol="star", color="gold", line=dict(width=1.5, color="black")),
        hovertemplate="%{x}년 예상<br>%{y:.2f}°C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
)
st.plotly_chart(fig, use_container_width=True)

# --- 상관계수와 직선 식 ---
st.metric("상관계수 (연도 - 연평균기온)", f"{r:.3f}")

# --- 기울기를 '100년에 몇 도' 로 환산해 비교 ---
st.subheader("100년에 몇 도 오를까?")


def big_card(title: str, sub: str, value, color: str) -> str:
    text = "계산 불가" if value is None else f"{value * 100:+.2f} °C"
    return f"""
    <div style="text-align:center; padding:1rem 0.5rem; border-radius:12px;
                border:2px solid {color};">
      <div style="font-size:1.05rem; font-weight:700;">{title}</div>
      <div style="font-size:0.85rem; opacity:0.7;">{sub}</div>
      <div style="font-size:2.8rem; font-weight:800; line-height:1.3; color:{color};">{text}</div>
      <div style="font-size:0.85rem; opacity:0.7;">100년당</div>
    </div>
    """


k1, k2 = st.columns(2)
k1.markdown(
    big_card("전체 기간", f"{int(x.min())}~{int(x.max())}년 ({len(annual)}개 해)", slope, "crimson"),
    unsafe_allow_html=True,
)
if slope_recent is not None:
    sub_recent = f"{int(xr.min())}~{int(xr.max())}년 ({len(recent)}개 해)"
else:
    sub_recent = "데이터 부족"
k2.markdown(
    big_card(f"최근 {RECENT_YEARS}년", sub_recent, slope_recent, "royalblue"),
    unsafe_allow_html=True,
)

if slope_recent is not None and slope != 0:
    ratio = slope_recent / slope
    st.write(f"최근 {RECENT_YEARS}년의 상승 속도는 전체 기간의 약 **{ratio:.1f}배**입니다.")
st.caption(
    "기울기(°C/년)에 100을 곱해 환산한 값입니다. 최근 구간은 해 수가 적어 "
    "연도별 변동에 크게 흔들릴 수 있어요."
)
st.write(f"전체 기간 회귀 직선: **기온 = {slope:.4f} × 연도 + ({intercept:.2f})**")

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(
        annual.rename(columns={"연평균기온": "연평균기온(°C)"}).round({"연평균기온(°C)": 2}),
        use_container_width=True,
        hide_index=True,
    )
