# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="자산배분 회귀분석", layout="wide")
st.title("📊 Yahoo Finance 기반 자산배분 회귀 분석 대시보드")

# 자산 목록 정의
assets = {
    "Bitcoin": "BTC-USD",
    "S&P 500": "SPY",
    "Gold": "GLD",
    "US Bonds (20Y)": "TLT",
    "MSCI EAFE": "EFA",
    "Emerging Markets": "EEM"
}

# 자산 선택
st.sidebar.header("1️⃣ 분석할 자산 선택")
selected_assets = st.sidebar.multiselect(
    "자산을 2개 이상 선택하세요",
    options=list(assets.keys()),
    default=["Bitcoin", "S&P 500", "Gold"]
)

if len(selected_assets) < 2:
    st.warning("⚠️ 자산은 최소 2개 이상 선택해야 분석이 가능합니다.")
    st.stop()

# ✅ 안정적인 Yahoo Finance 데이터 로더
@st.cache_data
def load_data(tickers_dict, start="2014-01-01"):
    tickers = list(tickers_dict.values())
    raw = yf.download(tickers=tickers, start=start, interval="1mo", group_by="ticker", auto_adjust=True)

    df = pd.DataFrame()
    for name, symbol in tickers_dict.items():
        try:
            if len(tickers) == 1:
                df[name] = raw["Adj Close"]
            else:
                df[name] = raw[(symbol, "Adj Close")]
        except KeyError:
            st.warning(f"⚠️ {name}({symbol})의 데이터를 가져올 수 없습니다.")
    return df.dropna()

# 데이터 불러오기
st.sidebar.info("💾 Yahoo Finance에서 데이터 불러오는 중...")
ticker_map = {name: assets[name] for name in selected_assets}
price_df = load_data(ticker_map)

# 월간 수익률 계산
monthly_returns = price_df.pct_change().dropna()

# 분석 연도 및 종속 변수 선택
years = sorted(monthly_returns.index.year.unique())
st.sidebar.header("2️⃣ 회귀 분석 설정")
selected_year = st.sidebar.selectbox("분석할 연도 선택", years)
dependent_var = st.sidebar.selectbox("종속 변수 선택", selected_assets)

# 연도 필터링
df_year = monthly_returns[monthly_returns.index.year == selected_year]

# 종속/설명 변수 분리
try:
    y = df_year[dependent_var]
    X_cols = [col for col in df_year.columns if col != dependent_var]
    X = df_year[X_cols]

    # 결측 제거
    combined = pd.concat([y, X], axis=1).dropna()
    y = combined.iloc[:, 0]
    X = combined.iloc[:, 1:]

    if y.empty or X.empty:
        st.warning("⚠️ 해당 연도에 유효한 회귀 데이터가 없습니다.")
        st.stop()

    # 회귀 분석 실행
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # 결과 출력
    st.subheader("📄 회귀분석 결과 요약")
    st.text(model.summary())

    # 회귀 계수 시각화
    st.subheader("📊 회귀 계수 (Beta) 시각화")
    coef = model.params.drop("const")
    fig, ax = plt.subplots(figsize=(10, 5))
    coef.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("회귀 계수 (Beta)")
    ax.set_title(f"{selected_year}년 {dependent_var}에 대한 회귀 계수")
    ax.grid(True)
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ 회귀 분석 중 오류가 발생했습니다:\n\n{e}")
