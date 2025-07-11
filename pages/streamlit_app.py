# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="Yahoo Finance 자산배분 분석", layout="wide")
st.title("📈 자산배분 회귀 분석 (Yahoo Finance 기반)")

# 자산 리스트 (Bitcoin + 글로벌 ETF들)
assets = {
    "Bitcoin": "BTC-USD",
    "S&P 500": "SPY",
    "Gold": "GLD",
    "US Bonds (20Y)": "TLT",
    "MSCI EAFE": "EFA",
    "Emerging Markets": "EEM"
}

st.sidebar.header("분석 대상 자산 설정")
selected_assets = st.sidebar.multiselect(
    "자산 선택 (최소 2개)",
    list(assets.keys()),
    default=["Bitcoin", "S&P 500", "Gold", "US Bonds (20Y)"]
)

if len(selected_assets) < 2:
    st.warning("⚠️ 자산을 최소 2개 이상 선택해야 회귀 분석이 가능합니다.")
    st.stop()

# 데이터 다운로드
@st.cache_data
def load_data(tickers, start="2014-01-01"):
    data = yf.download(tickers=list(tickers.values()), start=start, interval="1mo")["Adj Close"]
    data = data.dropna(how='all')  # 전체 결측 행 제거
    data = data.fillna(method='ffill')  # 일부 자산 결측치 보정
    return data

data = load_data({k: assets[k] for k in selected_assets})
monthly_returns = data.pct_change().dropna()

# 연도 및 종속 변수 선택
years = monthly_returns.index.year.unique()
selected_year = st.selectbox("분석할 연도 선택", sorted(years))
dependent_var = st.selectbox("종속 자산 선택", selected_assets)

# 해당 연도 데이터 선택
df_year = monthly_returns[monthly_returns.index.year == selected_year]

# 설명변수 및 종속변수 분리
try:
    y = df_year[assets[dependent_var]]
    X_cols = [assets[a] for a in selected_assets if a != dependent_var]
    X = df_year[X_cols]

    # 결측 제거
    data_combined = pd.concat([y, X], axis=1).dropna()
    y = data_combined.iloc[:, 0]
    X = data_combined.iloc[:, 1:]

    if X.empty or y.empty:
        st.warning("⚠️ 선택한 연도에 유효한 회귀 데이터가 없습니다.")
        st.stop()

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    st.subheader("📄 회귀 분석 요약")
    st.text(model.summary())

    st.subheader("📊 회귀 계수 시각화")
    coef = model.params.drop("const")
    fig, ax = plt.subplots(figsize=(10, 6))
    coef.plot(kind="bar", color="cornflowerblue", ax=ax)
    ax.set_title(f"{selected_year}년 {dependent_var} 수익률 회귀 계수")
    ax.set_ylabel("Beta (회귀 계수)")
    ax.grid(True)
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ 회귀 분석 중 오류 발생:\n\n{e}")
