# streamlit_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="자산배분 회귀 분석", layout="wide")
st.title("📈 자산배분 회귀 분석 대시보드")

# ✅ 자산 이름과 Yahoo Finance Ticker 매핑
assets = {
    "Bitcoin": "BTC-USD",
    "S&P 500": "SPY",
    "Gold": "GLD",
    "US Bonds (20Y)": "TLT",
    "MSCI EAFE": "EFA",
    "Emerging Markets": "EEM"
}

# 1️⃣ 자산 선택
st.sidebar.header("1️⃣ 자산 선택")
selected_assets = st.sidebar.multiselect(
    "분석에 사용할 자산을 2개 이상 선택하세요:",
    options=list(assets.keys()),
    default=["Bitcoin", "S&P 500", "Gold"]
)

if len(selected_assets) < 2:
    st.warning("⚠️ 자산을 최소 2개 이상 선택해주세요.")
    st.stop()

# 2️⃣ 데이터 로드 함수
@st.cache_data
def load_data(tickers_dict, start="2014-01-01"):
    tickers = list(tickers_dict.values())
    raw = yf.download(
        tickers=tickers,
        start=start,
        interval="1mo",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if raw.empty:
        return pd.DataFrame()  # 모든 요청 실패 시 빈 DataFrame 반환

    df = pd.DataFrame()
    for name, symbol in tickers_dict.items():
        try:
            if len(tickers) == 1:
                df[name] = raw["Adj Close"]
            else:
                df[name] = raw[(symbol, "Adj Close")]
        except Exception:
            st.warning(f"⚠️ {name}({symbol})의 데이터를 가져올 수 없습니다.")
    return df.dropna()

# 3️⃣ 데이터 로딩
st.sidebar.info("💾 데이터를 Yahoo Finance에서 불러오는 중...")
ticker_map = {name: assets[name] for name in selected_assets}
price_df = load_data(ticker_map)

# 4️⃣ 수익률 계산
monthly_returns = price_df.pct_change().dropna()

if monthly_returns.empty:
    st.error("❌ 선택한 자산의 데이터를 가져오지 못했습니다. 다른 자산을 선택하거나 나중에 다시 시도해보세요.")
    st.stop()

# 5️⃣ 분석 설정
st.sidebar.header("2️⃣ 분석 설정")
available_years = sorted(monthly_returns.index.year.unique())
selected_year = st.sidebar.selectbox("회귀 분석할 연도 선택", available_years)
dependent_var = st.sidebar.selectbox("종속 변수 선택 (나머지는 독립 변수로 사용)", selected_assets)

# 6️⃣ 해당 연도 데이터 필터링
df_year = monthly_returns[monthly_returns.index.year == selected_year]

if df_year.empty:
    st.warning("⚠️ 선택한 연도에 유효한 데이터가 없습니다.")
    st.stop()

# 7️⃣ 회귀분석 실행
try:
    y = df_year[dependent_var]
    X = df_year[[col for col in df_year.columns if col != dependent_var]]
    data_combined = pd.concat([y, X], axis=1).dropna()
    y = data_combined[dependent_var]
    X = data_combined.drop(columns=[dependent_var])

    if y.empty or X.empty:
        st.warning("⚠️ 회귀 분석에 필요한 데이터가 부족합니다.")
        st.stop()

    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # 8️⃣ 결과 출력
    st.subheader("📄 회귀 분석 결과")
    st.text(model.summary())

    # 9️⃣ 회귀계수 시각화
    st.subheader("📊 회귀 계수 시각화")
    coef = model.params.drop("const")
    fig, ax = plt.subplots(figsize=(8, 4))
    coef.plot(kind="bar", ax=ax, color="skyblue")
    ax.set_ylabel("회귀 계수 (Beta)")
    ax.set_title(f"{selected_year}년 {dependent_var}에 대한 회귀 계수")
    ax.grid(True)
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ 회귀 분석 중 오류가 발생했습니다: {e}")
