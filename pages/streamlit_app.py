# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.set_page_config(page_title="자산배분 회귀분석", layout="wide")
st.title("📊 자산배분 회귀분석 대시보드")

# CSV 파일 로딩
DATA_PATH = "data003 - Copy.csv"
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    st.error(f"❌ 파일을 찾을 수 없습니다: {DATA_PATH}")
    st.stop()

# 날짜 처리 및 인덱스 설정
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)

# 연도 및 종속 변수 선택
years = df.index.year.unique()
selected_year = st.selectbox("분석할 연도 선택", sorted(years))
dependent_var = st.selectbox("종속 자산 선택", df.columns)

# 해당 연도 데이터 필터링
df_year = df[df.index.year == selected_year]

# 모든 값을 숫자로 변환
df_year = df_year.apply(pd.to_numeric, errors='coerce')
df_year = df_year.dropna()

# 데이터 존재 여부 확인
if df_year.empty:
    st.warning("⚠️ 선택한 연도에 유효한 데이터가 없습니다.")
    st.stop()

# 설명 변수 자동 선택
excluded = [dependent_var, 'Tbill', 'Excess Return']
X_cols = [col for col in df_year.columns if col not in excluded]
X = df_year[X_cols]
y = df_year[dependent_var]

# X, y 유효성 확인
if X.empty or y.empty:
    st.warning("⚠️ 설명 변수(X) 또는 종속 변수(y)에 유효한 데이터가 없습니다.")
    st.stop()

# 회귀 분석 수행
try:
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    st.subheader("📄 회귀 분석 요약")
    st.text(model.summary())

    # 회귀 계수 시각화
    st.subheader("📈 회귀 계수 시각화")
    coef = model.params.drop('const')
    fig, ax = plt.subplots(figsize=(10, 6))
    coef.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_ylabel("회귀 계수")
    ax.set_title(f"{selected_year}년 {dependent_var} 수익률 회귀분석")
    ax.grid(True)
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ 회귀 분석 중 오류가 발생했습니다:\n\n{e}")
