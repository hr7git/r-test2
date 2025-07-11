import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.title("📊 자산배분 회귀분석 대시보드")

# CSV 파일 로딩
df = pd.read_csv("data003 - Copy.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df.set_index('Date', inplace=True)

# 연도와 종속 변수 선택
years = df.index.year.unique()
selected_year = st.selectbox("분석할 연도 선택", sorted(years))
dependent_var = st.selectbox("종속 자산 선택", df.columns)

# 해당 연도 데이터 선택
df_year = df[df.index.year == selected_year]

# 모든 데이터를 숫자로 변환
df_year = df_year.apply(pd.to_numeric, errors='coerce')

# NaN 제거
df_year = df_year.dropna()

# 종속 변수와 설명 변수 구분
excluded = [dependent_var, 'Tbill', 'Excess Return']
X_cols = [col for col in df_year.columns if col not in excluded]
y = df_year[dependent_var]
X = df_year[X_cols]

# 회귀 분석
try:
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit()

    # 회귀 결과 출력
    st.subheader("회귀 분석 요약")
    st.text(model.summary())

    # 회귀 계수 시각화
    st.subheader("회귀 계수 시각화")
    coef = model.params.drop('const')
    fig, ax = plt.subplots()
    coef.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_ylabel("회귀 계수")
    ax.set_title(f"{selected_year}년 {dependent_var} 수익률 회귀분석")
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ 회귀 분석 중 오류가 발생했습니다: {e}")
