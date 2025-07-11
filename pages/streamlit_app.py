# streamlit_app.py
import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

st.title("📊 자산배분 회귀분석 대시보드")

# 데이터 불러오기
df = pd.read_csv("data003 - Copy.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 사용자 입력: 연도 및 종속 변수 선택
years = df.index.year.unique()
selected_year = st.selectbox("분석할 연도 선택", sorted(years))
dependent_var = st.selectbox("종속 자산 선택", df.columns)

# 데이터 필터링
df_year = df[df.index.year == selected_year]
y = df_year[dependent_var]

# 설명 변수 자동 선택 (종속 변수, Tbill, Excess Return 제거)
excluded = [dependent_var, 'Tbill', 'Excess Return']
X_cols = [col for col in df_year.columns if col not in excluded]
X = df_year[X_cols]

# 회귀 분석
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

st.subheader("회귀 분석 요약")
st.text(model.summary())

# 회귀계수 시각화
st.subheader("회귀 계수 시각화")
coef = model.params.drop('const')
fig, ax = plt.subplots()
coef.plot(kind='bar', color='skyblue', ax=ax)
ax.set_ylabel("회귀 계수")
ax.set_title(f"{selected_year}년 {dependent_var} 수익률 회귀분석")
st.pyplot(fig)
