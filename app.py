import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# === PAGE CONFIG ===
st.set_page_config(page_title="SaaS Valuation Predictor", layout="wide")

# === LOAD AND CLEAN DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv('top_100_saas_companies_2025.csv')

    def convert_money(value):
        try:
            if pd.isnull(value):
                return np.nan
            if isinstance(value, (int, float)):
                return float(value)
            value = str(value).replace('$', '').replace(',', '').strip().upper()
            if value.endswith('B'):
                return float(value[:-1]) * 1e9
            elif value.endswith('M'):
                return float(value[:-1]) * 1e6
            elif value.endswith('K'):
                return float(value[:-1]) * 1e3
            else:
                return float(value)
        except:
            return np.nan

    for col in ['Total Funding', 'Valuation', 'ARR']:
        df[col] = df[col].apply(convert_money)

    df['Total Funding'] = df['Total Funding'].fillna(df['Total Funding'].median())
    df['Company Age'] = 2025 - df['Founded Year']
    df.drop(['Company Name', 'Top Investors', 'Founded Year'], axis=1, inplace=True)

    for col in ['HQ', 'Industry', 'Product']:
        df[col] = df[col].astype(str)

    return df

df = load_data()

# === ENCODING & SCALING ===
label_cols = ['HQ', 'Industry', 'Product']
le_dict = {col: LabelEncoder().fit(df[col]) for col in label_cols}
for col in label_cols:
    df[col] = le_dict[col].transform(df[col])

X = df.drop('Valuation', axis=1)
y = df['Valuation']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === MODEL ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# === SIDEBAR — USER INPUT ===
st.sidebar.header("📊 Predict Company Valuation")

def user_input_features():
    hq = st.sidebar.selectbox("HQ Location", le_dict['HQ'].classes_)
    industry = st.sidebar.selectbox("Industry", le_dict['Industry'].classes_)
    product = st.sidebar.selectbox("Product Type", le_dict['Product'].classes_)
    funding = st.sidebar.number_input("Total Funding ($)", min_value=0.0, value=100_000_000.0)
    arr = st.sidebar.number_input("ARR ($)", min_value=0.0, value=20_000_000.0)
    employees = st.sidebar.slider("Number of Employees", 10, 10000, 200)
    rating = st.sidebar.slider("G2 Rating", 0.0, 5.0, 4.2)
    age = st.sidebar.slider("Company Age", 1, 30, 5)

    # Encode
    features = pd.DataFrame([{
        'Total Funding': funding,
        'ARR': arr,
        'Employees': employees,
        'G2 Rating': rating,
        'HQ': le_dict['HQ'].transform([hq])[0],
        'Industry': le_dict['Industry'].transform([industry])[0],
        'Product': le_dict['Product'].transform([product])[0],
        'Company Age': age
    }])
    return features

user_input = user_input_features()
user_input_scaled = scaler.transform(user_input)
prediction = model.predict(user_input_scaled)[0]

# === MAIN PAGE ===
st.title("🚀 SaaS Company Valuation Predictor")
st.markdown("Predict the estimated company valuation (in USD) based on funding, ARR, industry, and other business metrics.")

# === SECTIONS ===

st.header("📌 Introduction")
st.markdown("""
This project analyzes the top 100 SaaS companies of 2025. The goal is to understand the factors influencing company valuations 
and build a model that can predict a company’s valuation based on real-world features like funding, ARR, G2 rating, etc.
""")

st.header("📊 EDA Highlights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Valuation Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['Valuation'], bins=30, ax=ax1)
    st.pyplot(fig1)

with col2:
    st.subheader("Top 10 Industries by Average Valuation")
    top_industry_val = df.groupby('Industry')['Valuation'].mean().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_industry_val.values, y=top_industry_val.index, ax=ax2)
    st.pyplot(fig2)

st.header("🤖 Model & Prediction")
st.markdown("""
We trained a **Random Forest Regressor** on the dataset using all major business features. 
Below is the real-time valuation prediction based on your inputs from the sidebar.
""")

st.success(f"💰 **Predicted Valuation: ${prediction:,.2f}**")

st.subheader("Model Performance Metrics")
st.write(f"- **RMSE:** {rmse:,.2f}")
st.write(f"- **R² Score:** {r2:.2f}")

st.header("✅ Conclusion")
st.markdown("""
- **Total Funding**, **ARR**, and **Industry type** are key drivers of SaaS company valuation.
- The model performs well with an R² of around {:.2f}.
- Future work could involve time-based trend analysis or adding investor reputation.
""".format(r2))

st.markdown("Made with ❤️ using Streamlit")
