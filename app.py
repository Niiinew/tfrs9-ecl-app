import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

st.set_page_config(page_title="TFRS9 PD & ECL Calculator", layout="wide")

# 🎨 Theme
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    color: #2c3e50;
    font-family: 'Segoe UI', sans-serif;
}
.css-10trblm, .css-1d391kg {
    color: #1c2b3a !important;
    font-weight: 700;
}
.stButton>button {
    background-color: #2c3e50;
    color: white;
    border-radius: 6px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #1a252f;
}
.stMetric {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid #e3e3e3;
}
</style>
""", unsafe_allow_html=True)

# ✅ Load models
models = {
    "XGBoost": joblib.load("XG_model.pkl"),
    "Logistic Regression": joblib.load("logit_model.pkl"),
    "Decision Tree": joblib.load("tree_model.pkl")
}

st.title("📊 TFRS9 PD & ECL Calculator")
model_choice = st.selectbox("📌 เลือกโมเดลที่ต้องการใช้", list(models.keys()))
model = models[model_choice]

# ✅ Upload
uploaded_file = st.file_uploader("📥 อัปโหลดไฟล์ข้อมูลลูกค้า (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 🔁 สร้างกลุ่ม GRADE_Group
    df['GRADE_Group'] = df['GRADE'].map({
        'A': 'Low', 'B': 'Low',
        'C': 'Medium',
        'D': 'High', 'E': 'High', 'F': 'High'
    }).fillna('Missing')

    # 🔁 สร้างกลุ่ม BADMONTHBOTday
    def group_badmonthbot(val):
        if val < 0.33333333:
            return "Low"
        elif val < 0.66666666:
            return "Medium"
        else:
            return "High"

    if 'BADMONTHBOT' in df.columns:
        df['BADMONTHBOTday'] = df['BADMONTHBOT'].apply(group_badmonthbot)

    # 🔁 สร้างกลุ่ม No_ever30_12m_group
    def group_no_ever30(x):
        if x == 0:
            return 'Never'
        elif x in [1, 2]:
            return 'Low'
        elif x in [3, 4]:
            return 'Medium'
        else:
            return 'High'

    if 'No_ever30_12m' in df.columns:
        df['No_ever30_12m'] = df['No_ever30_12m_ori'].apply(group_no_ever30)

    # ✅ ใช้ pd.cut แบ่งช่วงอายุ
    bins = [0, 29, 50, float('inf')]
    labels = ['Young', 'Adult', 'Senior']

    df['Age_group'] = pd.cut(df['Age_current'], bins=bins, labels=labels)

    st.markdown("### 🔍 ข้อมูลที่อัปโหลด")
    st.dataframe(df.head())

# ✅ ต้องอยู่นอก if uploaded_file:
if uploaded_file and st.button("✅ ทำนาย PD และคำนวณ ECL"):
    try:
        X = df.copy()
        X.drop(columns=["CID"], inplace=True, errors="ignore")

        if "GRADE" in X.columns:
            X = pd.get_dummies(X, columns=["GRADE"], drop_first=True)

        expected_features = (
            model.feature_names_in_
            if hasattr(model, "feature_names_in_")
            else model.get_booster().feature_names
        )
        for col in expected_features:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_features]
        X = X.apply(pd.to_numeric, errors='coerce').reset_index(drop=True)
        df = df.reset_index(drop=True)

        # ✅ Predict PD
        df["PD"] = model.predict_proba(X)[:, 1]

        # ✅ ถ้า Stage == 3 ให้ PD = 1
        df["Stage"] = pd.to_numeric(df["Stage"], errors="coerce").fillna(1).astype("Int64")
        df["PD"] = df.apply(lambda row: 1.0 if row["Stage"] == 3 else row["PD"], axis=1)

        df["EAD"] = pd.to_numeric(df["EAD"], errors="coerce").fillna(0).round(2)
        df["LGD"] = pd.to_numeric(df["LGD"], errors="coerce").fillna(0).round(2)
        df["PD"] = df["PD"].fillna(0).round(4)

        # ✅ คำนวณ ECL = PD * LGD * EAD (ทุก Stage)
        df["ECL"] = df["PD"] * df["LGD"] * df["EAD"]
        df = df.round({"PD": 4, "EAD": 2, "LGD": 2, "ECL": 2})

        st.success("✅ คำนวณเสร็จแล้ว")
        st.markdown("### 📈 Dashboard สรุปผล")
        col1, col2, col3 = st.columns(3)
        col1.metric("📌 ECL รวมทั้งหมด", f"{df['ECL'].sum():,.2f}")
        col2.metric("🎯 ค่าเฉลี่ย PD", f"{df['PD'].mean():.4f}")
        col3.metric("👥 จำนวนลูกค้า", len(df))

        st.markdown("### 🧁 สัดส่วนลูกค้าแต่ละ Stage")
        stage_df = df["Stage"].value_counts().reset_index()
        stage_df.columns = ["Stage", "จำนวนลูกค้า"]
        stage_df["Stage"] = stage_df["Stage"].astype(str)
        pie_fig = px.pie(
            stage_df,
            names="Stage",
            values="จำนวนลูกค้า",
            color="Stage",
            color_discrete_sequence=["#1f3b57", "#537791", "#9cb4c9"],
            hole=0.4,
        )
        pie_fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(pie_fig, use_container_width=True)

        # ✅ Reorder columns
        ordered_cols = [
            "CID", "BADMONTH_Bucket", "BADMONTHBOT", "BADMONTHBOTday",
            "PAYMENT_METHOD", "GRADE", "GRADE_Group",
            "everx_lag_3", "ever30_lag_3", "ever60_lag_3",
            "No_ever30_12m", "No_ever30_12m_group", "f_pd1",
            "Stage", "PD", "LGD", "EAD", "ECL"
        ]
        ordered_cols = [col for col in ordered_cols if col in df.columns]
        df = df[ordered_cols]

        st.markdown("### 📋 ผลลัพธ์ทั้งหมด")
        st.dataframe(df)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇ ดาวน์โหลดผลลัพธ์", csv, "result.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
