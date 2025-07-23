import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score

st.set_page_config(page_title="CLV Prediction App",layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>Customer Lifetime Value (CLV) Prediction</h1>",
    unsafe_allow_html=True
)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png",width=120)
st.sidebar.title("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload `online_retail_II.xlsx`",type=["xlsx"])
if uploaded_file:
    data = pd.read_excel(uploaded_file,sheet_name="Year 2010-2011")
    df=data.dropna(subset=["Customer ID"])
    df=df[df["Quantity"]>0]
    df["TotalPrice"]=df["Quantity"]*df["Price"]
    df["InvoiceDate"]=pd.to_datetime(df["InvoiceDate"])
    snapshot_date=df["InvoiceDate"].max()+timedelta(days=1)
    rfm = df.groupby("Customer ID").agg({
        "InvoiceDate":lambda x:(snapshot_date-x.max()).days,
        "Invoice":"nunique",
        "TotalPrice":"sum"
    })
    rfm.columns=["Recency","Frequency","Monetary"]
    rfm=rfm[rfm["Monetary"]>0]
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("🧾 Total Transactions",f"{df['Invoice'].nunique():,}")
    col2.metric("👥 Total Customers",f"{df['Customer ID'].nunique():,}")
    col3.metric("💵 Total Revenue",f"£{df['TotalPrice'].sum():,.2f}")
    st.markdown("---")
    st.markdown("### 📈 RFM Distributions")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.plotly_chart(px.histogram(rfm, x="Recency",nbins=30,color_discrete_sequence=["#2196F3"],
                                     title="Recency Distribution"),use_container_width=True)
    with col5:
        st.plotly_chart(px.histogram(rfm, x="Frequency",nbins=30,color_discrete_sequence=["#4CAF50"],
                                     title="Frequency Distribution"),use_container_width=True)
    with col6:
        st.plotly_chart(px.histogram(rfm, x="Monetary",nbins=30,color_discrete_sequence=["#FF5722"],
                                     title="Monetary Value Distribution"),use_container_width=True)
    st.markdown("---")
    st.markdown("### 🤖 CLV Model (Linear Regression)")
    X = rfm[["Recency", "Frequency"]]
    y = rfm["Monetary"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=42)
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    col7, col8=st.columns(2)
    col7.metric("📉 MAE", f"£{mean_absolute_error(y_test, y_pred):.2f}")
    col8.metric("📈 R² Score", f"{r2_score(y_test, y_pred):.2f}")
    st.markdown("---")
    st.markdown("### Predict CLV for Custom Input")
    with st.form("prediction_form"):
        recency=st.slider("Recency (days since last purchase)", 0, 365, 90)
        frequency=st.slider("Frequency (number of purchases)", 1, 100, 10)
        submitted=st.form_submit_button("Predict CLV ")
    if submitted:
        pred_clv = model.predict([[recency, frequency]])[0]
        st.success(f"✅ Estimated CLV:**£{pred_clv:.2f}**")
    st.markdown("---")
    st.markdown("###  Customer Segmentation")
    rfm["Segment"]=pd.qcut(rfm["Monetary"], 4, labels=["Low","Mid-Low","Mid-High","High"])
    seg_counts=rfm["Segment"].value_counts().sort_index()
    st.plotly_chart(px.pie(names=seg_counts.index,values=seg_counts.values,title="Customer Segments",
                           color_discrete_sequence=px.colors.qualitative.Set3),use_container_width=True)
    st.markdown("---")
    st.markdown("<center>✨ Built with ❤️ using Streamlit | By Rishi & ChatGPT</center>", unsafe_allow_html=True)

else:
    st.info("👈 Please upload the '.xsl' file from the sidebar to begin.")
