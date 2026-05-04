import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page config (ONLY ONCE, FIRST LINE)
st.set_page_config(
    page_title="Dataset Success Rate", 
    page_icon="📊", 
    layout="wide"
)

st.title("📊 Dataset Success Rate Predictor")
st.markdown("### Upload any CSV → Get instant data quality score & ML prediction!")

# Sidebar
st.sidebar.header("📈 Quick Stats")
st.sidebar.markdown("**Free**: 3 uploads/day\n**Pro**: Unlimited (₹199)")

# File uploader
uploaded_file = st.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Show dataframe
    st.dataframe(df.head(), use_container_width=True)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing %", f"{df.isnull().sum().sum()/df.size*100:.1f}%")
    with col4:
        st.metric("Duplicates", df.duplicated().sum())
    
    # Load model
    try:
        with open('model.pk', 'rb') as f:
            model = pickle.load(f)
        st.success("✅ ML Model loaded!")
        
        # Dummy prediction (replace with your model)
        success_rate = 85.3  # Your actual model prediction here
        st.metric("🎯 Success Rate", f"{success_rate:.1f}%")
        st.balloons()
        
    except FileNotFoundError:
        st.warning("⚠️ Upload 'model.pk' to GitHub for ML predictions!")
        st.info("📈 Demo Success Rate: **82.7%**")

# Pro upgrade
st.divider()
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("💎 **Go Pro**")
with col2:
    st.markdown("- Unlimited uploads\n- Priority support\n- Custom models")
    
if st.button("👉 Buy Pro (₹199)"):
    st.balloons()
    st.markdown("""
    [Pay via Gumroad](https://gumroad.com/l/your-link)  
    WhatsApp: +91-XXXXXXXXXX
    """)

st.markdown("---")
st.caption("*Built by Vikram Sathya | Aruppukkottai, Tamil Nadu*")