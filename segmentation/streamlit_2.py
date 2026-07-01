import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Config (Keep wide layout, update icon)
st.set_page_config(layout="wide", page_title='Customer Segmentation', page_icon='🛍️')

# Load Models
# (Using caching to prevent reloading models on every button click - a good UX/Performance tweak!)
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.joblib')
    pca = joblib.load('pca.joblib')
    kmeans = joblib.load('kmeans_3.joblib')
    return scaler, pca, kmeans

scaler, pca, kmeans = load_models()

# 2. Polished Header
st.title("🛍️ Customer Segmentation Predictor")
st.markdown("""
Welcome to the segmentation portal. Enter the customer's purchasing behavior and demographics below 
to predict their segment and view tailored marketing strategies.
""")
st.divider()

st.subheader('📊 Customer Features')

# 3. Form Inputs with Tooltips (help parameters)
col1, col2 = st.columns(2)

with col1:
    total_sales_x = st.number_input('Total Sales ($)', min_value=0.0, step=1.0, help='Total lifetime revenue from this customer.')
    avg_sales = st.number_input('Average Sales ($)', min_value=0.0, step=1.0, help='Average order value.')
    total_quantity = st.number_input('Total Quantity', min_value=0, step=1)
    avg_quantity = st.number_input('Average Quantity', min_value=0.0, step=1.0)
    avg_basket_size = st.number_input('Average Basket Size', min_value=0.0, step=1.0)
    total_orders = st.number_input('Total Orders', min_value=0, step=1)

with col2:
    unique_visits_days = st.number_input('Unique Visits (Days)', min_value=0, step=1)
    order_frequency = st.number_input('Order Frequency', min_value=0.0, step=1.0)
    relative_recency = st.number_input('Relative Recency', min_value=0.0, step=1.0, help='How recently the customer made their last purchase.')
    avg_days_between_visits = st.number_input('Avg Days Between Visits', min_value=0.0, step=1.0)
    coupon_dependecy_ratio = st.number_input('Coupon Dependency Ratio', min_value=0.0, step=1.0, help='Ratio of orders placed using a discount code.')

st.write("") # Adds a bit of vertical breathing room

# 4. Prominent Button
if st.button('🎯 Predict Customer Segment', type='primary', use_container_width=True):
    
    # 5. Visual Loading State
    with st.spinner('Analyzing customer data...'):
        features = pd.DataFrame({
            'total_sales_x': [total_sales_x],
            'avg_sales': [avg_sales],
            'total_quantity': [total_quantity],
            'avg_quantity': [avg_quantity],
            'avg_basket_size': [avg_basket_size],
            'total_orders': [total_orders],
            'unique_visits_days': [unique_visits_days],
            'order_frequency': [order_frequency],
            'relative_recency': [relative_recency],
            'avg_days_between_visits': [avg_days_between_visits],
            'coupon_dependecy_ratio': [coupon_dependecy_ratio]
        })

        segment_map = {
            0: 'Loyalists - Frequent Buyers',
            1: 'One-Time & Full-Price Buyers',
            2: 'Dormant & At-Risk Customers'
        }

        # Transformations & Prediction
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        prediction = kmeans.predict(features_pca)
        segment = segment_map[prediction[0]]
        
    st.divider()
    
    # 6. Improved Result Display using Columns and Metrics
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Predicted Segment", value=f"Cluster {prediction[0]}")
        
    with res_col2:
        if prediction[0] == 0:
            st.success(f'🌟 **{segment}**')
            st.write("Loyalists are customers who buy more frequently and spend more money. They are likely to be loyal to the brand and make repeat purchases.")
            with st.expander("💡 View Recommendation Strategy", expanded=True):
                st.markdown("""
                * **Incentivize:** Offer discounts and promotions to encourage repeat purchases.
                * **VIP Treatment:** Provide exclusive access to special offers and events.
                """)

        elif prediction[0] == 1:
            st.warning(f'🛍️ **{segment}**')
            st.write("These customers buy once or exclusively at full price. They are likely to be newer customers or impulse buyers.")
            with st.expander("💡 View Recommendation Strategy", expanded=True):
                st.markdown("""
                * **Onboarding:** Offer targeted discounts to convert them into repeat buyers.
                * **Engagement:** Provide exclusive access to special events to build brand affinity.
                """)

        elif prediction[0] == 2:
            st.error(f'⚠️ **{segment}**')
            st.write("Customers who have not made a purchase for a long time. They are highly inactive or at immediate risk of churning.")
            with st.expander("💡 View Recommendation Strategy", expanded=True):
                st.markdown("""
                * **Win-Back Campaigns:** Offer highly targeted, aggressive discounts to incentivize a return.
                * **Re-engagement:** Reach out with personalized "we miss you" offers or exclusive events.
                """)