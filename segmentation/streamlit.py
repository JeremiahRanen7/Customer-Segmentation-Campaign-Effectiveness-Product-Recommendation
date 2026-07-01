import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

st.set_page_config(layout="wide",page_title='Customer Segmentation',page_icon=':bar_chart:')

scaler = joblib.load('scaler.joblib')
pca = joblib.load('pca.joblib')
kmeans = joblib.load('kmeans_3.joblib')

st.title("Customer Segmentation Predictor")
st.markdown("This app allows you to segment customers based on their purchasing behavior and demographics.")

st.subheader('Customer Features')

col1,col2 = st.columns(2)

with col1 :
    total_sales_x = st.number_input('Total Sales', min_value = 0.0, step = 1.0, placeholder='Enter Total Sales')
    avg_sales = st.number_input('Average Sales', min_value = 0.0, step = 1.0, placeholder='Enter Average Sales')
    total_quantity = st.number_input('Total Quantity', min_value = 0, step = 1, placeholder='Enter Total Quantity')
    avg_quantity = st.number_input('Average Quantity', min_value = 0.0, step = 1.0, placeholder='Enter Average Quantity')
    avg_basket_size = st.number_input('Average Basket Size', min_value = 0.0, step = 1.0, placeholder='Enter Average Basket Size')
    total_orders = st.number_input('Total Orders', min_value = 0, step = 1, placeholder='Enter Total Orders')

with col2:
    unique_visits_days = st.number_input('Unique Visits Days', min_value = 0, step = 1, placeholder='Enter Unique Visits Days')
    order_frequency = st.number_input('Order Frequency', min_value = 0.0, step = 1.0, placeholder='Enter Order Frequency')
    relative_recency = st.number_input('Relative Recency', min_value = 0.0, step = 1.0, placeholder='Enter Relative Recency')
    avg_days_between_visits = st.number_input('Average Days Between Visits', min_value = 0.0, step = 1.0, placeholder='Enter Average Days Between Visits')
    coupon_dependecy_ratio = st.number_input('Coupon Dependency Ratio', min_value = 0.0, step = 1.0, placeholder='Enter Coupon Dependency Ratio')

if st.button('Predict Segment'):
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

    features = scaler.transform(features)
    features = pca.transform(features)
    prediction = kmeans.predict(features)
    segment = segment_map[prediction[0]]
    st.divider()
    st.subheader('Predicted Customer Segment')
    st.write(f'Predicted Customer Segment: {prediction[0]} - {segment}')

    if prediction[0] == 0:
        st.success('Loyalists - Frequent Buyers')
        st.markdown("""
        - Loyalists are customers who buy more frequently and spend more money. 
        They are likely to be loyal to the brand and make repeat purchases.
                    
        ### Recommendation Strategy:
        - Offer discounts and promotions to loyalists to encourage repeat purchases.
        - Provide exclusive access to loyalists to special offers and events.
        """
        )

    elif prediction[0] == 1:
        st.warning('One-Time & Full-Price Buyers')
        st.markdown("""
        - One-Time & Full-Price Buyers are customers who buy once or at full price. 
        They are likely to be new customers or customers who have made a one-time purchase.
                    
        ### Recommendation Strategy:
        - Offer targeted discounts and promotions to new customers and one-time buyers.
        - Provide exclusive access to new customers and one-time buyers to special offers and events.
        """
        )

    elif prediction[0] == 2:
        st.error('Dormant & At-Risk Customers')
        st.markdown("""
        - Dormant & At-Risk Customers are customers who have not made a purchase for a long time. 
        They are likely to be inactive or at risk of churning.
                    
        ### Recommendation Strategy:
        - Offer targeted discounts and promotions to dormant and at-risk customers.
        - Provide exclusive access to dormant and at-risk customers to special offers and events.
        """
        )



    