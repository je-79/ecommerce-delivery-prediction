
"""
Streamlit Web Application for E-Commerce Delivery Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Load and cache data"""
    try:
        loader = DataLoader('data/e_commerce.csv')
        df = loader.load_data()
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_model():
    """Load and cache model"""
    try:
        model = joblib.load('models/best_model.pkl')
        return model
    except Exception as e:
        st.warning("Model not found. Please run training first.")
        return None


def main():
    # Header
    st.markdown('<h1 class="main-header">📦 E-Commerce Delivery Predictor</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🏠 Home", "📊 Data Explorer", "🔮 Make Prediction", "📈 Model Performance"]
    )
    
    # Load data
    df = load_data()
    
    if df is None:
        st.error("Failed to load data. Please check data/e_commerce.csv exists.")
        return
    
    # Page routing
    if page == "🏠 Home":
        show_home_page(df)
    elif page == "📊 Data Explorer":
        show_data_explorer(df)
    elif page == "🔮 Make Prediction":
        show_prediction_page(df)
    elif page == "📈 Model Performance":
        show_model_performance(df)


def show_home_page(df):
    """Display home page"""
    st.header("Welcome to the Delivery Time Prediction System")
    
    st.markdown("""
    This application predicts whether e-commerce orders will be delivered on time 
    based on various features like product weight, discount, shipping mode, and more.
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Orders", f"{len(df):,}")
    
    with col2:
        delay_rate = (df['Reached.on.Time_Y.N'].sum() / len(df)) * 100
        st.metric("Delay Rate", f"{delay_rate:.1f}%")
    
    with col3:
        avg_discount = df['Discount_offered'].mean()
        st.metric("Avg Discount", f"{avg_discount:.1f}%")
    
    with col4:
        avg_weight = df['Weight_in_gms'].mean()
        st.metric("Avg Weight", f"{avg_weight:.0f}g")
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**Discount Paradox**")
        st.write(f"- On-time orders: {df[df['Reached.on.Time_Y.N']==0]['Discount_offered'].mean():.1f}% avg discount")
        st.write(f"- Delayed orders: {df[df['Reached.on.Time_Y.N']==1]['Discount_offered'].mean():.1f}% avg discount")
        st.write("→ High discounts correlate with delays!")
    
    with col2:
        st.info("**Weight Impact**")
        st.write(f"- On-time orders: {df[df['Reached.on.Time_Y.N']==0]['Weight_in_gms'].mean():.0f}g avg")
        st.write(f"- Delayed orders: {df[df['Reached.on.Time_Y.N']==1]['Weight_in_gms'].mean():.0f}g avg")
        st.write("→ Heavier items arrive on time more!")


def show_data_explorer(df):
    """Display data exploration page"""
    st.header("📊 Data Explorer")
    
    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10))
    
    with col2:
        st.write("**Statistics:**")
        st.dataframe(df.describe())
    
    # Distribution plots
    st.subheader("Feature Distributions")
    
    feature = st.selectbox(
        "Select feature to visualize",
        ['Discount_offered', 'Weight_in_gms', 'Cost_of_the_Product', 
         'Customer_rating', 'Customer_care_calls']
    )
    
    fig, ax = plt.subplots(figsize=(10, 4))
    for status in [0, 1]:
        data = df[df['Reached.on.Time_Y.N'] == status][feature]
        label = 'On Time' if status == 0 else 'Delayed'
        ax.hist(data, bins=30, alpha=0.6, label=label)
    
    ax.set_xlabel(feature)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{feature} Distribution by Delivery Status')
    ax.legend()
    st.pyplot(fig)
    plt.close()


def show_prediction_page(df):
    """Display prediction page"""
    st.header("🔮 Make a Prediction")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Model not available. Please train the model first.")
        return
    
    st.write("Enter order details to predict delivery status:")
    
    # Input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        warehouse = st.selectbox("Warehouse Block", ['A', 'B', 'C', 'D', 'F'])
        shipment_mode = st.selectbox("Shipment Mode", ['Ship', 'Flight', 'Road'])
        importance = st.selectbox("Product Importance", ['low', 'medium', 'high'])
        gender = st.selectbox("Customer Gender", ['M', 'F'])
    
    with col2:
        care_calls = st.slider("Customer Care Calls", 2, 7, 4)
        rating = st.slider("Customer Rating", 1, 5, 3)
        cost = st.number_input("Product Cost ($)", 100, 300, 200)
        prior_purchases = st.slider("Prior Purchases", 2, 6, 3)
    
    with col3:
        discount = st.slider("Discount Offered (%)", 0, 65, 10)
        weight = st.number_input("Weight (grams)", 1000, 8000, 3500)
    
    # Predict button
    if st.button("🎯 Predict Delivery Status", type="primary"):
        # Create input dataframe with original features only
        input_data = pd.DataFrame({
            'Warehouse_block': [warehouse],
            'Mode_of_Shipment': [shipment_mode],
            'Customer_care_calls': [care_calls],
            'Customer_rating': [rating],
            'Cost_of_the_Product': [cost],
            'Prior_purchases': [prior_purchases],
            'Product_importance': [importance],
            'Gender': [gender],
            'Discount_offered': [discount],
            'Weight_in_gms': [weight]
        })
        
        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder
        
        le_warehouse = LabelEncoder()
        le_warehouse.fit(['A', 'B', 'C', 'D', 'F'])
        input_data['Warehouse_block'] = le_warehouse.transform(input_data['Warehouse_block'])
        
        le_mode = LabelEncoder()
        le_mode.fit(['Flight', 'Road', 'Ship'])
        input_data['Mode_of_Shipment'] = le_mode.transform(input_data['Mode_of_Shipment'])
        
        le_importance = LabelEncoder()
        le_importance.fit(['high', 'low', 'medium'])
        input_data['Product_importance'] = le_importance.transform(input_data['Product_importance'])
        
        le_gender = LabelEncoder()
        le_gender.fit(['F', 'M'])
        input_data['Gender'] = le_gender.transform(input_data['Gender'])

         # Create the engineered features (in the correct order!)
        input_data['discount_weight_ratio'] = input_data['Discount_offered'] / (input_data['Weight_in_gms'] + 1)
        input_data['cost_per_gram'] = input_data['Cost_of_the_Product'] / (input_data['Weight_in_gms'] + 1)
        input_data['high_discount_flag'] = (input_data['Discount_offered'] > 15).astype(int)
        input_data['light_item_flag'] = (input_data['Weight_in_gms'] < 3000).astype(int)
        
    # Reorder columns to match training (very important!)
    expected_columns = [
        'Warehouse_block', 'Mode_of_Shipment', 'Customer_care_calls', 
        'Customer_rating', 'Cost_of_the_Product', 'Prior_purchases',
        'Product_importance', 'Gender', 'Discount_offered', 'Weight_in_gms',
        'discount_weight_ratio', 'cost_per_gram', 'high_discount_flag', 'light_item_flag'
    ]
    
    # Make sure all columns are present and in the right order
    input_data = input_data[expected_columns]
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]
        
        # Display result
        st.divider()
        
        if prediction == 1:
            st.error("⚠️ **LIKELY TO BE DELAYED**")
            st.write(f"Probability of delay: {probability[1]:.1%}")
            st.warning("""
            **Recommended Actions:**
            - Set realistic delivery expectations
            - Consider expedited shipping
            - Notify customer proactively
            - Monitor order closely
            """)
        else:
            st.success("✅ **LIKELY TO ARRIVE ON TIME**")
            st.write(f"Probability of on-time delivery: {probability[0]:.1%}")
            st.info("""
            **Order looks good:**
            - Standard processing recommended
            - Normal delivery expectations
            - Routine monitoring sufficient
            """)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.write("Please make sure the model was trained with the same features.")

def show_model_performance(df):
    """Display model performance page"""
    st.header("📈 Model Performance")
    
    st.write("""
    The predictive model was trained using Logistic Regression and achieved 
    the following performance metrics:
    """)
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", "63.6%")
    
    with col2:
        st.metric("Precision", "70.4%")
    
    with col3:
        st.metric("Recall", "67.4%")
    
    with col4:
        st.metric("F1-Score", "68.9%")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    comparison_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
        'Accuracy': [0.636, 0.660, 0.680],
        'Precision': [0.704, 0.769, 0.909],
        'Recall': [0.674, 0.615, 0.515],
        'F1-Score': [0.689, 0.683, 0.657]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Feature importance
    st.subheader("📊 Top Features by Importance")
    
    feature_importance = {
        'Feature': ['Weight', 'Discount', 'Cost', 'Prior Purchases', 'Rating'],
        'Importance': [28.5, 22.1, 17.6, 5.9, 5.9]
    }
    
    fig, ax = plt.subplots(figsize=(10, 4))
    importance_df = pd.DataFrame(feature_importance)
    ax.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    ax.set_xlabel('Importance (%)')
    ax.set_title('Feature Importance')
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    main()

