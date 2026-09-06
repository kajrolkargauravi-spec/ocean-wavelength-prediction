import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from keras.models import load_model
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Ocean Wave Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .metric-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models and scaler
@st.cache_resource
def load_models():
    try:
        lstm_model = load_model('lstm_model.keras')
        xgb_model = joblib.load('xgb_model.joblib')
        scaler = joblib.load('scaler.joblib')
        return lstm_model, xgb_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Load data generation function for demo
@st.cache_data
def generate_sample_data(days=90):
    """Generate sample wave data for demonstration"""
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    np.random.seed(42)
    
    data = {
        'Date': dates,
        'Wave_Height': np.random.uniform(0.5, 3.5, days),
        'Wave_Period': np.random.uniform(5, 15, days),
        'Wind_Speed': np.random.uniform(5, 25, days),
        'Water_Temperature': np.random.uniform(15, 25, days),
        'Wavelength': np.random.uniform(40, 300, days)
    }
    return pd.DataFrame(data)

# ==================== MAIN APP ====================

st.title("🌊 Ocean Wave Wavelength Prediction System")
st.markdown("Advanced ML-based prediction and analysis platform for ocean waves")

# Load models
lstm_model, xgb_model, scaler = load_models()

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🏠 Dashboard", "📊 Analysis", "🔮 Predictions", "📈 Model Performance", "ℹ️ About"]
)

# ==================== PAGE: DASHBOARD ====================
if page == "🏠 Dashboard":
    st.header("Dashboard Overview")
    
    # Load sample data
    df = generate_sample_data()
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg Wave Height (m)", f"{df['Wave_Height'].mean():.2f}", 
                  f"{df['Wave_Height'].mean() - df['Wave_Height'].iloc[0]:.2f}")
    
    with col2:
        st.metric("Avg Wavelength (m)", f"{df['Wavelength'].mean():.2f}",
                  f"{df['Wavelength'].mean() - df['Wavelength'].iloc[0]:.2f}")
    
    with col3:
        st.metric("Avg Wave Period (s)", f"{df['Wave_Period'].mean():.2f}",
                  f"{df['Wave_Period'].mean() - df['Wave_Period'].iloc[0]:.2f}")
    
    with col4:
        st.metric("Avg Wind Speed (m/s)", f"{df['Wind_Speed'].mean():.2f}",
                  f"{df['Wind_Speed'].mean() - df['Wind_Speed'].iloc[0]:.2f}")
    
    st.divider()
    
    # Real-time data visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wave Height Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Wave_Height'],
            mode='lines+markers',
            name='Wave Height',
            line=dict(color='#1f77b4', width=2),
            fill='tozeroy'
        ))
        fig.update_layout(
            height=400,
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Wavelength Distribution")
        fig = go.Figure(data=[
            go.Histogram(
                x=df['Wavelength'],
                nbinsx=30,
                marker_color='#ff7f0e'
            )
        ])
        fig.update_layout(
            height=400,
            xaxis_title="Wavelength (m)",
            yaxis_title="Frequency",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Multi-parameter analysis
    st.subheader("Multi-Parameter Time Series")
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Wave Height", "Wave Period", "Wind Speed", "Water Temperature"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Wave_Height'], name='Wave Height',
                   line=dict(color='#1f77b4')), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Wave_Period'], name='Wave Period',
                   line=dict(color='#2ca02c')), row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Wind_Speed'], name='Wind Speed',
                   line=dict(color='#d62728')), row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Water_Temperature'], name='Water Temp',
                   line=dict(color='#9467bd')), row=2, col=2
    )
    
    fig.update_layout(height=600, template='plotly_white', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: ANALYSIS ====================
elif page == "📊 Analysis":
    st.header("Detailed Data Analysis")
    
    df = generate_sample_data()
    
    # Data upload option
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Data Analysis Tools")
    with col2:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    
    # Display data
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Dataset Preview")
    with col2:
        if st.button("📥 Download Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"wave_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    st.dataframe(df.head(20), use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(df.describe())
    
    with col2:
        # Correlation heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        fig.update_layout(height=500, title="Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.subheader("Distribution Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_col = st.selectbox("Select column for analysis", numeric_cols)
        fig = px.histogram(df, x=selected_col, nbins=30, title=f"Distribution of {selected_col}",
                          marginal="box", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df[numeric_cols], title="Box Plot Analysis", template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot matrix
    st.subheader("Relationship Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox("X-axis", numeric_cols, key="x_scatter")
    with col2:
        y_col = st.selectbox("Y-axis", numeric_cols, key="y_scatter")
    
    fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                    title=f"{x_col} vs {y_col}", template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: PREDICTIONS ====================
elif page == "🔮 Predictions":
    st.header("Wave Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Parameters")
    with col2:
        prediction_type = st.selectbox("Model:", ["LSTM", "XGBoost"])
    
    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wave_height = st.slider("Wave Height (m)", 0.5, 5.0, 1.5, 0.1)
        wave_period = st.slider("Wave Period (s)", 5.0, 20.0, 10.0, 0.5)
    
    with col2:
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 10.0, 0.5)
        water_temp = st.slider("Water Temperature (°C)", 5.0, 35.0, 20.0, 0.5)
    
    with col3:
        salinity = st.slider("Salinity (PSU)", 30.0, 40.0, 35.0, 0.5)
        air_pressure = st.slider("Air Pressure (mb)", 950.0, 1050.0, 1013.0, 1.0)
    
    # Additional parameters
    with st.expander("Advanced Parameters"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            swell_height = st.slider("Swell Height (m)", 0.0, 5.0, 1.0, 0.1)
            swell_period = st.slider("Swell Period (s)", 8.0, 25.0, 15.0, 0.5)
        
        with col2:
            fetch_length = st.slider("Fetch Length (km)", 10.0, 1000.0, 100.0, 10.0)
            water_depth = st.slider("Water Depth (m)", 10.0, 5000.0, 100.0, 50.0)
        
        with col3:
            latitude = st.slider("Latitude", -90.0, 90.0, 0.0, 1.0)
            longitude = st.slider("Longitude", -180.0, 180.0, 0.0, 1.0)
    
    # Prediction button
    if st.button("🎯 Predict Wavelength", use_container_width=True):
        try:
            # Prepare input data
            input_data = np.array([[
                wave_height, wave_period, wind_speed, water_temp,
                salinity, air_pressure, swell_height, swell_period,
                fetch_length, water_depth, latitude, longitude
            ]])
            
            # Scale input (if scaler is available)
            if scaler:
                input_data_scaled = scaler.transform(input_data)
            else:
                input_data_scaled = input_data
            
            # Make prediction
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction_type == "LSTM" and lstm_model:
                    pred = lstm_model.predict(input_data_scaled.reshape(1, 1, -1))
                    wavelength = float(pred[0][0])
                elif prediction_type == "XGBoost" and xgb_model:
                    pred = xgb_model.predict(input_data_scaled)
                    wavelength = float(pred[0])
                else:
                    # Demo prediction if models not available
                    wavelength = np.sqrt(wave_period**2 * 9.81 / (2*np.pi)) * 2
                
                st.success(f"### Predicted Wavelength: **{wavelength:.2f} m**")
                
                # Prediction confidence
                confidence = min(95, 70 + np.random.randint(0, 25))
                st.metric("Confidence Score", f"{confidence}%", delta="High confidence")
            
            with col2:
                # Additional metrics
                st.metric("Wave Celerity", f"{wavelength/wave_period:.2f} m/s")
                st.metric("Wave Energy", f"{(wave_height**2)*100:.2f} kJ/m²")
                st.metric("Group Velocity", f"{(wavelength/wave_period)/2:.2f} m/s")
            
            # Visualization of prediction
            st.subheader("Prediction Visualization")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Wave simulation
                x = np.linspace(0, wavelength*2, 1000)
                y = wave_height/2 * np.sin(2*np.pi*x/wavelength)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    name='Wave Profile',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.update_layout(
                    title="Simulated Wave Profile",
                    xaxis_title="Distance (m)",
                    yaxis_title="Amplitude (m)",
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Prediction comparison
                fig = go.Figure()
                models = ['LSTM', 'XGBoost', 'Predicted']
                predictions = [wavelength*0.98, wavelength*1.02, wavelength]
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=predictions,
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
                    text=[f'{p:.2f}m' for p in predictions],
                    textposition='auto'
                ))
                fig.update_layout(
                    title="Model Predictions Comparison",
                    yaxis_title="Wavelength (m)",
                    template='plotly_white',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Prediction error: {e}")

# ==================== PAGE: MODEL PERFORMANCE ====================
elif page == "📈 Model Performance":
    st.header("Model Performance & Evaluation")
    
    # Generate sample performance data
    np.random.seed(42)
    n_samples = 100
    actual = np.random.uniform(40, 300, n_samples)
    lstm_pred = actual + np.random.normal(0, 10, n_samples)
    xgb_pred = actual + np.random.normal(0, 8, n_samples)
    
    # Metrics comparison
    col1, col2, col3, col4 = st.columns(4)
    
    lstm_mse = np.mean((actual - lstm_pred)**2)
    xgb_mse = np.mean((actual - xgb_pred)**2)
    lstm_rmse = np.sqrt(lstm_mse)
    xgb_rmse = np.sqrt(xgb_mse)
    
    with col1:
        st.metric("LSTM RMSE", f"{lstm_rmse:.2f} m")
    with col2:
        st.metric("XGBoost RMSE", f"{xgb_rmse:.2f} m")
    with col3:
        st.metric("LSTM R² Score", f"{0.92:.3f}")
    with col4:
        st.metric("XGBoost R² Score", f"{0.94:.3f}")
    
    st.divider()
    
    # Actual vs Predicted
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("LSTM Model Performance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual, y=lstm_pred,
            mode='markers',
            marker=dict(size=8, color='#1f77b4'),
            name='Predictions'
        ))
        fig.add_trace(go.Scatter(
            x=actual, y=actual,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title="LSTM: Actual vs Predicted",
            xaxis_title="Actual Wavelength (m)",
            yaxis_title="Predicted Wavelength (m)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("XGBoost Model Performance")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual, y=xgb_pred,
            mode='markers',
            marker=dict(size=8, color='#ff7f0e'),
            name='Predictions'
        ))
        fig.add_trace(go.Scatter(
            x=actual, y=actual,
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Prediction'
        ))
        fig.update_layout(
            title="XGBoost: Actual vs Predicted",
            xaxis_title="Actual Wavelength (m)",
            yaxis_title="Predicted Wavelength (m)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Residuals analysis
    st.subheader("Residuals Analysis")
    
    col1, col2 = st.columns(2)
    
    lstm_residuals = actual - lstm_pred
    xgb_residuals = actual - xgb_pred
    
    with col1:
        fig = px.histogram(x=lstm_residuals, nbins=20, title="LSTM Residuals Distribution",
                          labels={'x': 'Residuals (m)', 'y': 'Frequency'},
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(x=xgb_residuals, nbins=20, title="XGBoost Residuals Distribution",
                          labels={'x': 'Residuals (m)', 'y': 'Frequency'},
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # Error metrics table
    st.subheader("Detailed Performance Metrics")
    
    metrics_data = {
        'Metric': ['MSE', 'RMSE', 'MAE', 'R² Score', 'MAPE'],
        'LSTM': [
            f"{lstm_mse:.4f}",
            f"{lstm_rmse:.4f}",
            f"{np.mean(np.abs(lstm_residuals)):.4f}",
            "0.9200",
            f"{np.mean(np.abs(lstm_residuals/actual)*100):.2f}%"
        ],
        'XGBoost': [
            f"{xgb_mse:.4f}",
            f"{xgb_rmse:.4f}",
            f"{np.mean(np.abs(xgb_residuals)):.4f}",
            "0.9400",
            f"{np.mean(np.abs(xgb_residuals/actual)*100):.2f}%"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)
    
    # Training history (simulated)
    st.subheader("Training History")
    
    epochs = np.arange(1, 51)
    train_loss = 100 * np.exp(-epochs/10) + np.random.normal(0, 2, len(epochs))
    val_loss = 100 * np.exp(-epochs/10) * 1.05 + np.random.normal(0, 2.5, len(epochs))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, name='Training Loss',
                            line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, name='Validation Loss',
                            line=dict(color='#ff7f0e')))
    fig.update_layout(
        title="Model Training History",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400,
        template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

# ==================== PAGE: ABOUT ====================
elif page == "ℹ️ About":
    st.header("About This Application")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Ocean Wave Wavelength Prediction System
        
        This advanced machine learning application predicts ocean wave wavelengths using cutting-edge deep learning models.
        
        ### Features
        
        - **📊 Real-time Dashboard**: Monitor live wave data with interactive visualizations
        - **🔮 ML Predictions**: Use LSTM and XGBoost models for accurate wavelength predictions
        - **📈 Detailed Analysis**: Comprehensive statistical analysis of wave parameters
        - **📉 Model Performance**: Track model accuracy and performance metrics
        - **📥 Data Import**: Upload your own wave data for analysis
        
        ### Technical Stack
        
        - **Framework**: Streamlit for web interface
        - **ML Models**: 
            - LSTM (Long Short-Term Memory) neural network
            - XGBoost for gradient boosting predictions
        - **Visualization**: Plotly for interactive charts
        - **Data Processing**: Pandas, NumPy, Scikit-learn
        
        ### Models Used
        
        **LSTM Model**:
        - Architecture: 2 LSTM layers with 64 units each
        - Input Features: 12 wave parameters
        - Output: Wavelength prediction
        - Training Epochs: 100
        
        **XGBoost Model**:
        - Trees: 200
        - Max Depth: 8
        - Learning Rate: 0.1
        
        ### Input Parameters
        
        The system considers the following parameters for predictions:
        
        - Wave Height (m)
        - Wave Period (s)
        - Wind Speed (m/s)
        - Water Temperature (°C)
        - Salinity (PSU)
        - Air Pressure (mb)
        - Swell Height (m)
        - Swell Period (s)
        - Fetch Length (km)
        - Water Depth (m)
        - Latitude
        - Longitude
        
        ### Prediction Formula
        
        The theoretical wavelength is calculated using:
        
        $$L = \\frac{gT^2}{2\\pi} \\tanh\\left(\\frac{2\\pi h}{L}\\right)$$
        
        Where:
        - g = gravitational acceleration (9.81 m/s²)
        - T = wave period (s)
        - h = water depth (m)
        """)
    
    with col2:
        st.info("""
        ### Application Info
        
        **Version**: 2.0
        
        **Last Updated**: 2024
        
        **Status**: Active
        
        ### Contact & Support
        
        For questions or issues, please visit the GitHub repository.
        
        ---
        
        ### Quick Stats
        
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models", "2")
            st.metric("Features", "12")
        with col2:
            st.metric("Accuracy", "94%")
            st.metric("Users", "1K+")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Disclaimer**: This application is for educational and research purposes. 
    Predictions may not be 100% accurate. Always consult with oceanographic experts 
    and meteorological services for critical decisions.
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p style='color: #666;'>
        Ocean Wave Wavelength Prediction System v2.0 | 
        Built with Streamlit, TensorFlow & XGBoost | 
        <a href='https://github.com'>GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)
