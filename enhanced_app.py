import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Configure page
st.set_page_config(page_title="Ocean Wave Prediction", page_icon="🌊", layout="wide")

st.title("🌊 Ocean Wave Wavelength Prediction System")

# Navigation
page = st.sidebar.radio("Select Page:", ["Dashboard", "Analysis", "Predictions", "About"])

# Generate sample data
@st.cache_data
def generate_sample_data(days=90):
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

df = generate_sample_data()

# PAGE 1: DASHBOARD
if page == "Dashboard":
    st.header("📊 Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Wave Height (m)", f"{df['Wave_Height'].mean():.2f}")
    col2.metric("Avg Wavelength (m)", f"{df['Wavelength'].mean():.2f}")
    col3.metric("Avg Wave Period (s)", f"{df['Wave_Period'].mean():.2f}")
    col4.metric("Avg Wind Speed (m/s)", f"{df['Wind_Speed'].mean():.2f}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Wave Height Trend")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Wave_Height'], mode='lines', fill='tozeroy'))
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Wavelength Distribution")
        fig = px.histogram(df, x='Wavelength', nbins=30)
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# PAGE 2: ANALYSIS
elif page == "Analysis":
    st.header("📈 Data Analysis")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.write(df.describe())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Matrix")
        corr = df[['Wave_Height', 'Wave_Period', 'Wind_Speed', 'Water_Temperature', 'Wavelength']].corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Distribution Analysis")
        fig = px.histogram(df, x='Wave_Height', nbins=20, title='Wave Height Distribution')
        fig.update_layout(height=400, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# PAGE 3: PREDICTIONS
elif page == "Predictions":
    st.header("🔮 Wave Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        wave_height = st.slider("Wave Height (m)", 0.5, 5.0, 1.5)
        wave_period = st.slider("Wave Period (s)", 5.0, 20.0, 10.0)
    
    with col2:
        wind_speed = st.slider("Wind Speed (m/s)", 0.0, 30.0, 10.0)
        water_temp = st.slider("Water Temperature (°C)", 5.0, 35.0, 20.0)
    
    with col3:
        salinity = st.slider("Salinity (PSU)", 30.0, 40.0, 35.0)
        air_pressure = st.slider("Air Pressure (mb)", 950.0, 1050.0, 1013.0)
    
    if st.button("🎯 Predict Wavelength", use_container_width=True):
        # Simple physics-based calculation
        g = 9.81
        wavelength = (g * wave_period**2) / (2*np.pi)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"### Predicted Wavelength: **{wavelength:.2f} m**")
            st.metric("Confidence", "85%", delta="Demo Mode")
        
        with col2:
            st.metric("Wave Celerity", f"{wavelength/wave_period:.2f} m/s")
            st.metric("Wave Energy", f"{(wave_height**2)*100:.2f} kJ/m²")
            st.metric("Group Velocity", f"{(wavelength/wave_period)/2:.2f} m/s")
        
        # Wave visualization
        x = np.linspace(0, wavelength*2, 1000)
        y = wave_height/2 * np.sin(2*np.pi*x/wavelength)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='#1f77b4', width=3)))
        fig.update_layout(
            title="Wave Profile",
            xaxis_title="Distance (m)",
            yaxis_title="Amplitude (m)",
            height=400,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

# PAGE 4: ABOUT
elif page == "About":
    st.header("ℹ️ About")
    st.markdown("""
    ## Ocean Wave Wavelength Prediction System
    
    This is a simplified demo of the wave prediction system.
    
    ### Features:
    - 📊 Real-time dashboard
    - 📈 Data analysis tools
    - 🔮 Wave predictions
    - 🌊 Wave visualizations
    
    ### Formula Used:
    L = (g × T²) / (2π)
        
    Where:
    - L = Wavelength (m)
    - g = 9.81 m/s²
    - T = Wave period (s)
    """)

st.divider()
st.markdown("<div style='text-align: center'><p>🌊 Ocean Wave Prediction System | Deployed on Streamlit Cloud</p></div>", unsafe_allow_html=True)
    For deep water waves:
