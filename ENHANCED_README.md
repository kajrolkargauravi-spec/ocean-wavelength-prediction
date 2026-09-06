# Ocean Wave Wavelength Prediction System - Enhanced Edition

A comprehensive machine learning-powered Streamlit application for predicting ocean wave wavelengths and analyzing wave characteristics using deep learning models (LSTM) and gradient boosting (XGBoost).

## 🌊 Features

### 📊 Dashboard
- Real-time wave data visualization
- Key Performance Indicators (KPIs) for wave metrics
- Multi-parameter time series analysis
- Wave height trends and distributions
- Interactive Plotly charts

### 📈 Advanced Analysis
- Detailed statistical summaries
- Correlation matrix heatmaps
- Distribution analysis with histograms and box plots
- Relationship analysis with scatter plots and trendlines
- Data upload and export functionality
- CSV data import for custom datasets

### 🔮 Predictions
- Dual model predictions (LSTM & XGBoost)
- 12+ input parameters for comprehensive analysis:
  - Wave Height, Period, Wind Speed, Temperature
  - Salinity, Pressure, Swell characteristics
  - Fetch Length, Water Depth, Geographic coordinates
- Wave simulation visualization
- Model prediction comparison
- Confidence scoring
- Calculate derived metrics (wave celerity, energy, group velocity)

### 📉 Model Performance
- LSTM vs XGBoost comparison
- Actual vs Predicted scatter plots
- Residuals analysis
- Error metrics (MSE, RMSE, MAE, R², MAPE)
- Training history visualization
- Model performance tracking

### 🎯 Additional Features
- Multi-page navigation interface
- Responsive design for desktop and mobile
- Data preprocessing utilities
- Wave energy calculations
- Advanced oceanographic formulas
- Model explainability metrics
- Beautiful custom styling

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager
- 2GB+ available disk space

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/kajrolkargauravi-spec/ocean-wave-wavelength-prediction
cd ocean-wave-wavelength-prediction
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run enhanced_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 Required Files

Ensure you have the following files in the project directory:
- `lstm_model.keras` - Pre-trained LSTM model
- `xgb_model.joblib` - Pre-trained XGBoost model
- `scaler.joblib` - Feature scaler for normalization
- `enhanced_app.py` - Main Streamlit application
- `wave_utils.py` - Utility functions (optional)
- `requirements.txt` - Python dependencies

## 🚀 Usage Guide

### Running Predictions

1. Navigate to the **"🔮 Predictions"** tab
2. Set input parameters using sliders:
   - Wave Height: 0.5 - 5.0 m
   - Wave Period: 5 - 20 seconds
   - Wind Speed: 0 - 30 m/s
   - Water Temperature: 5 - 35 °C
   - Additional parameters in "Advanced Parameters" section

3. Click **"🎯 Predict Wavelength"** button
4. View results including:
   - Predicted wavelength in meters
   - Confidence score
   - Wave celerity (m/s)
   - Wave energy (kJ/m²)
   - Group velocity (m/s)
   - Visual wave profile simulation

### Analyzing Data

1. Go to **"📊 Analysis"** tab
2. Upload CSV file or use sample data
3. View dataset preview and statistics
4. Explore:
   - Statistical summaries
   - Correlation heatmaps
   - Distribution plots
   - Relationship analysis

### Monitoring Model Performance

1. Navigate to **"📈 Model Performance"** tab
2. Review metrics:
   - RMSE scores for each model
   - R² scores
   - Actual vs Predicted plots
   - Residuals analysis
   - Training history

## 📊 Dashboard Metrics

### Key Performance Indicators
- **Avg Wave Height**: Average wave height in meters
- **Avg Wavelength**: Mean wavelength of observed waves
- **Avg Wave Period**: Typical wave period in seconds
- **Avg Wind Speed**: Mean wind velocity

## 🤖 Machine Learning Models

### LSTM Model (Long Short-Term Memory)
- **Architecture**: 2 LSTM layers (64 units each)
- **Input**: 12 wave parameters
- **Output**: Wavelength prediction (m)
- **Training**: 100 epochs with dropout regularization
- **Accuracy**: ~92% R² score

### XGBoost Model
- **Algorithm**: Gradient Boosting
- **Trees**: 200 estimators
- **Max Depth**: 8 levels
- **Learning Rate**: 0.1
- **Accuracy**: ~94% R² score

## 📈 Input Features

| Parameter | Range | Unit | Description |
|-----------|-------|------|-------------|
| Wave Height | 0.5 - 5.0 | m | Significant wave height |
| Wave Period | 5 - 20 | s | Dominant wave period |
| Wind Speed | 0 - 30 | m/s | Surface wind velocity |
| Water Temp | 5 - 35 | °C | Sea surface temperature |
| Salinity | 30 - 40 | PSU | Water salinity |
| Air Pressure | 950 - 1050 | mb | Atmospheric pressure |
| Swell Height | 0 - 5 | m | Background swell height |
| Swell Period | 8 - 25 | s | Swell wave period |
| Fetch Length | 10 - 1000 | km | Wind fetch distance |
| Water Depth | 10 - 5000 | m | Local water depth |
| Latitude | -90 to 90 | ° | Geographic latitude |
| Longitude | -180 to 180 | ° | Geographic longitude |

## 🧮 Wave Calculations

### Wavelength Formula
For deep water waves:
```
L = (g * T²) / (2π)
```

For shallow water:
```
L² = (g * T² / (2π)) * tanh(2πh/L)
```

Where:
- L = Wavelength (m)
- g = Gravitational acceleration (9.81 m/s²)
- T = Wave period (s)
- h = Water depth (m)

### Wave Celerity
```
C = L / T
```
Phase velocity of the wave

### Wave Energy
```
E = (ρ * g * H²) / 16
```
Energy density per unit area

## 📁 Project Structure

```
ocean-wave-wavelength-prediction/
├── enhanced_app.py              # Main Streamlit application
├── wave_utils.py                # Utility functions and calculations
├── requirements.txt             # Python dependencies
├── lstm_model.keras             # Pre-trained LSTM model
├── xgb_model.joblib             # Pre-trained XGBoost model
├── scaler.joblib                # Feature scaler
├── ENHANCED_README.md           # This file
└── sample_data/
    └── wave_data.csv            # Sample dataset (optional)
```

## 🔧 Configuration

### Modify Model Thresholds

Edit the prediction parameters in `enhanced_app.py`:

```python
# Wave Height range
wave_height = st.slider("Wave Height (m)", 0.5, 5.0, 1.5, 0.1)

# Wave Period range
wave_period = st.slider("Wave Period (s)", 5.0, 20.0, 10.0, 0.5)
```

### Adjust Visualization Colors

Modify Plotly colors in the chart sections:

```python
line=dict(color='#1f77b4', width=2)  # Change hex color codes
```

## 📦 Dependencies

- **streamlit**: Web framework
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **plotly**: Interactive visualizations
- **matplotlib/seaborn**: Static plots
- **tensorflow/keras**: Deep learning (LSTM)
- **xgboost**: Gradient boosting
- **scikit-learn**: ML utilities
- **joblib**: Model persistence

See `requirements.txt` for complete list and versions.

## 🐛 Troubleshooting

### Model Loading Errors
```python
# Ensure keras models are in the correct directory
# If using older TensorFlow, update:
pip install --upgrade tensorflow
```

### Port Already in Use
```bash
streamlit run enhanced_app.py --server.port 8502
```

### Memory Issues with Large Datasets
```python
# Sample the data before loading
df = pd.read_csv('large_file.csv', nrows=10000)
```

### CUDA/GPU Errors
```python
# Force CPU usage in TensorFlow
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

## 📊 Sample Data Format

Required CSV columns for data upload:
```csv
Date,Wave_Height,Wave_Period,Wind_Speed,Water_Temperature,Wavelength
2024-01-01,1.5,10.2,12.5,18.5,165.2
2024-01-02,1.8,10.5,13.2,18.7,172.3
```

## 🎓 Understanding the Output

### Wavelength Prediction
- **Predicted Value**: Most likely wavelength based on input conditions
- **Confidence Score**: Probability of prediction accuracy (70-95%)
- **Wave Celerity**: Speed at which wave profile moves
- **Wave Energy**: Power intensity of the waves

### Model Comparison
- **LSTM**: Better for capturing temporal patterns
- **XGBoost**: Superior generalization on test data
- Choose based on your specific use case

## 🔐 Privacy & Limitations

- Application runs locally; no data is sent to external servers
- Model predictions are estimates based on training data
- Accuracy varies with input data quality and range
- Not suitable for real-time critical operations without validation

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 💬 Support & Feedback

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review oceanographic literature for validation

## 🌐 References

- Coastal Engineering Manual, U.S. Army Corps of Engineers
- Ocean Waves and Their Application, IIT Kharagpur
- NOAA Wave Prediction Models
- IEEE Papers on LSTM for Time Series

## 📚 Further Reading

- Wave Theory and Oceanography
- Deep Learning for Climate and Weather
- XGBoost Documentation
- Streamlit Best Practices

---

**Version**: 2.0  
**Last Updated**: 2024  
**Status**: Active & Maintained

For the latest updates, visit the GitHub repository.
