"""
Advanced Features for Streamlit Ocean Wave Prediction App
Additional analysis tools and visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal, stats
import json
from datetime import datetime


class AdvancedVisualization:
    """Advanced plotting and visualization tools"""
    
    @staticmethod
    def create_3d_surface_plot(X, Y, Z, title="3D Surface"):
        """Create 3D surface plot"""
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
        fig.update_layout(
            title=title,
            autosize=True,
            height=600,
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Wavelength (m)'
            )
        )
        return fig
    
    @staticmethod
    def create_parallel_coordinates(df, dimensions, color_column=None):
        """Create parallel coordinates plot"""
        if color_column:
            fig = px.parallel_coordinates(
                df,
                dimensions=dimensions,
                color=color_column,
                color_continuous_scale='Viridis'
            )
        else:
            fig = px.parallel_coordinates(df, dimensions=dimensions)
        
        fig.update_layout(height=600)
        return fig
    
    @staticmethod
    def create_sunburst(df, labels, parents, values, title="Sunburst Chart"):
        """Create sunburst chart"""
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total"
        ))
        fig.update_layout(title=title, height=600)
        return fig
    
    @staticmethod
    def create_animated_scatter(df, x_col, y_col, animation_col, color_col=None):
        """Create animated scatter plot"""
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            animation_frame=animation_col,
            color=color_col,
            hover_data=df.columns,
            title=f"{x_col} vs {y_col} Over {animation_col}"
        )
        fig.update_layout(height=600)
        return fig
    
    @staticmethod
    def create_violin_plot(df, x_col, y_col, title="Distribution Comparison"):
        """Create violin plot for distribution comparison"""
        fig = px.violin(df, x=x_col, y=y_col, title=title, box=True, points="outliers")
        fig.update_layout(height=500)
        return fig
    
    @staticmethod
    def create_density_heatmap(df, x_col, y_col, title="Density Heatmap"):
        """Create 2D density heatmap"""
        fig = px.density_heatmap(
            df,
            x=x_col,
            y=y_col,
            nbinsx=30,
            nbinsy=30,
            title=title
        )
        fig.update_layout(height=600)
        return fig


class SignalProcessing:
    """Signal processing tools for wave analysis"""
    
    @staticmethod
    def apply_fft(signal_data, sampling_rate=1.0):
        """Perform Fast Fourier Transform"""
        n = len(signal_data)
        fft_values = np.fft.fft(signal_data)
        frequencies = np.fft.fftfreq(n, 1/sampling_rate)
        power = np.abs(fft_values)**2
        
        # Return only positive frequencies
        positive_freq_idx = frequencies > 0
        return frequencies[positive_freq_idx], power[positive_freq_idx]
    
    @staticmethod
    def wavelet_transform(signal_data, mother_wavelet='morlet', scales=None):
        """Perform continuous wavelet transform"""
        if scales is None:
            scales = np.arange(1, 129)
        
        from scipy import signal as sig
        coefficients, frequencies = sig.morlet2(signal_data, M=len(scales), w=scales)
        
        return coefficients, frequencies
    
    @staticmethod
    def butter_filter(data, cutoff, order=5, btype='low', sampling_rate=1.0):
        """Apply Butterworth filter"""
        from scipy import signal as sig
        
        nyquist = sampling_rate / 2
        normalized_cutoff = cutoff / nyquist
        
        b, a = sig.butter(order, normalized_cutoff, btype=btype)
        filtered = sig.filtfilt(b, a, data)
        
        return filtered
    
    @staticmethod
    def detect_peaks(data, height=None, distance=None):
        """Detect peaks in signal"""
        peaks, properties = signal.find_peaks(data, height=height, distance=distance)
        return peaks, properties
    
    @staticmethod
    def spectral_analysis(time_series, sampling_rate=1.0):
        """Perform spectral analysis"""
        frequencies, power = SignalProcessing.apply_fft(time_series, sampling_rate)
        
        # Find dominant frequency
        dominant_idx = np.argmax(power)
        dominant_freq = frequencies[dominant_idx]
        
        # Calculate spectral moments
        m0 = np.sum(power)  # Total energy
        m1 = np.sum(frequencies * power)
        m2 = np.sum((frequencies**2) * power)
        
        # Spectral width
        spectral_width = np.sqrt(m2/m0 - (m1/m0)**2)
        
        return {
            'frequencies': frequencies,
            'power': power,
            'dominant_frequency': dominant_freq,
            'total_energy': m0,
            'spectral_width': spectral_width,
            'mean_frequency': m1/m0
        }


class StatisticalAnalysis:
    """Advanced statistical analysis tools"""
    
    @staticmethod
    def outlier_detection(data, method='iqr', threshold=1.5):
        """Detect outliers using various methods"""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data))
            outliers = z_scores > threshold
        
        elif method == 'mad':  # Median Absolute Deviation
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            outliers = np.abs(modified_z_scores) > threshold
        
        return outliers
    
    @staticmethod
    def distribution_fitting(data):
        """Fit data to common distributions"""
        distributions = ['normal', 'gamma', 'exponential', 'weibull']
        results = {}
        
        try:
            # Normal distribution
            params = stats.norm.fit(data)
            ks_stat = stats.kstest(data, 'norm', args=params)[0]
            results['normal'] = {'params': params, 'ks_statistic': ks_stat}
        except:
            pass
        
        try:
            # Gamma distribution
            params = stats.gamma.fit(data)
            ks_stat = stats.kstest(data, 'gamma', args=params)[0]
            results['gamma'] = {'params': params, 'ks_statistic': ks_stat}
        except:
            pass
        
        try:
            # Weibull distribution
            params = stats.weibull_max.fit(data)
            ks_stat = stats.kstest(data, 'weibull_max', args=params)[0]
            results['weibull'] = {'params': params, 'ks_statistic': ks_stat}
        except:
            pass
        
        return results
    
    @staticmethod
    def correlation_analysis(df):
        """Perform detailed correlation analysis"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Pearson correlation
        pearson_corr = numeric_df.corr(method='pearson')
        
        # Spearman correlation
        spearman_corr = numeric_df.corr(method='spearman')
        
        # Kendall correlation
        kendall_corr = numeric_df.corr(method='kendall')
        
        return {
            'pearson': pearson_corr,
            'spearman': spearman_corr,
            'kendall': kendall_corr
        }
    
    @staticmethod
    def hypothesis_testing(data1, data2, test_type='ttest'):
        """Perform hypothesis testing"""
        if test_type == 'ttest':
            statistic, p_value = stats.ttest_ind(data1, data2)
            return {'test': 't-test', 'statistic': statistic, 'p_value': p_value}
        
        elif test_type == 'mannwhitneyu':
            statistic, p_value = stats.mannwhitneyu(data1, data2)
            return {'test': 'Mann-Whitney U', 'statistic': statistic, 'p_value': p_value}
        
        elif test_type == 'ks':
            statistic, p_value = stats.ks_2samp(data1, data2)
            return {'test': 'Kolmogorov-Smirnov', 'statistic': statistic, 'p_value': p_value}


class TimeSeriesAnalysis:
    """Time series analysis tools"""
    
    @staticmethod
    def autocorrelation(data, max_lag=50):
        """Calculate autocorrelation"""
        from pandas.plotting import autocorrelation_plot
        acf_values = []
        
        for lag in range(max_lag):
            if lag == 0:
                acf_values.append(1.0)
            else:
                c0 = np.mean((data - np.mean(data))**2)
                c_lag = np.mean((data[:-lag] - np.mean(data)) * (data[lag:] - np.mean(data)))
                acf_values.append(c_lag / c0)
        
        return np.array(acf_values)
    
    @staticmethod
    def seasonal_decomposition(data, period=12):
        """Simple seasonal decomposition"""
        from scipy import signal as sig
        
        # Extract trend using moving average
        trend = sig.savgol_filter(data, window_length=period if period % 2 == 1 else period+1, polyorder=2)
        
        # Extract seasonal component
        detrended = data - trend
        seasonal = np.zeros_like(data)
        for i in range(period):
            seasonal[i::period] = np.mean(detrended[i::period])
        
        # Residual
        residual = data - trend - seasonal
        
        return {
            'original': data,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    @staticmethod
    def forecast_metrics(actual, predicted):
        """Calculate various forecast accuracy metrics"""
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted)**2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted))) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'SMAPE': smape
        }


class AnomalyDetection:
    """Anomaly detection algorithms"""
    
    @staticmethod
    def isolation_forest(data, contamination=0.1):
        """Detect anomalies using Isolation Forest"""
        from sklearn.ensemble import IsolationForest
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        predictions = iso_forest.fit_predict(data)
        
        anomalies = predictions == -1
        return anomalies, iso_forest
    
    @staticmethod
    def local_outlier_factor(data, n_neighbors=20):
        """Detect anomalies using Local Outlier Factor"""
        from sklearn.neighbors import LocalOutlierFactor
        
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        predictions = lof.fit_predict(data)
        
        anomalies = predictions == -1
        scores = lof.negative_outlier_factor_
        
        return anomalies, scores
    
    @staticmethod
    def moving_average_anomaly(data, window=20, threshold=2.0):
        """Detect anomalies using moving average"""
        ma = pd.Series(data).rolling(window=window, center=True).mean()
        mstd = pd.Series(data).rolling(window=window, center=True).std()
        
        anomalies = (np.abs(data - ma) > threshold * mstd)
        
        return anomalies


class DataExporter:
    """Export analysis results in various formats"""
    
    @staticmethod
    def export_to_json(data, filename):
        """Export data to JSON"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        return filename
    
    @staticmethod
    def export_to_csv(data, filename):
        """Export data to CSV"""
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
        
        df.to_csv(filename, index=False)
        return filename
    
    @staticmethod
    def export_to_excel(data, filename, sheet_names=None):
        """Export data to Excel"""
        if isinstance(data, dict):
            with pd.ExcelWriter(filename) as writer:
                for i, (key, value) in enumerate(data.items()):
                    sheet_name = sheet_names[i] if sheet_names else key
                    pd.DataFrame(value).to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            data.to_excel(filename, index=False)
        
        return filename
    
    @staticmethod
    def create_report(analysis_results, output_file='report.html'):
        """Generate HTML report"""
        html_content = """
        <html>
        <head>
            <title>Wave Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #1f77b4; }
                h2 { color: #ff7f0e; border-bottom: 2px solid #ddd; padding: 10px 0; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
                th { background-color: #f2f2f2; }
                .metric { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>🌊 Ocean Wave Analysis Report</h1>
            <p><strong>Generated:</strong> {}</p>
        """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        for key, value in analysis_results.items():
            html_content += f"\n<h2>{key}</h2>\n"
            if isinstance(value, dict):
                html_content += "<table>\n<tr><th>Metric</th><th>Value</th></tr>\n"
                for k, v in value.items():
                    html_content += f"<tr><td>{k}</td><td>{v:.4f if isinstance(v, (int, float)) else v}</td></tr>\n"
                html_content += "</table>\n"
            elif isinstance(value, pd.DataFrame):
                html_content += value.to_html() + "\n"
            else:
                html_content += f"<div class='metric'>{value}</div>\n"
        
        html_content += "</body></html>"
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file


# Streamlit component functions
def render_advanced_analysis_dashboard():
    """Render advanced analysis dashboard"""
    st.header("🔬 Advanced Analysis Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Signal Processing", "Statistics", "Anomalies", "Time Series"])
    
    with tab1:
        st.subheader("Signal Processing Tools")
        st.write("Coming soon: FFT, Wavelet Analysis, Filtering")
    
    with tab2:
        st.subheader("Statistical Analysis")
        st.write("Coming soon: Distribution Fitting, Hypothesis Testing")
    
    with tab3:
        st.subheader("Anomaly Detection")
        st.write("Coming soon: Isolation Forest, Local Outlier Factor")
    
    with tab4:
        st.subheader("Time Series Analysis")
        st.write("Coming soon: Autocorrelation, Seasonal Decomposition, Forecasting")


if __name__ == "__main__":
    print("Advanced Features Module")
    print("This module provides advanced analysis tools for the Streamlit app")
