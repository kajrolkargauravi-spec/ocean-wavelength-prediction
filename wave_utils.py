"""
Utility functions for wave analysis and calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class WaveCalculator:
    """Calculate wave properties using oceanographic formulas"""
    
    G = 9.81  # Gravitational acceleration (m/s²)
    
    @staticmethod
    def calculate_wavelength(period, depth=None):
        """
        Calculate wavelength from wave period
        Using dispersion relation for waves
        
        Parameters:
        -----------
        period : float
            Wave period in seconds
        depth : float, optional
            Water depth in meters. If None, assumes deep water
        
        Returns:
        --------
        float : Wavelength in meters
        """
        if depth is None or depth > period**2 * WaveCalculator.G / (2*np.pi):
            # Deep water wave
            wavelength = (WaveCalculator.G * period**2) / (2*np.pi)
        else:
            # Shallow water - iterative solution
            k = 2*np.pi / 1  # Initial guess
            for _ in range(20):  # Iterations
                k_new = (2*np.pi)**2 / (WaveCalculator.G * period**2 * np.tanh(k * depth))
                k = k_new
            wavelength = 2*np.pi / k
        
        return wavelength
    
    @staticmethod
    def calculate_wave_speed(period, depth=None):
        """Calculate wave phase velocity"""
        wavelength = WaveCalculator.calculate_wavelength(period, depth)
        return wavelength / period
    
    @staticmethod
    def calculate_wave_energy(wave_height):
        """Calculate wave energy density (kJ/m²)"""
        return (9.81 * wave_height**2) / 16
    
    @staticmethod
    def calculate_wave_power(wave_height, period, wavelength=None):
        """Calculate wave power (kW/m)"""
        if wavelength is None:
            wavelength = WaveCalculator.calculate_wavelength(period)
        
        rho = 1025  # Seawater density (kg/m³)
        energy_density = (rho * WaveCalculator.G * wave_height**2) / 16
        wave_speed = wavelength / period
        power = energy_density * wave_speed
        
        return power / 1000  # Convert to kW/m
    
    @staticmethod
    def steepness(wave_height, wavelength=None, period=None):
        """
        Calculate wave steepness (H/L ratio)
        Important for determining breaking waves
        """
        if wavelength is None:
            wavelength = WaveCalculator.calculate_wavelength(period)
        
        return wave_height / wavelength
    
    @staticmethod
    def froude_number(wave_speed, water_depth):
        """Calculate Froude number for wave-current interaction"""
        return wave_speed / np.sqrt(WaveCalculator.G * water_depth)


class DataProcessor:
    """Process and clean wave data"""
    
    @staticmethod
    def clean_data(df, columns=None, remove_outliers=True, zscore_threshold=3):
        """
        Clean wave data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        columns : list
            Columns to clean (if None, cleans all numeric columns)
        remove_outliers : bool
            Whether to remove outliers
        zscore_threshold : float
            Z-score threshold for outlier detection
        
        Returns:
        --------
        pd.DataFrame : Cleaned dataframe
        """
        df_clean = df.copy()
        
        if columns is None:
            columns = df_clean.select_dtypes(include=[np.number]).columns
        
        # Remove NaN values
        df_clean = df_clean.dropna(subset=columns)
        
        # Remove outliers if requested
        if remove_outliers:
            for col in columns:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                mask = np.abs((df_clean[col] - mean) / std) < zscore_threshold
                df_clean = df_clean[mask]
        
        return df_clean
    
    @staticmethod
    def resample_data(df, period='1D', method='mean'):
        """
        Resample time series data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data with datetime index
        period : str
            Resampling period (e.g., '1D', '1H', '1W')
        method : str
            Aggregation method ('mean', 'median', 'max', 'min')
        
        Returns:
        --------
        pd.DataFrame : Resampled data
        """
        if 'Date' in df.columns:
            df = df.set_index('Date')
        
        if method == 'mean':
            return df.resample(period).mean()
        elif method == 'median':
            return df.resample(period).median()
        elif method == 'max':
            return df.resample(period).max()
        elif method == 'min':
            return df.resample(period).min()
    
    @staticmethod
    def normalize_data(df, columns=None, method='minmax'):
        """
        Normalize data
        
        Parameters:
        -----------
        df : pd.DataFrame
        columns : list
        method : str
            'minmax' or 'zscore'
        
        Returns:
        --------
        pd.DataFrame : Normalized data
        """
        df_norm = df.copy()
        
        if columns is None:
            columns = df_norm.select_dtypes(include=[np.number]).columns
        
        if method == 'minmax':
            for col in columns:
                min_val = df_norm[col].min()
                max_val = df_norm[col].max()
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            for col in columns:
                mean = df_norm[col].mean()
                std = df_norm[col].std()
                df_norm[col] = (df_norm[col] - mean) / std
        
        return df_norm
    
    @staticmethod
    def create_lagged_features(df, column, lags=[1, 2, 3, 7]):
        """Create lagged features for time series"""
        df_lagged = df.copy()
        
        for lag in lags:
            df_lagged[f'{column}_lag_{lag}'] = df_lagged[column].shift(lag)
        
        return df_lagged.dropna()
    
    @staticmethod
    def create_rolling_features(df, column, windows=[7, 14, 30]):
        """Create rolling window features"""
        df_rolling = df.copy()
        
        for window in windows:
            df_rolling[f'{column}_rolling_mean_{window}'] = df_rolling[column].rolling(window).mean()
            df_rolling[f'{column}_rolling_std_{window}'] = df_rolling[column].rolling(window).std()
        
        return df_rolling.dropna()


class WaveAnalyzer:
    """Analyze wave data and patterns"""
    
    @staticmethod
    def identify_swell(wave_height, wind_speed, threshold_height=1.5, max_wind=10):
        """
        Identify swell conditions (waves not generated by local wind)
        
        Returns:
        --------
        bool : True if likely swell
        """
        return wave_height > threshold_height and wind_speed < max_wind
    
    @staticmethod
    def calculate_significant_wave_height(wave_heights):
        """
        Calculate significant wave height (Hs)
        Defined as mean of highest 1/3 of waves
        """
        sorted_heights = np.sort(wave_heights)[::-1]
        n = len(sorted_heights) // 3
        return np.mean(sorted_heights[:n])
    
    @staticmethod
    def wave_spectrum(periods, energy_spectrum):
        """
        Analyze wave spectrum
        
        Returns:
        --------
        dict : Spectral parameters
        """
        # Energy-weighted mean period
        m0 = np.trapz(energy_spectrum, periods)
        m1 = np.trapz(periods * energy_spectrum, periods)
        mean_period = m1 / m0 if m0 > 0 else 0
        
        # Peak period
        peak_idx = np.argmax(energy_spectrum)
        peak_period = periods[peak_idx]
        
        # Peak frequency
        peak_frequency = 1 / peak_period if peak_period > 0 else 0
        
        return {
            'm0': m0,
            'm1': m1,
            'mean_period': mean_period,
            'peak_period': peak_period,
            'peak_frequency': peak_frequency
        }
    
    @staticmethod
    def weather_window_analysis(wind_forecast, wave_forecast, threshold_wind=15, threshold_wave=2):
        """
        Identify weather windows for operations
        
        Returns:
        --------
        list : Indices of suitable windows
        """
        suitable_windows = []
        in_window = False
        window_start = None
        
        for i, (wind, wave) in enumerate(zip(wind_forecast, wave_forecast)):
            if wind < threshold_wind and wave < threshold_wave:
                if not in_window:
                    in_window = True
                    window_start = i
            else:
                if in_window:
                    suitable_windows.append((window_start, i))
                    in_window = False
        
        if in_window:
            suitable_windows.append((window_start, len(wind_forecast)))
        
        return suitable_windows


class ForecastingTools:
    """Tools for wave forecasting"""
    
    @staticmethod
    def persistence_forecast(last_value, horizon=7):
        """Simple persistence forecast"""
        return np.array([last_value] * horizon)
    
    @staticmethod
    def linear_trend_forecast(data, horizon=7):
        """Forecast using linear trend"""
        x = np.arange(len(data))
        coeffs = np.polyfit(x, data, 1)
        forecast_x = np.arange(len(data), len(data) + horizon)
        return np.polyval(coeffs, forecast_x)
    
    @staticmethod
    def seasonal_decomposition_forecast(series, period=365, horizon=30):
        """
        Seasonal decomposition forecasting
        Assumes yearly seasonality
        """
        # Simple deseasonalization
        deseasonalized = []
        for i in range(len(series)):
            seasonal_idx = i % period
            seasonal_avg = np.mean([series[j] for j in range(seasonal_idx, len(series), period)])
            deseasonalized.append(series[i] / seasonal_avg if seasonal_avg > 0 else series[i])
        
        # Trend using last values
        trend_forecast = np.polyfit(np.arange(len(deseasonalized[-30:])), deseasonalized[-30:], 1)
        
        # Generate forecast
        forecast = []
        for i in range(horizon):
            seasonal_idx = (len(series) + i) % period
            seasonal_factor = np.mean([series[j] for j in range(seasonal_idx, len(series), period)])
            trend_value = np.polyval(trend_forecast, i)
            forecast.append(trend_value * seasonal_factor)
        
        return np.array(forecast)


# Validation functions
def validate_wave_data(df):
    """Validate wave data for quality"""
    issues = []
    
    # Check required columns
    required_cols = ['Wave_Height', 'Wave_Period', 'Wind_Speed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check physical constraints
    if 'Wave_Height' in df.columns:
        if (df['Wave_Height'] < 0).any():
            issues.append("Negative wave heights detected")
        if (df['Wave_Height'] > 30).any():
            issues.append("Unrealistically high wave heights")
    
    if 'Wave_Period' in df.columns:
        if (df['Wave_Period'] < 1).any():
            issues.append("Wave period too short")
        if (df['Wave_Period'] > 30).any():
            issues.append("Unrealistically long wave periods")
    
    return issues
