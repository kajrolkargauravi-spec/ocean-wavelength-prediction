"""
Model Training Script for Ocean Wave Wavelength Prediction
This script trains LSTM and XGBoost models on wave data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class WaveDataGenerator:
    """Generate synthetic wave data for training"""
    
    @staticmethod
    def generate_data(n_samples=1000, seed=42):
        """Generate realistic wave data"""
        np.random.seed(seed)
        
        # Generate base parameters
        wave_height = np.random.uniform(0.5, 3.5, n_samples)
        wave_period = np.random.uniform(5, 15, n_samples)
        wind_speed = np.random.uniform(0, 25, n_samples)
        water_temp = np.random.uniform(10, 25, n_samples)
        salinity = np.random.uniform(32, 38, n_samples)
        air_pressure = np.random.uniform(950, 1050, n_samples)
        
        # Additional parameters
        swell_height = wave_height * np.random.uniform(0.5, 1.5, n_samples)
        swell_period = wave_period * np.random.uniform(1.2, 1.8, n_samples)
        fetch_length = np.random.uniform(50, 500, n_samples)
        water_depth = np.random.uniform(50, 1000, n_samples)
        latitude = np.random.uniform(-60, 60, n_samples)
        longitude = np.random.uniform(-180, 180, n_samples)
        
        # Calculate wavelength using wave theory
        g = 9.81
        wavelength = (g * wave_period**2) / (2 * np.pi) * (1 + np.random.normal(0, 0.05, n_samples))
        
        # Create dataframe
        data = pd.DataFrame({
            'Wave_Height': wave_height,
            'Wave_Period': wave_period,
            'Wind_Speed': wind_speed,
            'Water_Temperature': water_temp,
            'Salinity': salinity,
            'Air_Pressure': air_pressure,
            'Swell_Height': swell_height,
            'Swell_Period': swell_period,
            'Fetch_Length': fetch_length,
            'Water_Depth': water_depth,
            'Latitude': latitude,
            'Longitude': longitude,
            'Wavelength': wavelength
        })
        
        return data


class LSTMModelTrainer:
    """Train LSTM model for wave prediction"""
    
    def __init__(self, input_features=12):
        self.input_features = input_features
        self.model = None
        self.scaler = MinMaxScaler()
    
    def build_model(self):
        """Build LSTM architecture"""
        model = Sequential([
            layers.LSTM(64, activation='relu', input_shape=(1, self.input_features),
                       return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(64, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Reshape for LSTM (samples, timesteps, features)
        X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_val_reshaped = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
        
        # Scale targets
        y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1))
        y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1))
        
        # Build model
        self.model = self.build_model()
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train
        history = self.model.fit(
            X_train_reshaped, y_train_scaled,
            validation_data=(X_val_reshaped, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        predictions = self.model.predict(X_reshaped)
        return predictions
    
    def save_model(self, filepath):
        """Save model to disk"""
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def save_scaler(self, filepath):
        """Save scaler to disk"""
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")


class XGBoostModelTrainer:
    """Train XGBoost model for wave prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dval = xgb.DMatrix(X_val_scaled, label=y_val)
        
        # Parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Train
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dval, 'validation')],
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)
        predictions = self.model.predict(dtest)
        return predictions
    
    def save_model(self, filepath):
        """Save model to disk"""
        if self.model:
            self.model.save_model(filepath)
            print(f"Model saved to {filepath}")
    
    def save_scaler(self, filepath):
        """Save scaler to disk"""
        joblib.dump(self.scaler, filepath)
        print(f"Scaler saved to {filepath}")


class ModelEvaluator:
    """Evaluate model performance"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """Calculate performance metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
    
    @staticmethod
    def plot_results(y_true, y_pred, title="Model Performance"):
        """Plot actual vs predicted"""
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], 
                [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Wavelength (m)')
        plt.ylabel('Predicted Wavelength (m)')
        plt.title(f'{title} - Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Residuals
        plt.subplot(1, 2, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Wavelength (m)')
        plt.ylabel('Residuals (m)')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt
    
    @staticmethod
    def print_metrics(metrics, model_name="Model"):
        """Print metrics summary"""
        print(f"\n{'='*50}")
        print(f"{model_name} Performance Metrics")
        print(f"{'='*50}")
        for metric, value in metrics.items():
            if metric == 'MAPE':
                print(f"{metric:10s}: {value:10.2f}%")
            else:
                print(f"{metric:10s}: {value:10.4f}")
        print(f"{'='*50}\n")


def main():
    """Main training script"""
    print("="*60)
    print("Ocean Wave Wavelength Prediction - Model Training")
    print("="*60)
    
    # Generate data
    print("\n1. Generating synthetic wave data...")
    data_gen = WaveDataGenerator()
    df = data_gen.generate_data(n_samples=2000)
    print(f"   Generated {len(df)} samples")
    print(f"   Features: {df.shape[1]}")
    
    # Prepare data
    print("\n2. Preparing data...")
    X = df.drop('Wavelength', axis=1).values
    y = df['Wavelength'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Validation set: {len(X_val)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Train LSTM
    print("\n3. Training LSTM Model...")
    print("-" * 60)
    lstm_trainer = LSTMModelTrainer(input_features=X.shape[1])
    lstm_history = lstm_trainer.train(X_train, y_train, X_val, y_val, epochs=100)
    
    print("\n4. LSTM Predictions...")
    lstm_pred = lstm_trainer.predict(X_test)
    lstm_metrics = ModelEvaluator.calculate_metrics(y_test, lstm_pred.flatten())
    ModelEvaluator.print_metrics(lstm_metrics, "LSTM Model")
    
    # Train XGBoost
    print("\n5. Training XGBoost Model...")
    print("-" * 60)
    xgb_trainer = XGBoostModelTrainer()
    xgb_trainer.train(X_train, y_train, X_val, y_val)
    
    print("\n6. XGBoost Predictions...")
    xgb_pred = xgb_trainer.predict(X_test)
    xgb_metrics = ModelEvaluator.calculate_metrics(y_test, xgb_pred)
    ModelEvaluator.print_metrics(xgb_metrics, "XGBoost Model")
    
    # Save models
    print("\n7. Saving Models...")
    lstm_trainer.save_model('lstm_model.keras')
    lstm_trainer.save_scaler('scaler.joblib')
    
    xgb_trainer.save_model('xgb_model.joblib')
    xgb_trainer.save_scaler('xgb_scaler.joblib')
    
    print("\n✅ Training Complete!")
    print("="*60)
    
    # Plot results
    print("\n8. Generating visualizations...")
    
    fig1 = ModelEvaluator.plot_results(y_test, lstm_pred.flatten(), "LSTM Model")
    fig1.savefig('lstm_performance.png', dpi=300, bbox_inches='tight')
    
    fig2 = ModelEvaluator.plot_results(y_test, xgb_pred, "XGBoost Model")
    fig2.savefig('xgb_performance.png', dpi=300, bbox_inches='tight')
    
    print("✅ Visualizations saved!")
    print("\nYou can now run the Streamlit app:")
    print("  streamlit run enhanced_app.py")


if __name__ == "__main__":
    main()
