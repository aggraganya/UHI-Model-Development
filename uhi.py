import pandas as pd
import numpy as np
import json
import rasterio
from rasterio.transform import from_bounds
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UrbanHeatIslandModel:
    def __init__(self):
        """Initialize the UHI Model with default parameters."""
        self.model = None
        self.feature_names = [
            'impervious_surface_percentage',
            'ndvi',
            'ndwi',
            'population_density',
            'surface_albedo',
            'elevation',
            'slope',
            'nighttime_lights',
            'building_density',
            'distance_to_highway'
        ]
        self.feature_importance = {}
        self.training_metrics = {}
        
    def generate_synthetic_data(self, n_samples=10000, noise_level=0.1):
        """
        Generate synthetic training data for demonstration purposes.
        In real implementation, this would be replaced with actual satellite data.
        """
        np.random.seed(42)
        
        # Generate feature data
        data = {}
        
        # Natural factors
        data['ndvi'] = np.random.uniform(-0.5, 1.0, n_samples)  # Vegetation index
        data['ndwi'] = np.random.uniform(-1.0, 1.0, n_samples)  # Water index
        data['elevation'] = np.random.uniform(0, 3000, n_samples)  # Elevation in meters
        data['slope'] = np.random.exponential(2, n_samples)  # Slope in degrees
        data['surface_albedo'] = np.random.uniform(0.1, 0.4, n_samples)  # Surface reflectance
        
        # Anthropogenic factors
        data['impervious_surface_percentage'] = np.random.uniform(0, 100, n_samples)
        data['population_density'] = np.random.exponential(100, n_samples)  # People per km²
        data['building_density'] = np.random.uniform(0, 1, n_samples)
        data['nighttime_lights'] = np.random.exponential(10, n_samples)
        data['distance_to_highway'] = np.random.exponential(1000, n_samples)  # Distance in meters
        
        # Generate UHI intensity based on realistic relationships
        uhi_intensity = (
            0.45 * (data['impervious_surface_percentage'] / 100) +  # Positive correlation
            -0.32 * np.clip(data['ndvi'], 0, 1) +  # Negative correlation (more vegetation = less heat)
            0.15 * np.log1p(data['population_density']) / 10 +  # Log relationship with population
            -0.08 * data['surface_albedo'] * 10 +  # Higher albedo = less heat
            0.1 * (data['building_density']) +
            0.05 * np.log1p(data['nighttime_lights']) / 5 +
            -0.02 * np.log1p(data['distance_to_highway']) / 10 +
            np.random.normal(0, noise_level, n_samples)  # Add noise
        )
        
        # Ensure UHI intensity is realistic (0-10°C)
        uhi_intensity = np.clip(uhi_intensity, 0, 10)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df['uhi_intensity'] = uhi_intensity
        
        # Add coordinate information for mapping
        df['lat'] = np.random.uniform(-60, 75, n_samples)  # Exclude polar regions
        df['lon'] = np.random.uniform(-180, 180, n_samples)
        
        # Add land cover type for context
        land_cover_types = ['urban', 'suburban', 'rural', 'desert', 'forest', 'water']
        weights = [0.3, 0.2, 0.25, 0.1, 0.1, 0.05]  # Urban areas more common in dataset
        df['land_cover'] = np.random.choice(land_cover_types, n_samples, p=weights)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for model training."""
        # Handle missing values
        df_processed = df.copy()
        
        # Fill missing values with median for numerical columns
        for col in self.feature_names:
            if col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Create interaction features
        df_processed['urban_vegetation_ratio'] = (
            df_processed['impervious_surface_percentage'] / 
            (df_processed['ndvi'] + 0.01)  # Add small value to avoid division by zero
        )
        
        df_processed['heat_capacity_proxy'] = (
            df_processed['impervious_surface_percentage'] * 
            df_processed['building_density']
        )
        
        # Add these new features to feature list
        additional_features = ['urban_vegetation_ratio', 'heat_capacity_proxy']
        self.feature_names.extend(additional_features)
        
        return df_processed
    
    def train_model(self, df, model_type='random_forest'):
        """Train the UHI prediction model."""
        # Preprocess data
        df_processed = self.preprocess_data(df)
        
        # Prepare features and target
        X = df_processed[self.feature_names]
        y = df_processed['uhi_intensity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Calculate metrics
        self.training_metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'cv_scores': cross_val_score(self.model, X, y, cv=5, scoring='r2')
        }
        
        # Extract feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            self.feature_importance = {
                feature: float(score) 
                for feature, score in zip(self.feature_names, importance_scores)
            }
        
        print("Model Training Complete!")
        print(f"Test R²: {self.training_metrics['test_r2']:.3f}")
        print(f"Test RMSE: {self.training_metrics['test_rmse']:.3f}°C")
        print(f"CV R² (mean): {np.mean(self.training_metrics['cv_scores']):.3f} ± {np.std(self.training_metrics['cv_scores']):.3f}")
        
        return X_test, y_test, y_pred_test
    
    def generate_factor_importance_output(self, output_path='factor_importance.json'):
        """Generate the factor importance ranking as specified in the requirements."""
        if not self.feature_importance:
            raise ValueError("Model must be trained before generating factor importance.")
        
        # Sort features by importance (descending)
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Create the output format as specified
        factor_importance_output = {
            "factor_importance": [
                {
                    "factor": factor,
                    "score": score
                }
                for factor, score in sorted_features
            ],
            "model_metadata": {
                "model_type": type(self.model).__name__,
                "training_date": datetime.now().isoformat(),
                "performance_metrics": self.training_metrics
            }
        }
        
        # Save to JSON file
        with open(output_path, 'w') as f:
            json.dump(factor_importance_output, f, indent=2)
        
        print(f"Factor importance saved to {output_path}")
        
        return factor_importance_output
    
    def predict_uhi_intensity(self, feature_data):
        """Predict UHI intensity for new data."""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")
        
        # Ensure feature_data has all required features
        missing_features = set(self.feature_names) - set(feature_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X = feature_data[self.feature_names]
        predictions = self.model.predict(X)
        
        return predictions
    
    def create_synthetic_raster_output(self, predictions, coordinates, output_path='uhi_map.tif'):
        """
        Create a synthetic GeoTIFF raster output for demonstration.
        In real implementation, this would use actual geographic coordinates and projections.
        """
        # Create a simple grid for demonstration
        lat_min, lat_max = coordinates['lat'].min(), coordinates['lat'].max()
        lon_min, lon_max = coordinates['lon'].min(), coordinates['lon'].max()
        
        # Create a 100x100 grid for demonstration
        grid_size = 100
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)
        
        # Interpolate predictions to grid (simplified approach)
        from scipy.interpolate import griddata
        
        points = np.column_stack([coordinates['lon'], coordinates['lat']])
        grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
        
        # Interpolate UHI values to grid
        uhi_grid = griddata(points, predictions, grid_points, method='linear', fill_value=0)
        uhi_grid = uhi_grid.reshape(grid_size, grid_size)
        
        # Create transform for GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, grid_size, grid_size)
        
        # Write GeoTIFF
        with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=grid_size,
            width=grid_size,
            count=1,
            dtype=uhi_grid.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(uhi_grid, 1)
        
        print(f"UHI raster map saved to {output_path}")
        
        return uhi_grid, transform
    
    def create_highway_heat_map(self, df, highway_buffer_km=0.5, output_path='highway_heat_map.tif'):
        """
        Create highway-specific heat island map.
        This is a simplified version - real implementation would use actual highway data.
        """
        # Filter data points near highways (simplified: distance_to_highway < buffer)
        highway_buffer_m = highway_buffer_km * 1000
        highway_data = df[df['distance_to_highway'] <= highway_buffer_m].copy()
        
        if len(highway_data) == 0:
            print("No data points found near highways with current buffer.")
            return None
        
        # Predict UHI for highway areas
        highway_predictions = self.predict_uhi_intensity(highway_data)
        highway_coordinates = highway_data[['lat', 'lon']]
        
        # Create raster (similar to main UHI map)
        highway_raster, transform = self.create_synthetic_raster_output(
            highway_predictions, 
            highway_coordinates, 
            output_path
        )
        
        print(f"Highway heat map saved to {output_path}")
        print(f"Highway buffer: {highway_buffer_km} km")
        print(f"Data points near highways: {len(highway_data)}")
        
        return highway_raster, transform
    
    def visualize_results(self, df, predictions):
        """Create visualizations of the results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Feature importance plot
        if self.feature_importance:
            features = list(self.feature_importance.keys())
            scores = list(self.feature_importance.values())
            
            axes[0, 0].barh(features, scores)
            axes[0, 0].set_xlabel('Importance Score')
            axes[0, 0].set_title('Feature Importance Rankings')
            axes[0, 0].grid(True, alpha=0.3)
        
        # UHI intensity distribution
        axes[0, 1].hist(predictions, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('UHI Intensity (°C)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Predicted UHI Intensity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot: Impervious surface vs UHI
        axes[1, 0].scatter(
            df['impervious_surface_percentage'], 
            predictions, 
            alpha=0.5, 
            s=1
        )
        axes[1, 0].set_xlabel('Impervious Surface Percentage (%)')
        axes[1, 0].set_ylabel('UHI Intensity (°C)')
        axes[1, 0].set_title('Impervious Surface vs UHI Intensity')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter plot: NDVI vs UHI
        axes[1, 1].scatter(df['ndvi'], predictions, alpha=0.5, s=1)
        axes[1, 1].set_xlabel('NDVI (Vegetation Index)')
        axes[1, 1].set_ylabel('UHI Intensity (°C)')
        axes[1, 1].set_title('Vegetation Index vs UHI Intensity')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('uhi_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

# Example usage and demonstration
def main():
    """Main function demonstrating the complete UHI modeling workflow."""
    
    print("=== Urban Heat Island Model Demo ===\n")
    
    # Initialize model
    uhi_model = UrbanHeatIslandModel()
    
    # Generate synthetic data (replace with real data loading in production)
    print("1. Generating synthetic training data...")
    training_data = uhi_model.generate_synthetic_data(n_samples=10000)
    print(f"Generated {len(training_data)} training samples")
    print(f"Features: {list(training_data.columns)}")
    
    # Train the model
    print("\n2. Training the UHI prediction model...")
    X_test, y_test, y_pred_test = uhi_model.train_model(training_data, model_type='random_forest')
    
    # Generate factor importance output (as specified in requirements)
    print("\n3. Generating factor importance ranking...")
    factor_output = uhi_model.generate_factor_importance_output('uhi_factor_importance.json')
    
    # Display top factors
    print("\nTop 5 Most Important Factors:")
    for i, factor in enumerate(factor_output['factor_importance'][:5]):
        print(f"{i+1}. {factor['factor']}: {factor['score']:.3f}")
    
    # Create predictions for visualization
    print("\n4. Creating predictions for full dataset...")
    all_predictions = uhi_model.predict_uhi_intensity(training_data)
    
    # Create raster outputs
    print("\n5. Creating GeoTIFF raster outputs...")
    coordinates = training_data[['lat', 'lon']]
    uhi_raster, transform = uhi_model.create_synthetic_raster_output(
        all_predictions, 
        coordinates, 
        'global_uhi_map.tif'
    )
    
    # Create highway heat map
    print("\n6. Creating highway heat island map...")
    highway_raster, highway_transform = uhi_model.create_highway_heat_map(
        training_data, 
        highway_buffer_km=0.5,
        output_path='highway_heat_map.tif'
    )
    
    # Create visualizations
    print("\n7. Creating analysis visualizations...")
    uhi_model.visualize_results(training_data, all_predictions)
    
    # Summary statistics
    print(f"\n=== Model Summary ===")
    print(f"Model Type: {type(uhi_model.model).__name__}")
    print(f"Training Samples: {len(training_data)}")
    print(f"Features Used: {len(uhi_model.feature_names)}")
    print(f"Test R²: {uhi_model.training_metrics['test_r2']:.3f}")
    print(f"Mean UHI Intensity: {np.mean(all_predictions):.2f}°C")
    print(f"Max UHI Intensity: {np.max(all_predictions):.2f}°C")
    
    # Create summary CSV
    summary_data = {
        'sample_id': range(len(training_data)),
        'latitude': training_data['lat'],
        'longitude': training_data['lon'],
        'predicted_uhi_intensity': all_predictions,
        'land_cover': training_data['land_cover'],
        'impervious_surface_pct': training_data['impervious_surface_percentage'],
        'ndvi': training_data['ndvi'],
        'population_density': training_data['population_density']
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('uhi_predictions_summary.csv', index=False)
    print(f"\nDetailed results saved to: uhi_predictions_summary.csv")
    
    return uhi_model, training_data, all_predictions

if __name__ == "__main__":
    # Run the complete workflow
    model, data, predictions = main()
