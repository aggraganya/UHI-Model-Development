import pandas as pd
import numpy as np
import json
import rasterio
from rasterio.transform import from_bounds
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class UrbanHeatIslandModel:
    def __init__(self):
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
    
        np.random.seed(42)
        data = {}
        
        data['ndvi'] = np.random.uniform(-0.5, 1.0, n_samples)  # Vegetation index
        data['ndwi'] = np.random.uniform(-1.0, 1.0, n_samples)  # Water index
        data['elevation'] = np.random.uniform(0, 3000, n_samples)  # Elevation in meters
        data['slope'] = np.random.exponential(2, n_samples)  # Slope in degrees
        data['surface_albedo'] = np.random.uniform(0.1, 0.4, n_samples)  # Surface reflectance
        
        data['impervious_surface_percentage'] = np.random.uniform(0, 100, n_samples)
        data['population_density'] = np.random.exponential(100, n_samples)  # People per km²
        data['building_density'] = np.random.uniform(0, 1, n_samples)
        data['nighttime_lights'] = np.random.exponential(10, n_samples)
        data['distance_to_highway'] = np.random.exponential(1000, n_samples)  # Distance in meters
        
        uhi_intensity = (
            0.45 * (data['impervious_surface_percentage'] / 100) + 
            -0.32 * np.clip(data['ndvi'], 0, 1) +  
            0.15 * np.log1p(data['population_density']) / 10 + 
            -0.08 * data['surface_albedo'] * 10 +  
            0.1 * (data['building_density']) +
            0.05 * np.log1p(data['nighttime_lights']) / 5 +
            -0.02 * np.log1p(data['distance_to_highway']) / 10 +
            np.random.normal(0, noise_level, n_samples) 
        )
        
        uhi_intensity = np.clip(uhi_intensity, 0, 10)
        
        df = pd.DataFrame(data)
        df['uhi_intensity'] = uhi_intensity
        
        df['lat'] = np.random.uniform(-60, 75, n_samples)  # Exclude polar regions
        df['lon'] = np.random.uniform(-180, 180, n_samples)
        
        land_cover_types = ['urban', 'suburban', 'rural', 'desert', 'forest', 'water']
        weights = [0.3, 0.2, 0.25, 0.1, 0.1, 0.05] 
        df['land_cover'] = np.random.choice(land_cover_types, n_samples, p=weights)
        
        return df
    
    def preprocess_data(self, df):
        df_processed = df.copy()

        for col in self.feature_names:
            if col in df_processed.columns:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)

        df_processed['urban_vegetation_ratio'] = (
            df_processed['impervious_surface_percentage'] / 
            (df_processed['ndvi'] + 0.01)  
        )
        
        df_processed['heat_capacity_proxy'] = (
            df_processed['impervious_surface_percentage'] * 
            df_processed['building_density']
        )

        additional_features = ['urban_vegetation_ratio', 'heat_capacity_proxy']
        self.feature_names.extend(additional_features)
        
        return df_processed
    
    def train_model(self, df, model_type='random_forest'):
        df_processed = self.preprocess_data(df)
        
        X = df_processed[self.feature_names]
        y = df_processed['uhi_intensity']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        self.model.fit(X_train, y_train)

        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)

        self.training_metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'cv_scores': cross_val_score(self.model, X, y, cv=5, scoring='r2')
        }
        
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
        
        if not self.feature_importance:
            raise ValueError("Model must be trained before generating factor importance.")
        
        sorted_features = sorted(
            self.feature_importance.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )

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

        with open(output_path, 'w') as f:
            json.dump(factor_importance_output, f, indent=2)
        
        print(f"Factor importance saved to {output_path}")
        
        return factor_importance_output 
    
    def predict_uhi_intensity(self, feature_data):
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")

        missing_features = set(self.feature_names) - set(feature_data.columns)
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        X = feature_data[self.feature_names]
        predictions = self.model.predict(X)
        
        return predictions
    
    def create_synthetic_raster_output(self, predictions, coordinates, output_path='uhi_map.tif'):

        lat_min, lat_max = coordinates['lat'].min(), coordinates['lat'].max()
        lon_min, lon_max = coordinates['lon'].min(), coordinates['lon'].max()

        grid_size = 100
        lat_grid = np.linspace(lat_min, lat_max, grid_size)
        lon_grid = np.linspace(lon_min, lon_max, grid_size)

        from scipy.interpolate import griddata
        
        points = np.column_stack([coordinates['lon'], coordinates['lat']])
        grid_lon, grid_lat = np.meshgrid(lon_grid, lat_grid)
        grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])

        uhi_grid = griddata(points, predictions, grid_points, method='linear', fill_value=0)
        uhi_grid = uhi_grid.reshape(grid_size, grid_size)

        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, grid_size, grid_size)

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
        highway_buffer_m = highway_buffer_km * 1000
        highway_data = df[df['distance_to_highway'] <= highway_buffer_m].copy()
        
        if len(highway_data) == 0:
            print("No data points found near highways with current buffer.")
            return None

        highway_predictions = self.predict_uhi_intensity(highway_data)
        highway_coordinates = highway_data[['lat', 'lon']]

        highway_raster, transform = self.create_synthetic_raster_output(
            highway_predictions, 
            highway_coordinates, 
            output_path
        )
        print(f"Highway heat map saved to {output_path}")
        print(f"Highway buffer: {highway_buffer_km} km")
        print(f"Data points near highways: {len(highway_data)}")
        
        return highway_raster, transform
    
    def create_results_table(self, df, predictions):
        print("\n" + "="*80)
        print(" DETAILED RESULTS SUMMARY TABLE")
        print("="*80)

        print("\n UHI INTENSITY STATISTICS:")
        print("-" * 50)
        stats_data = {
            'Metric': ['Count', 'Mean', 'Std Dev', 'Min', '25th %ile', 'Median', '75th %ile', '90th %ile', 'Max'],
            'Value': [
                f"{len(predictions):,}",
                f"{np.mean(predictions):.3f}°C",
                f"{np.std(predictions):.3f}°C", 
                f"{np.min(predictions):.3f}°C",
                f"{np.percentile(predictions, 25):.3f}°C",
                f"{np.median(predictions):.3f}°C",
                f"{np.percentile(predictions, 75):.3f}°C",
                f"{np.percentile(predictions, 90):.3f}°C",
                f"{np.max(predictions):.3f}°C"
            ]
        }
        
        stats_df = pd.DataFrame(stats_data)
        print(stats_df.to_string(index=False, justify='left'))

        if self.feature_importance:
            print(f"\n🏆 FEATURE IMPORTANCE RANKING:")
            print("-" * 50)

            importance_data = []
            sorted_importance = sorted(self.feature_importance.items(), 
                                     key=lambda x: abs(x[1]), reverse=True)
            
            for rank, (feature, score) in enumerate(sorted_importance, 1):
                importance_data.append({
                    'Rank': rank,
                    'Feature': feature.replace('_', ' ').title(),
                    'Score': f"{score:.4f}",
                    'Impact': 'Positive' if score > 0 else 'Negative',
                    'Magnitude': 'High' if abs(score) > 0.2 else 'Medium' if abs(score) > 0.05 else 'Low'
                })
            importance_df = pd.DataFrame(importance_data)
            print(importance_df.to_string(index=False, justify='left'))
        if self.training_metrics:
            print(f"\n MODEL PERFORMANCE METRICS:")
            print("-" * 50)
            
            metrics_data = {
                'Metric': ['Training R²', 'Test R²', 'Training RMSE', 'Test RMSE', 'CV Mean R²', 'CV Std R²'],
                'Value': [
                    f"{self.training_metrics.get('train_r2', 0):.4f}",
                    f"{self.training_metrics.get('test_r2', 0):.4f}",
                    f"{self.training_metrics.get('train_rmse', 0):.4f}°C",
                    f"{self.training_metrics.get('test_rmse', 0):.4f}°C",
                    f"{np.mean(self.training_metrics.get('cv_scores', [0])):.4f}",
                    f"{np.std(self.training_metrics.get('cv_scores', [0])):.4f}"
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            print(metrics_df.to_string(index=False, justify='left'))
        print(f"\n UHI INTENSITY CATEGORIES:")
        print("-" * 50)

        categories = []
        low_count = np.sum(predictions < 2)
        medium_count = np.sum((predictions >= 2) & (predictions < 4))
        high_count = np.sum((predictions >= 4) & (predictions < 6))
        extreme_count = np.sum(predictions >= 6)
        
        category_data = {
            'Category': ['Low (< 2°C)', 'Medium (2-4°C)', 'High (4-6°C)', 'Extreme (≥ 6°C)'],
            'Count': [low_count, medium_count, high_count, extreme_count],
            'Percentage': [
                f"{(low_count/len(predictions)*100):.1f}%",
                f"{(medium_count/len(predictions)*100):.1f}%", 
                f"{(high_count/len(predictions)*100):.1f}%",
                f"{(extreme_count/len(predictions)*100):.1f}%"
            ]
        }
        
        category_df = pd.DataFrame(category_data)
        print(category_df.to_string(index=False, justify='left'))

        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            print(f"\n🔗 FEATURE CORRELATION WITH UHI INTENSITY:")
            print("-" * 60)

            numeric_df = df.select_dtypes(include=[np.number]).copy()
            numeric_df['uhi_intensity'] = predictions
            correlations = numeric_df.corr()['uhi_intensity'].drop('uhi_intensity')

            correlations_sorted = correlations.abs().sort_values(ascending=False)
            corr_data = []
            for feature in correlations_sorted.index:
                corr_val = correlations[feature]
                corr_data.append({
                    'Feature': feature.replace('_', ' ').title(),
                    'Correlation': f"{corr_val:.4f}",
                    'Strength': 'Strong' if abs(corr_val) > 0.6 else 'Medium' if abs(corr_val) > 0.3 else 'Weak',
                    'Direction': 'Positive' if corr_val > 0 else 'Negative'
                })
            
            corr_df = pd.DataFrame(corr_data)
            print(corr_df.to_string(index=False, justify='left'))
        
        print("\n" + "="*80)
def main():
    print("=== Urban Heat Island Model Demo ===\n")

    uhi_model = UrbanHeatIslandModel()

    print("1. Generating synthetic training data...")
    training_data = uhi_model.generate_synthetic_data(n_samples=10000)
    print(f"Generated {len(training_data)} training samples")
    print(f"Features: {list(training_data.columns)}")

    print("\n2. Training the UHI prediction model...")
    X_test, y_test, y_pred_test = uhi_model.train_model(training_data, model_type='random_forest')
    

    print("\n3. Generating factor importance ranking...")
    factor_output = uhi_model.generate_factor_importance_output('uhi_factor_importance.json')
   
    print("\nTop 5 Most Important Factors:")
    for i, factor in enumerate(factor_output['factor_importance'][:5]):
        print(f"{i+1}. {factor['factor']}: {factor['score']:.3f}")

    print("\n4. Creating predictions for full dataset...")
    try:
        all_predictions = uhi_model.predict_uhi_intensity(training_data)
        print(f" Generated {len(all_predictions)} predictions")
    except Exception as e:
        print(f" Prediction error: {e}")

        all_predictions = y_pred_test

    print("\n5. Creating GeoTIFF raster outputs...")
    coordinates = training_data[['lat', 'lon']]
    uhi_raster, transform = uhi_model.create_synthetic_raster_output(
        all_predictions, 
        coordinates, 
        'global_uhi_map.tif'
    )

    print("\n6. Creating highway heat island map...")
    highway_raster, highway_transform = uhi_model.create_highway_heat_map(
        training_data, 
        highway_buffer_km=0.5,
        output_path='highway_heat_map.tif'
    )

    print("\n7. View the results...")
    results=uhi_model.create_results_table(training_data, all_predictions)

    print(f"\n=== Model Summary ===")
    print(f"Model Type: {type(uhi_model.model).__name__}")
    print(f"Training Samples: {len(training_data)}")
    print(f"Features Used: {len(uhi_model.feature_names)}")
    print(f"Test R²: {uhi_model.training_metrics['test_r2']:.3f}")
    print(f"Mean UHI Intensity: {np.mean(all_predictions):.2f}°C")
    print(f"Max UHI Intensity: {np.max(all_predictions):.2f}°C")

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

    try:
        model, data, predictions = main()
        if model is not None:
            print("\n Workflow completed successfully!")
        else:
            print("\n  Workflow completed with some issues - check messages above")
    except Exception as e:
        print(f"\n Workflow failed: {e}")
        print("Try running the code in smaller sections to identify the issue")