import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def load_data(dataset, data_dir):    
    print('Dataset name is : ', dataset, '\n')
    
    dataframe = pd.read_csv(data_dir+f'{dataset}.csv')

    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe.set_index('date', inplace=True)

    data_array = dataframe.values
    col_values = dataframe.std().values

    def optimized_custom_minmax_scale(data, col_values):
        min_val = np.min(data, axis=0) - col_values
        max_val = np.max(data, axis=0) + col_values
        return (data - min_val) / (max_val - min_val)

    scaled_array = optimized_custom_minmax_scale(data_array, col_values)
    return scaled_array


def generate_missing_data(data: np.ndarray, missing_rate: float, consecutive_missing_rate: float, max_missing_rate_per_feature: float, noise_rate: float) -> np.ndarray:
    data_with_missing = data.copy()
    total_values = data.shape[0] * data.shape[1]
    num_missing = int(total_values * missing_rate)
    base_consecutive_length = max(1, int(data.shape[0] * consecutive_missing_rate))
    
    remaining_missing = num_missing
    print('total_values:', total_values)
    print('num_missing:', num_missing)
    print('base_consecutive_length:', base_consecutive_length)
    
    column_missing_counts = np.zeros(data.shape[1], dtype=int)
    max_missing_per_column = int(data.shape[0] * max_missing_rate_per_feature)

    while remaining_missing > 0:
        available_columns = np.where(column_missing_counts < max_missing_per_column)[0]
        
        if len(available_columns) == 0:
            print("No available columns. Exiting.")
            break
        
        col = np.random.choice(available_columns)
        allowed_missing = min(remaining_missing, max_missing_per_column - column_missing_counts[col])
        
        if allowed_missing <= 0:
            continue
        
        noise = int(np.random.normal(0, base_consecutive_length * noise_rate))
        consecutive_missing_length = max(1, min(base_consecutive_length + noise, allowed_missing))
        
        cut_off = 0.9
        p = 0.5
        column_data = data_with_missing[:, col]
        is_null = np.isnan(column_data)

        while p < cut_off and allowed_missing > 0:
            if allowed_missing >= consecutive_missing_length:
                max_start_index = data.shape[0] - consecutive_missing_length
                start_index = np.random.randint(0, max_start_index + 1)
                end_index = start_index + consecutive_missing_length
                
                if not np.any(is_null[start_index:end_index]):
                    column_data[start_index:end_index] = np.nan
                    is_null[start_index:end_index] = True
                    missing_added = consecutive_missing_length
                    remaining_missing -= missing_added
                    allowed_missing -= missing_added
                    column_missing_counts[col] += missing_added
            
            cut_off -= 0.05
            p = np.random.random()
        
        data_with_missing[:, col] = column_data

    return data_with_missing


def evaluate_imputation(raw_data: np.ndarray, filled_data: np.ndarray, data_with_missing: np.ndarray, nan_indices: list) -> (dict, dict):
    per_column_metrics = {}
    all_true = []
    all_pred = []

    for missing_col in nan_indices:
        print(f"Evaluating imputation for column {missing_col}:")

        nan_rows = np.isnan(data_with_missing[:, missing_col])

        true_values = raw_data[nan_rows, missing_col]

        predicted_values = filled_data[nan_rows, missing_col]

        valid_imputations = ~np.isnan(predicted_values)

        if not np.any(valid_imputations):
            print(f"  No valid imputations for column {missing_col}. Skipping evaluation.\n")
            continue

        true_values_valid = true_values[valid_imputations]
        predicted_values_valid = predicted_values[valid_imputations]

        rmse = np.sqrt(mean_squared_error(true_values_valid, predicted_values_valid))
        mae = mean_absolute_error(true_values_valid, predicted_values_valid)
        
        if len(true_values_valid) >= 2:
            r2 = r2_score(true_values_valid, predicted_values_valid)
        else:
            r2 = float('nan')
            print(f"  Not enough samples to compute R² Score for column {missing_col}.\n")

        per_column_metrics[missing_col] = {
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2
        }

        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        if not np.isnan(r2):
            print(f"  R² Score: {r2:.6f}\n")
        else:
            print(f"  R² Score: nan\n")

        # Aggregated metrics
        all_true.extend(true_values_valid)
        all_pred.extend(predicted_values_valid)

    # Compute aggregated metrics
    if all_true and all_pred:
        aggregated_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
        aggregated_mae = mean_absolute_error(all_true, all_pred)
        
        if len(all_true) >= 2:
            aggregated_r2 = r2_score(all_true, all_pred)
        else:
            aggregated_r2 = float('nan')
            print("  Not enough samples to compute Aggregated R² Score.\n")

        aggregated_metrics = {
            'RMSE': aggregated_rmse,
            'MAE': aggregated_mae,
            'R2_Score': aggregated_r2
        }

        print("Aggregated Metrics:")
        print(f"  MSE: {aggregated_rmse**2:.6f}")
        print(f"  MAE: {aggregated_mae:.6f}")
        if not np.isnan(aggregated_r2):
            print(f"  R² Score: {aggregated_r2:.6f}\n")
        else:
            print("  R² Score: nan\n")
    else:
        aggregated_metrics = {}
        print("No valid imputations found across all columns. No aggregated metrics.\n")

    return per_column_metrics, aggregated_metrics


def save_evaluation_results(
    dataset: str,
    model_name: str,
    per_column_metrics: dict,
    aggregated_metrics: dict,
    # metric: str,
    # weights: str,
    num_similar_features: int,
    n_neighbors: int,
    missing_rate: float,
    consecutive_rate: float
):
    results_dir = 'RESULTS'
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{model_name}_{dataset}_{num_similar_features}_{n_neighbors}_{missing_rate:.4f}_{consecutive_rate:.4f}.txt"
    file_path = os.path.join(results_dir, filename)
    
    with open(file_path, 'w') as f:
        f.write("Per-Column Metrics:\n")
        for col, metrics in per_column_metrics.items():
            f.write(f"Column {col}:\n")
            f.write(f"  RMSE: {metrics['RMSE']:.6f}\n")
            f.write(f"  MAE: {metrics['MAE']:.6f}\n")
            f.write(f"  R² Score: {metrics['R2_Score']:.6f}\n\n")
        
        f.write("Aggregated Metrics:\n")
        if aggregated_metrics:
            f.write(f"  RMSE: {aggregated_metrics['RMSE']:.6f}\n")
            f.write(f"  MAE: {aggregated_metrics['MAE']:.6f}\n")
            f.write(f"  R² Score: {aggregated_metrics['R2_Score']:.6f}\n")
        else:
            f.write("  No aggregated metrics available.\n")
    
    print(f"Evaluation results saved to {file_path}\n")


def find_nan_regions(data: np.ndarray) -> List[Tuple[int, int]]:
    isnan = np.isnan(data)
    regions = []
    start = None
    for i, val in enumerate(isnan):
        if val and start is None:
            start = i
        elif not val and start is not None:
            regions.append((start, i - 1))
            start = None
    if start is not None:
        regions.append((start, len(isnan) - 1))
    return regions


def plot_missing_with_similars(
    dataset: str,
    raw_data: np.ndarray,
    data_with_missing: np.ndarray,
    data_interpolated: np.ndarray,
    closest_features_dict: dict,
    nan_indices: np.ndarray,
    window_size: int = 500,
    max_plots: int = 30,
    output_folder: str = 'RESULTS/Visualizations'
):
    
    os.makedirs(output_folder, exist_ok=True)
    
    plot_count = 0
    for missing_col in nan_indices:
        missing_data_col = data_with_missing[:, missing_col]
        nan_regions = find_nan_regions(missing_data_col)
        
        if not nan_regions:
            continue  
        
        similar_cols_list = closest_features_dict.get(missing_col, [])
        similar_cols = [col for col, corr in similar_cols_list]
        
        for region in nan_regions:
            if plot_count >= max_plots:
                break
            
            start, end = region
            plot_start = max(start - window_size, 0)
            plot_end = min(end + window_size, data_with_missing.shape[0] - 1)
            
            time = np.arange(plot_start, plot_end + 1)
            plt.figure(figsize=(20, 8))
            
            plt.plot(
                time, 
                raw_data[plot_start:plot_end + 1, missing_col], 
                label=f'Raw Data (Column {missing_col})', 
                color='black',
                alpha=0.7
            )
            
            imputed_time = np.arange(start, end + 1)
            plt.plot(
                imputed_time,
                data_interpolated[start:end + 1, missing_col], 
                label=f'Imputed Data (Column {missing_col})',
                color='blue',
                linestyle='dashed'
            )
            
            for sim_col, corr in similar_cols_list:
                if sim_col != missing_col:
                    plt.plot(
                        time, 
                        data_interpolated[plot_start:plot_end + 1, sim_col], 
                        label=f'Similar Column {sim_col} (corr={corr:.2f})', 
                        alpha=0.2
                    )
            
            plt.axvspan(start, end, color='pink', alpha=0.3)
            
            plt.title(f'Imputation Visualization for Column {missing_col} (Rows {plot_start}-{plot_end})')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            plot_filename = f"Imputation_Visualization_Col{missing_col}_Rows{start}-{end}.png"
            plot_path = os.path.join(output_folder, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            
            plot_count += 1
    
    print(f"Saved {plot_count} visualization plots to '{output_folder}' folder.\n")