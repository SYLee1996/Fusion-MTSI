import os
import random
import argparse
import numpy as np

from Fusion_MTSI_UTILS import load_data, generate_missing_data, evaluate_imputation, plot_missing_with_similars, save_evaluation_results
from Fusion_MTSI_MODEL import linear_interpolation, spearman_corr_numba, get_top_similar_features, compute_fusion_mtsi_distance, KNNRegressor


def get_args_parser():
    parser = argparse.ArgumentParser(description="Process log data for anomaly detection.", add_help=False)
    
    parser.add_argument("--dataset", type=str, required=True, choices=['electricity', 
                                                                       'national_illness', 
                                                                       'traffic', 
                                                                       'weather',
                                                                       'ETTm1', 'ETTm2','ETTh1','ETTh2',
                                                                       'exchange_rate'])
    parser.add_argument("--missing_rate", type=float, default=0.1)
    parser.add_argument("--consecutive_missing_rate", type=float, default=0.15)
    parser.add_argument("--max_missing_rate_per_feature", type=float, default=0.5)
    parser.add_argument("--noise_rate", type=float, default=0.1)
    
    parser.add_argument("--model_name", type=str, default='Fusion_MTSI')
    parser.add_argument("--num_similar_features", type=int, default=10)
    parser.add_argument("--n_neighbors", type=int, default=100)
    parser.add_argument("--weights", type=str, default='distance')
    parser.add_argument("--metric", type=str, default='fusion_mtsi')
    
    parser.add_argument("--seed", type=int, default=10)
    
    # Visualization flag
    parser.add_argument("--visualize", action='store_true',
                        help="If set, visualization images will be saved.")
    return parser


def knn_impute_inplace(
    missing_data: np.ndarray,
    refer_data: np.ndarray,
    missing_col: int,
    similar_cols: list,
    n_neighbors: int,
    weights: str,
    metric: str,
) -> None:

    nan_rows = np.isnan(missing_data[:, missing_col])
    if not np.any(nan_rows):
        print(f"No missing values found in column {missing_col}. Skipping imputation.")
        return

    valid_rows = ~nan_rows

    similar_cols_indices = [col for col, corr in similar_cols]
    corrs = [corr for col, corr in similar_cols]

    X_train = refer_data[valid_rows][:, similar_cols_indices]   # shape : (timestep, features)
    y_train = refer_data[valid_rows, missing_col]               # shape : (timestep, )
    X_test = refer_data[nan_rows][:, similar_cols_indices]
    
    def to_calculate_feature_weight(missing_data):
        X_train = missing_data[valid_rows][:, similar_cols_indices]   # shape : (timestep, features)
        y_train = missing_data[valid_rows, missing_col]               # shape : (timestep, )

        for idx, corr in enumerate(corrs):
            if corr <= 0:
                X_train[:, idx] = 1 - X_train[:, idx]
        return X_train, y_train
    
    X_train_to_feature, y_train_to_feature = to_calculate_feature_weight(missing_data)
    
    valid_train = ~np.isnan(X_train).any(axis=1)
    
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()
    
    for idx, corr in enumerate(corrs):
        if corr <= 0:
            X_train_clean[:, idx] = 1 - X_train_clean[:, idx]
            X_test_clean[:, idx] = 1 - X_test_clean[:, idx]
    
    y_train_clean = y_train[valid_train]
    X_train_clean = X_train_clean[valid_train]

    valid_pred = ~np.isnan(X_test_clean).any(axis=1)
    X_test_valid = X_test_clean[valid_pred]

    if X_train_clean.size == 0:
        print(f"No valid training data for column {missing_col}. Skipping imputation.")
        return
    if X_test_valid.size == 0:
        print(f"No valid test data for column {missing_col}. Skipping imputation.")
        return
    
    # Fusion-MTSI distance calculation
    if metric == 'euclidean':
        fusion_mtsi_distances = 1
    elif metric == 'fusion_mtsi':
        # fusion_mtsi_distances = compute_fusion_mtsi_distance(X_train_to_feature, y_train_to_feature, c=1.0)
        mtsi_distances = compute_fusion_mtsi_distance(X_train_to_feature, y_train_to_feature, c=1.0)
        spearman_exponent = 0.01
        fusion_mtsi_distances = mtsi_distances /((np.abs(np.array(corrs))+ 1e-8) ** spearman_exponent)
        
    knn_regression = KNNRegressor(
        n_neighbors=n_neighbors, 
        weights=weights, 
        metric=metric, # euclidean, fusion-mtsi
        series_weights=fusion_mtsi_distances
    )
    
    knn_regression.fit(X_train_clean, y_train_clean)

    y_pred = knn_regression.predict(X_test_valid)

    nan_indices_all = np.where(nan_rows)[0]
    valid_pred_indices = nan_indices_all[valid_pred]
    refer_data[valid_pred_indices, missing_col] = y_pred

def perform_knn_imputation_inplace(
    missing_data: np.ndarray,
    refer_data: np.ndarray,
    closest_features: dict,
    n_neighbors: int,
    weights: str,
    metric: str,
) -> np.ndarray:
    
    for idx, (missing_col, similar_cols) in enumerate(closest_features.items()):
        print(f"{idx+1}th Imputing column {missing_col} using similar columns {[col for col, corr in similar_cols]}")
        
        knn_impute_inplace(
            missing_data=missing_data,
            refer_data=refer_data,
            missing_col=missing_col,
            similar_cols=similar_cols,
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
    return refer_data

def main(args):
    
    # --------------------------------------------
    # Step 1: Loading and Preparing Data
    # --------------------------------------------
    
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)
    
    data_dir = os.path.expanduser(f'./dataset/')
    scaled_array = load_data(args.dataset, data_dir)

    
    data_with_missing = generate_missing_data(scaled_array, 
                                              missing_rate=args.missing_rate, 
                                              consecutive_missing_rate=args.consecutive_missing_rate, 
                                              max_missing_rate_per_feature=args.max_missing_rate_per_feature, 
                                              noise_rate=args.noise_rate)

    column_nan_counts = np.isnan(data_with_missing).sum(axis=0)

    missing_columns = np.sum(column_nan_counts > 0)
    total_columns = data_with_missing.shape[1]
    missing_columns_ratio = missing_columns / total_columns

    print('Data shape: ', scaled_array.shape)
    print(f"\nTotal columns: {total_columns}")
    print(f"Columns with missing values: {missing_columns}")
    print(column_nan_counts)
    print(f"Missing columns ratio: {missing_columns_ratio:.2%}")

    
    # --------------------------------------------
    # Step 2: Identifying NaN Features and Calculating Spearman Correlation
    # --------------------------------------------
    nan_indices = np.isnan(data_with_missing).any(axis=0).nonzero()[0]
    print('NaN Columns: ', nan_indices, '\n\n')
    print('---' * 10, 'IMPUTATION', '---' * 10) 

    data_interpolated = np.apply_along_axis(linear_interpolation, axis=0, arr=data_with_missing.copy())

    correlation_matrix = spearman_corr_numba(data_interpolated, nan_indices)

    # --------------------------------------------
    # Step 3: Selecting Top-N Similar Features
    # --------------------------------------------
    unsorted_similar_features = get_top_similar_features(nan_indices, correlation_matrix, args.num_similar_features)

    sorted_keys = sorted(unsorted_similar_features.keys(), key=lambda k: -max(abs(corr) for _, corr in unsorted_similar_features[k]))
    similar_features = {k: unsorted_similar_features[k] for k in sorted_keys}

    # --------------------------------------------
    # Step 4: Imputation
    # --------------------------------------------

    filled_data = perform_knn_imputation_inplace(
        missing_data=data_with_missing,
        refer_data=data_interpolated,
        closest_features=similar_features,
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        metric=args.metric
    )
    # np.save(f'{args.dataset}_raw_data', scaled_array)
    # np.save(f'{args.dataset}_{args.model_name}', filled_data)
    print("Imputation completed. \n\n\n")
        
    # --------------------------------------------
    # Step 5: Eval metrics and save results
    # --------------------------------------------

    # Evaluate imputation
    per_column_metrics, aggregated_metrics = evaluate_imputation(
        raw_data=scaled_array,
        filled_data=filled_data,
        data_with_missing=data_with_missing,
        nan_indices=nan_indices
    )
    
    # Save results
    save_evaluation_results(
        dataset=args.dataset,
        model_name=args.model_name,
        per_column_metrics=per_column_metrics,
        aggregated_metrics=aggregated_metrics,
        num_similar_features=args.num_similar_features,
        n_neighbors=args.n_neighbors,
        missing_rate=args.missing_rate,
        consecutive_rate=args.consecutive_missing_rate
    )

    # --------------------------------------------
    # Step 6: Visualization (optional)
    # --------------------------------------------
    
    if args.visualize:
        visualization_folder = os.path.join('RESULTS', f"{args.model_name}_{args.dataset}_{args.num_similar_features}_{args.n_neighbors}_{args.missing_rate:.4f}_{args.consecutive_missing_rate:.4f}_Visualizations")
        plot_missing_with_similars(
            dataset=args.dataset,
            raw_data=scaled_array,
            data_with_missing=data_with_missing,
            data_interpolated=data_interpolated,
            closest_features_dict=similar_features,
            nan_indices=nan_indices,
            window_size=500,
            output_folder=visualization_folder
        )
    else:
        print("Visualization is disabled. Skipping plot generation.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Fusion-MTSI', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
