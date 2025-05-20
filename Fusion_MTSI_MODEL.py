import numpy as np
from numba import njit, prange


###################################################### linear_interpolation ######################################################
def linear_interpolation(array):
    nans = np.isnan(array)
    array[nans] = np.interp(np.flatnonzero(nans), np.flatnonzero(~nans), array[~nans])
    return array

###################################################### spearman_correlation ######################################################
@njit(parallel=True)
def spearman_corr_numba(data, nan_indices):
    n, m = data.shape
    rank_data = np.empty_like(data)
    
    # Rank each column in parallel
    for i in prange(m):
        rank_data[:, i] = rank_data_column(data[:, i])
    
    # Calculate Spearman correlations for nan_indices only
    return pearson_corr_numba(rank_data, nan_indices)

@njit
def rank_data_column(col):
    # Rank data for each column
    sorted_idx = np.argsort(col)
    ranks = np.empty_like(sorted_idx)
    ranks[sorted_idx] = np.arange(len(col))
    return ranks

@njit#(parallel=True)
def pearson_corr_numba(rank_data, nan_indices):
    n, m = rank_data.shape
    corr_matrix = np.empty((len(nan_indices), m))  # Only store correlations for nan_indices rows
    
    for idx, i in enumerate(nan_indices):
        corr_matrix[idx, i] = 0.0  # Self-correlation is 1
        for j in range(m):
            if i != j:
                col_i = rank_data[:, i]
                col_j = rank_data[:, j]
                corr = pearson_correlation(col_i, col_j)
                corr_matrix[idx, j] = corr
    return corr_matrix

@njit
def pearson_correlation(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    num = np.sum((x - mean_x) * (y - mean_y))
    den = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    return num / den if den != 0 else 0.0

###################################################### get_top_similar_features ######################################################
def get_top_similar_features(nan_indices, correlation_matrix, num_similar_features):
    similar_features_dict = {}
    num_features = correlation_matrix.shape[1]

    for corr, nan_col in zip(correlation_matrix, nan_indices):
        corrs = corr.copy()
        
        # Set the correlation with itself to 0 to exclude it
        if corrs[nan_col] == 1.0:
            corrs[nan_col] = 0.0 

        # Limit the number of top_indices to num_similar_features
        effective_num = min(num_similar_features, num_features - 1)
        
        # Sort correlations by absolute value and get the top N indices.
        top_indices = np.argpartition(-np.abs(corrs), effective_num)[:effective_num]

        # Sort the selected indices again based on absolute correlation values (descending order)
        top_indices = top_indices[np.argsort(-np.abs(corrs[top_indices]))]
        
        # Get the original correlation coefficients of the selected features.
        top_corrs = corrs[top_indices]
        
        # Store the results as a list of tuples (feature index, correlation coefficient)
        similar_features = list(zip(top_indices, top_corrs))
        
        # Store in the dictionary (feature index as key)
        similar_features_dict[nan_col] = similar_features
        
    return similar_features_dict


###################################################### Fusion-MTSI distacne ######################################################
@njit
def dtw_distance(ts_a, ts_b):
    valid_mask = ~np.isnan(ts_a) & ~np.isnan(ts_b)
    ts_a_valid = ts_a[valid_mask]
    ts_b_valid = ts_b[valid_mask]

    n = len(ts_a_valid)
    m = len(ts_b_valid)

    if n == 0 or m == 0:
        return np.inf

    dtw = np.full((2, m + 1), np.inf)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        dtw[1, 0] = np.inf
        for j in range(1, m + 1):
            cost = abs(ts_a_valid[i - 1] - ts_b_valid[j - 1])
            dtw[1, j] = cost + min(
                dtw[0, j],
                dtw[1, j - 1],
                dtw[0, j - 1]
            )
        dtw[0, :] = dtw[1, :]
    return dtw[1, m]
@njit
def msm_cost(x, y, c):
    return c + abs(x - y)

@njit
def msm_distance(ts_a, ts_b, c):
    valid_mask = ~np.isnan(ts_a) & ~np.isnan(ts_b)
    ts_a_valid = ts_a[valid_mask]
    ts_b_valid = ts_b[valid_mask]

    n = len(ts_a_valid)
    m = len(ts_b_valid)

    if n == 0 or m == 0:
        return np.inf

    D = np.full((2, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        D[1, 0] = np.inf
        for j in range(1, m + 1):
            cost = abs(ts_a_valid[i - 1] - ts_b_valid[j - 1])

            if i == 1:
                prev_a = ts_a_valid[i - 1]
            else:
                prev_a = ts_a_valid[i - 2]

            if j == 1:
                prev_b = ts_b_valid[j - 1]
            else:
                prev_b = ts_b_valid[j - 2]

            D[1, j] = min(
                D[0, j - 1] + cost,
                D[0, j] + msm_cost(ts_a_valid[i - 1], prev_a, c),
                D[1, j - 1] + msm_cost(ts_b_valid[j - 1], prev_b, c)
            )
        D[0, :] = D[1, :]
    return D[1, m]

@njit(parallel=True)
def compute_fusion_mtsi_distance(X_train, y_train, c=1.0):
    n_features = X_train.shape[1]
    distances = np.zeros(n_features)
    for i in prange(n_features):
        ts_a = X_train[:, i]
        ts_b = y_train

        valid_mask = ~np.isnan(ts_a) & ~np.isnan(ts_b)
        ts_a_valid = ts_a[valid_mask]
        ts_b_valid = ts_b[valid_mask]

        if len(ts_a_valid) == 0:
            distances[i] = distances[i] = -1.0
            continue

        dist = np.log(msm_distance(ts_a_valid, ts_b_valid, c) + 1.0) + np.log(dtw_distance(ts_a_valid, ts_b_valid) + 1.0)
        distances[i] = dist
        
    valid_distances = distances[distances != -1.0]
    if valid_distances.size > 0:
        max_distance = np.max(valid_distances)
    else:
        max_distance = 1 
    
    distances[distances == -1.0] = max_distance + 1.0

    return distances


###################################################### Knn-Regression ######################################################
@njit(parallel=True)
def knn_predict(X_train, y_train, X_test, n_neighbors, series_weights):
    n_test = X_test.shape[0]
    predictions = np.empty(n_test, dtype=np.float64)
    
    for i in prange(n_test):
        
        # Calculate Euclidean distances
        square = (X_train - X_test[i])**2
        
        # Apply feature weights
        weighted_square = square * series_weights  # Apply Fusion-MTSI weights for each feature
        square_sum = np.sum(weighted_square, axis=1)

        # Instance weights
        instance_distances = np.sqrt(np.sum((X_train - X_test[i])**2, axis=1))
        instance_weights = instance_distances / np.sum(instance_distances)

        # Calculate weighted sum of distances
        weighted_square_sum = square_sum * instance_weights

        # Final distance calculation (if needed)
        distances = np.sqrt(weighted_square_sum)

        # Find indices of n_neighbors closest neighbors
        idx = np.argpartition(distances, n_neighbors)[:n_neighbors]
        nearest_distances = distances[idx]
        nearest_values = y_train[idx]
        
        # Calculate distance-based weights (handle zero distances)
        weights = 1.0 / (nearest_distances + 1e-8)  # Avoid division by zero
        weighted_sum = np.sum(weights * nearest_values)
        sum_weights = np.sum(weights)
        
        predictions[i] = weighted_sum / sum_weights if sum_weights > 0 else np.mean(nearest_values)
    return predictions

class KNNRegressor():
    def __init__(self, n_neighbors=5, weights='distance', metric='euclidean', c=1.0, series_weights=1):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.c = c
        self.series_weights = series_weights

    def fit(self, X, y):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).flatten()
        return self

    def predict(self, X):
        X = np.asarray(X)
        return knn_predict(self.X_train, self.y_train, X, self.n_neighbors, self.series_weights)