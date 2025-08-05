import numpy as np

# 1. Read the dataset into a numpy array
file_path = '../../data/Asgmnt1_data.txt'

data = np.loadtxt(file_path)

# Z-normalize the data (zero mean, unit variance)
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data_znorm = (data - data_mean) / data_std

print('Data shape:', data.shape)
print('Z-normalized data mean (should be ~0):', np.mean(data_znorm, axis=0)[:5])
print('Z-normalized data std (should be ~1):', np.std(data_znorm, axis=0)[:5])

# 2. Compute the Euclidean distance matrix manually
def euclidean_distance_matrix(X):
    n_samples = X.shape[0]
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.sqrt(np.sum((X[i] - X[j]) ** 2))
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist  # symmetric
    return dist_matrix

# For demonstration, compute on a small subset due to memory constraints
subset = data_znorm[:100]  # Use first 100 samples for distance matrix
distance_matrix = euclidean_distance_matrix(subset)
print('Distance matrix shape:', distance_matrix.shape)

# 3. Haar matrix generator
def haar_matrix(n):
    if n == 1:
        return np.array([[1]])
    H = haar_matrix(n // 2)
    top = np.kron(H, [1, 1])
    bottom = np.kron(np.eye(len(H)), [1, -1])
    return np.vstack((top, bottom)) / np.sqrt(2)

# Generate a 128x128 Haar matrix
haar_128 = haar_matrix(128)
print('Haar matrix shape:', haar_128.shape)

# Transform data_znorm using the Haar matrix
data_wavelet = data_znorm @ haar_128.T
print('Wavelet-transformed data shape:', data_wavelet.shape)

# Use the first four coefficients for a new distance matrix
subset_wavelet = data_wavelet[:100, :4]  # First 100 samples, first 4 coefficients
distance_matrix_wavelet = euclidean_distance_matrix(subset_wavelet)
print('Wavelet-based distance matrix shape:', distance_matrix_wavelet.shape)

# 4. PCA implementation (step by step, no scikit-learn)
def pca(X, n_components):
    # Step 1: Center the data (already zero mean if z-normalized, but let's be explicit)
    X_centered = X - np.mean(X, axis=0)

    # Step 2: Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)

    # Step 3: Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # Step 4: Sort eigenvectors by eigenvalues in descending order
    sorted_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_idx]
    eigvals = eigvals[sorted_idx]

    # Step 5: Select top n_components eigenvectors
    principal_components = eigvecs[:, :n_components]

    # Step 6: Project data onto principal components
    X_pca = X_centered @ principal_components
    return X_pca, principal_components, eigvals[:n_components]

# Apply PCA to z-normalized data, keep first 4 components
data_pca, pca_components, pca_eigvals = pca(data_znorm, 4)
print('PCA-transformed data shape:', data_pca.shape)
    
# Use the first four PCA components for a new distance matrix
subset_pca = data_pca[:100]  # First 100 samples, already 4 components
distance_matrix_pca = euclidean_distance_matrix(subset_pca)
print('PCA-based distance matrix shape:', distance_matrix_pca.shape)

# 5. Benchmarking distance matrix calculations
import time

print('\n--- Benchmarking distance matrix calculations (using whole dataset) ---')

# Q2: Original z-normalized data
start = time.time()
distance_matrix = euclidean_distance_matrix(data_znorm)
elapsed_q2 = time.time() - start
print(f'Time for Q2 (z-normalized, shape {data_znorm.shape}): {elapsed_q2:.2f} seconds')

# Q3: Wavelet transform, first 4 coefficients
data_wavelet_full = data_znorm @ haar_128.T
start = time.time()
distance_matrix_wavelet = euclidean_distance_matrix(data_wavelet_full[:, :4])
elapsed_q3 = time.time() - start
print(f'Time for Q3 (wavelet, shape {data_wavelet_full[:, :4].shape}): {elapsed_q3:.2f} seconds')

# Q4: PCA, first 4 components
data_pca_full, _, _ = pca(data_znorm, 4)
start = time.time()
distance_matrix_pca = euclidean_distance_matrix(data_pca_full)
elapsed_q4 = time.time() - start
print(f'Time for Q4 (PCA, shape {data_pca_full.shape}): {elapsed_q4:.2f} seconds')

# 6. Compare pairwise relationships (greater/less) between matrices
import random

def compare_relationships(mat_gt, mat_other, num_pairs=10000, seed=42):
    np.random.seed(seed)
    n = mat_gt.shape[0]
    count_agree = 0
    for _ in range(num_pairs):
        i, j = np.random.randint(0, n, 2)
        k, l = np.random.randint(0, n, 2)
        if i == j or k == l:
            continue
        rel_gt = mat_gt[i, j] > mat_gt[k, l]
        rel_other = mat_other[i, j] > mat_other[k, l]
        if rel_gt == rel_other:
            count_agree += 1
    return count_agree / num_pairs * 100

# Use the first 100 samples for demonstration (as in previous steps)
gt = distance_matrix
wavelet = distance_matrix_wavelet
pca_matrix = distance_matrix_pca

percent_wavelet = compare_relationships(gt, wavelet)
percent_pca = compare_relationships(gt, pca_matrix)

print(f'Wavelet matrix relationship agreement: {percent_wavelet:.2f}%')
print(f'PCA matrix relationship agreement: {percent_pca:.2f}%')

