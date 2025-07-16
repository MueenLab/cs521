# Assignment 1 Dimensionality Reduction Chat History

**STUDENT**: 

I am taking a graduate level data mining class, and now I am going to do the first assignment. The assignment is about dimensionality reduction. We are given a dataset, and we are going to implement dimensionality reduction algorithms, such as PCA and wavelet transform. I would like you to assist me.

**AI AGENT**: 

Great! I can help you implement dimensionality reduction algorithms like PCA and wavelet transform for your assignment. Here’s how we can proceed:

1. Load the dataset from Asgmnt1_data.txt.
2. Implement PCA (Principal Component Analysis).
3. Implement a wavelet transform (e.g., using PyWavelets).
4. Apply these algorithms to the dataset and visualize or analyze the results.

Would you like to start with PCA, wavelet transform, or both? Also, do you need help with data loading and preprocessing?

**STUDENT**: 

Let's solve the assignment step by step. First of all, i need to read the file Asgmnt1_data.txt in `data/` into a numpy array

**AI AGENT**:
```python
import numpy as np

# 1. Read the dataset into a numpy array
file_path = 'data/Asgmnt1_data.txt'

data = np.loadtxt(file_path)
```

The code to read `Asgmnt1_data.txt` into a numpy array is now in assignment_1.py. However, it looks like numpy is not installed in your environment.

Would you like me to install numpy for you?

**STUDENT**: 

please create a venv and install numpy

**AI AGENT**: 

The virtual environment has been created and numpy is installed. You can now run your script to read the dataset.

Would you like to proceed to the next step, such as inspecting the data or starting with PCA implementation?

**STUDENT**: 

please run the script first

**AI AGENT**: 

The script ran successfully. The dataset was loaded into a numpy array with shape (16000, 128).

Would you like to inspect the data further, or proceed to implementing PCA for dimensionality reduction?

**STUDENT**: 

The first question is to z-normalize the data

**AI AGENT**: 
```python
# Z-normalize the data (zero mean, unit variance)
data_mean = np.mean(data, axis=0)
data_std = np.std(data, axis=0)
data_znorm = (data - data_mean) / data_std

print('Data shape:', data.shape)
print('Z-normalized data mean (should be ~0):', np.mean(data_znorm, axis=0)[:5])
print('Z-normalized data std (should be ~1):', np.std(data_znorm, axis=0)[:5])
```
Z-normalization code has been added. The script now computes the mean and standard deviation for each feature, normalizes the data, and prints the first few values of the mean and standard deviation to verify the normalization.

Would you like to run the script to check the output, or proceed to the next step?

**STUDENT**: 

please run the script

**AI AGENT**: 

The script ran successfully. The z-normalization worked as expected: the mean is approximately zero and the standard deviation is approximately one for the first few features.

Would you like to proceed to the next step, such as implementing PCA?

**STUDENT**: 

The second task is to create a distance matrix of the data using Euclidean distance. We have to implement the algorithm by ourselves

**AI AGENT**: 
```python
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
```
A manual implementation for computing the Euclidean distance matrix has been added. For demonstration and efficiency, it computes the distance matrix for the first 100 samples of the z-normalized data.

Would you like to run the script to see the output, or adjust the subset size?

**STUDENT**: 

please run the script

**AI AGENT**: 

The script ran successfully and computed the Euclidean distance matrix for the first 100 samples. The output confirms the shape as (100, 100).

Would you like to proceed to the next task, or do you need further analysis or visualization of the distance matrix?

**STUDENT**: 

in question 3, we need to transform `data_znorm` using Wavelet Transform. First of all, write a Haar matrix generator, and use it to generate a 128x128 haar matrix

**AI AGENT**: 
```python
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
```
A Haar matrix generator function has been added, and a 128x128 Haar matrix is generated and its shape printed.

Would you like to run the script to verify the Haar matrix, or proceed to applying the wavelet transform to your data?

**STUDENT**: 

Then transform the np array `data_znorm` using the Haar matrix, and use the first four coefficients to calculate a new distance matrix 

**AI AGENT**: 
```python
# Transform data_znorm using the Haar matrix
data_wavelet = data_znorm @ haar_128.T
print('Wavelet-transformed data shape:', data_wavelet.shape)

# Use the first four coefficients for a new distance matrix
subset_wavelet = data_wavelet[:100, :4]  # First 100 samples, first 4 coefficients
distance_matrix_wavelet = euclidean_distance_matrix(subset_wavelet)
print('Wavelet-based distance matrix shape:', distance_matrix_wavelet.shape)
```
The code now transforms `data_znorm` using the Haar matrix, extracts the first four wavelet coefficients, and computes a new Euclidean distance matrix using those coefficients for the first 100 samples.

Would you like to run the script to see the output, or proceed to further analysis or visualization?

**STUDENT**: 

please run the script

**AI AGENT**: 

The script ran successfully. The wavelet transform and the new distance matrix using the first four coefficients were computed and their shapes printed.

If you need further analysis, visualization, or want to proceed to the next assignment step, let me know!

**STUDENT**: 

Question 4 requires us to transform the data `data_znorm` using PCA without calling scikit learn or other 3rd party libraries. Please write down the algorithm step by step

**AI AGENT**: 

Here is a step-by-step outline for performing PCA on `data_znorm` without using scikit-learn or other third-party libraries:

1. **Center the Data**  
   Subtract the mean of each feature (column) from the data to ensure zero mean.

2. **Compute the Covariance Matrix**  
   Calculate the covariance matrix of the centered data.

3. **Compute Eigenvalues and Eigenvectors**  
   Find the eigenvalues and eigenvectors of the covariance matrix.

4. **Sort Eigenvectors by Eigenvalues**  
   Sort the eigenvectors in descending order of their corresponding eigenvalues (variance explained).

5. **Select Principal Components**  
   Choose the top k eigenvectors (principal components) based on the largest eigenvalues.

6. **Project Data onto Principal Components**  
   Transform the original data by projecting it onto the selected principal components.

Would you like to see the code for these steps added to your script? If so, how many principal components do you want to keep?

**STUDENT**: 

Please write me the code

**AI AGENT**: 
```python
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
```
The code for step-by-step PCA (without scikit-learn or other libraries) has been added. It projects the z-normalized data onto the first 4 principal components and prints the shape of the transformed data.

Would you like to run the script to verify the PCA output, or proceed to further analysis or distance matrix calculation using the PCA results?

**STUDENT**: 

yes please run the script

**AI AGENT**: 

The script ran successfully. The PCA implementation worked, and the data was projected onto the first 4 principal components, resulting in a shape of (16000, 4).

Would you like to compute a distance matrix using the PCA-transformed data, or proceed to the next assignment step?

**STUDENT**: 

We need to compute a distance matrix using the PCA transformed data

**AI AGENT**: 
```python
# Use the first four PCA components for a new distance matrix
subset_pca = data_pca[:100]  # First 100 samples, already 4 components
distance_matrix_pca = euclidean_distance_matrix(subset_pca)
print('PCA-based distance matrix shape:', distance_matrix_pca.shape)
```
The code now computes a Euclidean distance matrix using the first 4 PCA components for the first 100 samples and prints its shape.

Would you like to run the script to verify the output, or proceed to the next step?

**STUDENT**: 

Please run the script again

**AI AGENT**: 

The script ran successfully. The PCA-based distance matrix was computed and its shape is (100, 100), as expected.

If you need further analysis, visualization, or want to proceed to the next assignment step, let me know!

**STUDENT**: 

Question 5: we need to compare the time required to calculate the distance matrix in question 2, 3 and 4. Can you write down the benchmarking code in a new section? We can use the whole dataset for the benchmarking

**AI AGENT**: 
```python
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

```
Benchmarking code has been added to compare the time required to calculate the distance matrices for Q2 (z-normalized), Q3 (wavelet, first 4 coefficients), and Q4 (PCA, first 4 components) using the whole dataset.

Would you like to run the script to see the timing results? Note: This may take significant time and memory due to the size of the dataset.

**STUDENT**: 

We will skip the execution for now

**AI AGENT**: 

Understood! The benchmarking code is ready, but execution will be skipped for now.

If you need further help with analysis, visualization, or the next steps in your assignment, just let me know!

**STUDENT**: 

Question 6: Consider the matrix from step-2 as the ground truth. For each pair of cells in the matrix, check if the relationships (i.e. greater or less) hold in the matrices from steps 3 and 4. Report the percentage times the relationships hold. If the computation is too much time consuming, consider using some random pairs of cells instead of all pairs of cells in the matrices.

**AI AGENT**: 
```python
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
```
The code to compare pairwise relationships between the ground truth, wavelet, and PCA distance matrices has been added. It reports the percentage of times the relationships agree, using random pairs for efficiency.