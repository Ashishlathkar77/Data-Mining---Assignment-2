## Problem 1: Support Vector Machine (SVM) Classification with Polynomial and Gaussian Kernels

### Problem:
The task is to classify the data using SVM with Polynomial (degree=2) and Gaussian (RBF) kernels.

### Solution:
Data Preprocessing: The data was scaled using StandardScaler.

Modeling:
Polynomial Kernel: An SVM with a polynomial kernel of degree 2 was used.
Gaussian Kernel: An SVM with a Gaussian (RBF) kernel and γ=2 was applied.

Results:
Accuracy with Polynomial Kernel (degree=2): 40.98%
Accuracy with Gaussian Kernel (γ=2): 0.55%

### Command to run script:
```python
python problem1.py
```

## Problem 2: K-Means Clustering
### Problem:
Perform K-means clustering on a dataset and evaluate the clustering performance with different values of k (3, 5, 7).

### Solution:
Algorithm: The Euclidean distance metric was used for clustering, and K-means was applied with 10 random initialization.

Evaluation: The Sum of Squared Errors (SSE) was calculated for each value of k.

### Results:
Average SSE for k=3: 587.3186

Average SSE for k=5: 385.6879

Average SSE for k=7: 280.7045

### command to run script:
```python
python problem2.py
```

## Problem 3: Classification with Random Forest and Logistic Regression
### Problem:
Classify the dataset using Random Forest and Logistic Regression models, employing different class imbalance handling techniques:

SMOTE (Oversampling)
Random Undersampling
Class Weights

### Solution:
Data Preprocessing: The dataset was encoded, and different preprocessing strategies were applied:

SMOTE was used to oversample the minority class.

Random Undersampling was applied to the majority class.

Class weights were computed and used in the models.

Modeling: The models were evaluated using F1-score and classification reports.

### Results:
Random Forest with SMOTE (Oversampling) F1-score: 0.761

Random Forest with Random Undersampling F1-score: 0.713

Random Forest with Class Weights F1-score: 0.714

Logistic Regression with SMOTE F1-score: 0.763

### command to run script:

```python
python problem3.py
```

## Instructions for Running the Code
Set Up Environment: Ensure you have the necessary Python libraries installed, including numpy, scikit-learn, pandas, and imbalanced-learn.
Dataset Paths: Update the dataset paths in each problem script as per your local or cloud directory.

### Run Scripts:
For Problem 1, use the script problem1.py to execute the SVM classification with different kernels.

For Problem 2, use problem2.py to perform K-means clustering with various values of k.

For Problem 3, use problem3.py for the classification tasks with class imbalance handling techniques.

* To automatically install the dependencies and run all the problem scripts one by one, run the following command:

```python
python run_project_all.py
```

This will:

- Install the required dependencies from requirements.txt.
- Execute the scripts problem1.py, problem2.py, and problem3.py in sequence.

### Check the Output
Once the script completes, check the output of each problem. The results will be printed to the terminal/console where you ran the script.

### Conclusion
SVM Results: The Polynomial Kernel performed better than the Gaussian Kernel for this dataset.

K-means Clustering: The SSE decreased with increasing k, indicating better clustering performance as more clusters were used.

Classification with Random Forest and Logistic Regression: Handling class imbalance through oversampling (SMOTE) yielded the highest F1-score, followed by using class weights, with random undersampling performing the worst.
