import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

data_path = '/content/Data_Problem1/'
X_train_path = data_path + 'X_train.txt'
X_test_path = data_path + 'X_test.txt'
y_train_path = data_path + 'y_train.txt'
y_test_path = data_path + 'y_test.txt'

X_train = np.loadtxt(X_train_path)
X_test = np.loadtxt(X_test_path)

print("X_train shape:", X_train.shape)
print("X_test shape before transpose:", X_test.shape)

X_test = X_test.T  

print("X_test shape after transpose:", X_test.shape)

# Load the labels
y_train = np.loadtxt(y_train_path)
y_test = np.loadtxt(y_test_path)

# Preprocess data (scaling)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_pred_poly = np.zeros_like(y_test)
y_pred_rbf = np.zeros_like(y_test)

# Train and predict using SVM for each class
for i in range(y_train.shape[1]): 
    # Polynomial Kernel
    poly_svm = SVC(kernel='poly', degree=2, gamma='auto')
    poly_svm.fit(X_train_scaled, y_train[:, i])
    y_pred_poly[:, i] = poly_svm.predict(X_test_scaled)
    
    # Gaussian Kernel (RBF)
    rbf_svm = SVC(kernel='rbf', gamma=2)
    rbf_svm.fit(X_train_scaled, y_train[:, i])
    y_pred_rbf[:, i] = rbf_svm.predict(X_test_scaled)

# Compute accuracy for each test sample
def compute_accuracy(y_true, y_pred):
    accuracies = []
    for true, pred in zip(y_true, y_pred):
        intersection = np.sum(np.logical_and(true, pred))
        union = np.sum(np.logical_or(true, pred))
        if union > 0:
            accuracies.append(intersection / union)
    return np.mean(accuracies) * 100

# Calculate accuracy for both models
accuracy_poly = compute_accuracy(y_test, y_pred_poly)
accuracy_rbf = compute_accuracy(y_test, y_pred_rbf)

print(f"Accuracy with Polynomial Kernel (degree=2): {accuracy_poly:.2f}%")
print(f"Accuracy with Gaussian Kernel (γ=2): {accuracy_rbf:.2f}%")
