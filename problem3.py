import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

data = pd.read_csv("/content/Data_Problem3/German Credit Data.txt", delimiter=',', header=None)


X = data.iloc[:, :-1]  
y = data.iloc[:, -1]   

encoder = LabelEncoder()
X_encoded = X.apply(encoder.fit_transform)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

undersample = RandomUnderSampler(random_state=42)
X_train_undersample, y_train_undersample = undersample.fit_resample(X_train, y_train)

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

def train_and_evaluate(X_train_data, y_train_data, X_test_data, y_test_data, model):
    model.fit(X_train_data, y_train_data)
    y_pred = model.predict(X_test_data)
    f1 = f1_score(y_test_data, y_pred, average='weighted')  
    return f1, classification_report(y_test_data, y_pred)

rf_smote = RandomForestClassifier(random_state=42, class_weight='balanced')
f1_smote, report_smote = train_and_evaluate(X_train_smote, y_train_smote, X_test, y_test, rf_smote)

rf_undersample = RandomForestClassifier(random_state=42, class_weight='balanced')
f1_undersample, report_undersample = train_and_evaluate(X_train_undersample, y_train_undersample, X_test, y_test, rf_undersample)

rf_class_weights = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)
f1_class_weights, report_class_weights = train_and_evaluate(X_train, y_train, X_test, y_test, rf_class_weights)

logreg_smote = LogisticRegression(class_weight='balanced', random_state=42)
f1_logreg_smote, report_logreg_smote = train_and_evaluate(X_train_smote, y_train_smote, X_test, y_test, logreg_smote)

print("Random Forest with SMOTE (Oversampling) F1-score:", f1_smote)
print("Classification Report:\n", report_smote)

print("\nRandom Forest with Random Undersampling F1-score:", f1_undersample)
print("Classification Report:\n", report_undersample)

print("\nRandom Forest with Class Weights F1-score:", f1_class_weights)
print("Classification Report:\n", report_class_weights)

print("\nLogistic Regression with SMOTE F1-score:", f1_logreg_smote)
print("Classification Report:\n", report_logreg_smote)
