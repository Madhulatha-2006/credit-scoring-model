import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Simulate a dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n_samples),
    'debt': np.random.normal(15000, 5000, n_samples),
    'late_payments': np.random.poisson(2, n_samples),
    'credit_history_years': np.random.randint(1, 20, n_samples),
    'open_credit_lines': np.random.randint(1, 10, n_samples),
    'loan_default': np.random.binomial(1, 0.3, n_samples),
    'employment_status': np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], n_samples)
})

# 2. Target variable: Creditworthy if no defaults & few late payments
data['creditworthy'] = ((data['loan_default'] == 0) & (data['late_payments'] < 3)).astype(int)

# 3. Feature Engineering
data['debt_to_income_ratio'] = data['debt'] / (data['income'] + 1)

# One-hot encoding
data = pd.get_dummies(data, columns=['employment_status'], drop_first=True)

# 4. Define features and target
X = data.drop(columns=['creditworthy', 'loan_default'])  # drop redundant info
y = data['creditworthy']

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Build Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# 7. Training & Evaluation
for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # optional: skip for tree models
        ('classifier', model)
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print(f"\n📊 Model: {name}")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
