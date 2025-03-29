import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('customer_churn_data.csv')

# Handle missing values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_columns = ['Gender', 'Geography']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Feature scaling
scaler = StandardScaler()
numerical_columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Creating additional features
data['TenureCluster'] = pd.cut(data['Tenure'], bins=[0, 3, 6, 9, 12], labels=[1, 2, 3, 4])
data['BalanceSalaryRatio'] = data['Balance'] / data['EstimatedSalary']

# Handling class imbalance
X = data.drop('Exited', axis=1)
y = data['Exited']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
