import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
try:
    data = pd.read_csv('diabetes.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: 'diabetes.csv' not found in the current directory.")
    exit()

# Select only the desired features
features = ['Glucose', 'BMI', 'Insulin', 'Age', 'DiabetesPedigreeFunction']
X = data[features]
y = data['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data and save scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved as 'scaler.pkl'")

# Train model and save
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
joblib.dump(model, 'rf_model.pkl')
print("Model saved as 'rf_model.pkl'")