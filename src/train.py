from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os

data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "src/models/model.joblib")
print("Model saved as model.joblib")
