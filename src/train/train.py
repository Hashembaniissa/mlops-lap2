import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_model():
    data = load_iris()
    X, y = data.data, data.target

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    joblib.dump(model, "model.joblib")
    print("Model trained and saved successfully as model.joblib")

if __name__ == "__main__":
    train_model()
