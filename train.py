from sklearn.datasets import load_iris
from sklearn.svm import NuSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = load_iris()
X = df.data
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = NuSVC()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
result = accuracy_score(y_test, y_pred)

with open('results.txt', 'w') as f:
    f.write("Model Accuracy: " + result)