import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Krok 1: Wczytaj zestaw danych Iris
iris = load_iris()
X = iris.data  # cechy
y = iris.target  # etykiety

# Krok 2: Podział danych na zestaw treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Krok 3: Inicjalizacja modelu Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Krok 4: Trening modelu
model.fit(X_train, y_train)

# Krok 5: Dokonanie predykcji
y_pred = model.predict(X_test)

# Krok 6: Ocena modelu
print("Macierz pomyłek:")
print(confusion_matrix(y_test, y_pred))

print("\nRaport klasyfikacji:")
print(classification_report(y_test, y_pred))

# Krok 7: Wizualizacja macierzy pomyłek
plt.figure(figsize=(10,7))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Macierz Pomyłek')
plt.xlabel('Przewidywana Klasa')
plt.ylabel('Rzeczywista Klasa')
plt.show()
