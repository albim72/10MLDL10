import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# Załaduj zbiór danych
data = pd.read_csv('car_data.csv')

# Zajrzyj do pierwszych wierszy zbioru danych
print(data.head())

# Sprawdź brakujące dane
print(data.isnull().sum())

# Usuń wiersze z brakującymi danymi
data.dropna(inplace=True)

# Wybór cech i zmiennej celu (np. cena)
X = data[['year', 'mileage', 'engine_power', 'brand', 'fuel', 'body_type']]  # przykładowe cechy
y = data['price']  # zmienna celu (cena)

# One-Hot Encoding dla zmiennych kategorycznych (brand, fuel, body_type)
X = pd.get_dummies(X, columns=['brand', 'fuel', 'body_type'], drop_first=True)

# Podział danych na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Skalowanie cech
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Tworzenie modelu regresji liniowej
linear_model = Sequential([
    Dense(1, input_dim=X_train.shape[1])  # Jeden neuron (dla regresji)
])

# Kompilacja modelu
linear_model.compile(optimizer='adam', loss='mean_squared_error')

# Trening modelu regresji
linear_model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predykcja na zbiorze testowym
y_pred_linear = linear_model.predict(X_test)

# Ewaluacja modelu regresji liniowej
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f'MSE Regresji Liniowej: {mse_linear}')
print(f'R² Regresji Liniowej: {r2_linear}')

# Tworzenie modelu sieci neuronowej
neural_network = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Warstwa ukryta 1
    Dense(32, activation='relu'),  # Warstwa ukryta 2
    Dense(16, activation='relu'),  # Warstwa ukryta 3
    Dense(1)  # Warstwa wyjściowa (dla regresji)
])

# Kompilacja modelu
neural_network.compile(optimizer='adam', loss='mean_squared_error')

# Trening modelu sieci neuronowej
neural_network.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# Predykcja na zbiorze testowym
y_pred_nn = neural_network.predict(X_test)

# Ewaluacja modelu sieci neuronowej
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)
print(f'MSE Sieci Neuronowej: {mse_nn}')
print(f'R² Sieci Neuronowej: {r2_nn}')

print("Porównanie wyników:")
print(f"Regresja Liniowa - MSE: {mse_linear}, R²: {r2_linear}")
print(f"Sieć Neuronowa - MSE: {mse_nn}, R²: {r2_nn}")
