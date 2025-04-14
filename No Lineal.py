# -*- coding: utf-8 -*-
"""Regresión Múltiple Simplificada - House Prices.ipynb"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datos
train_data = pd.read_csv('train.csv')

# 2. Preprocesamiento rápido
# Eliminar columnas con muchos nulos y no relevantes
cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
train_data = train_data.drop(cols_to_drop, axis=1)

# Codificar variables categóricas e imputar nulos
train_data = pd.get_dummies(train_data, drop_first=True)
train_data = train_data.fillna(train_data.median())

# 3. Dividir datos
X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# 4. Entrenar modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Métricas (versión corregida)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5  # Calculamos RMSE manualmente
r2 = r2_score(y_test, y_pred)

print("\n--- Métricas del Modelo ---")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 6. Gráfico 1: Predicciones vs Valores Reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Línea de perfecta predicción
plt.title("Predicciones vs Valores Reales")
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.show()