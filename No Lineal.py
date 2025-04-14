import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datos
train_data = pd.read_csv('train.csv')

# 2. Preprocesamiento rápido
cols_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
train_data = train_data.drop(cols_to_drop, axis=1)
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

# 5. Métricas
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

# Validación cruzada
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print("\n--- Métricas del Modelo ---")
print(f"RMSE: {rmse:.2f}")  # Error en dólares
print(f"R²: {r2:.2f}")      # % de varianza explicada
print(f"R² Validación Cruzada (promedio): {cv_scores.mean():.2f}")

# 6. Gráficos

# Distribución de SalePrice
plt.figure(figsize=(10, 6))
sns.histplot(train_data['SalePrice'], kde=True)
plt.title("Distribución de Precios de Viviendas")
plt.xlabel("Precio (USD)")
plt.show()

# Predicciones vs Reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.title("Predicciones vs Valores Reales")
plt.xlabel("Precio Real")
plt.ylabel("Precio Predicho")
plt.show()

# Histograma de residuos
residuos = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuos, kde=True, bins=30)
plt.title("Distribución de Residuos")
plt.xlabel("Error (Real - Predicción)")
plt.show()
