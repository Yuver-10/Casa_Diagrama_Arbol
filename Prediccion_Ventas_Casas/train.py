import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# 1) Cargar archivo CSV
data = pd.read_csv("casas_sucias.csv")

# 2) Eliminar duplicados y reemplazar valores raros
data = data.drop_duplicates()
data = data.replace("?", np.nan)       # cambiar "?" por NaN

# 3) Limpiar columnas de texto y pasarlas a números

# --- superficie: quitar "m2", "m" y convertir a float
data["superficie"] = (
    data["superficie"]
    .astype(str)
    .str.replace("m2", "", regex=False)
    .str.replace("m", "", regex=False)
    .str.strip()
)
data["superficie"] = pd.to_numeric(data["superficie"], errors="coerce")

# --- habitaciones: reemplazar palabras y convertir a número
data["habitaciones"] = (
    data["habitaciones"]
    .astype(str)
    .str.replace("tres", "3", regex=False)
    .str.replace("dos", "2", regex=False)
    .str.replace("cuatro", "4", regex=False)
)
data["habitaciones"] = pd.to_numeric(data["habitaciones"], errors="coerce")

# --- antiguedad: cambiar "nueva" a 0, quitar "años"
data["antiguedad"] = (
    data["antiguedad"]
    .astype(str)
    .str.replace("nueva", "0", regex=False)
    .str.replace("años", "", regex=False)
    .str.strip()
)
data["antiguedad"] = pd.to_numeric(data["antiguedad"], errors="coerce")
# pasar valores negativos a positivos
data["antiguedad"] = data["antiguedad"].abs()

# --- ubicacion: normalizar a urbano / rural
data["ubicacion"] = (
    data["ubicacion"]
    .astype(str)
    .str.lower()
    .str.replace("urb", "urbano", regex=False)
    .str.replace("rur", "rural", regex=False)
)
data["ubicacion"] = data["ubicacion"].replace(
    {"urbanoo": "urbano", "rurall": "rural", "urbnaa": "urbano"}
)

# --- precio: convertir a número y limpiar valores extremos
data["precio"] = (
    data["precio"]
    .astype(str)
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)
data["precio"] = pd.to_numeric(data["precio"], errors="coerce")
# eliminar precios anómalos
data = data[(data["precio"] > 1000) & (data["precio"] < 9000000)]

# 4) Convertir ubicacion en variable categórica numérica
# Nota: 'urbano' será 1, 'rural' será 0
data["ubicacion"] = data["ubicacion"].astype("category").cat.codes

# 5) Eliminar filas con valores nulos generados en la limpieza
data = data.dropna()

# 6) Guardar dataset limpio
data.to_csv("casas_limpias.csv", index=False)

print("✅ Limpieza lista. Archivo guardado como casas_limpias.csv")

# --------------------------------------------------------------------------- 
# 4. Modelado con Árbol de Regresión
# --------------------------------------------------------------------------- 

# Separar variables predictoras (X) y variable objetivo (y)
features = ["superficie", "habitaciones", "antiguedad", "ubicacion"]
X = data[features]
y = data["precio"]

# Dividir el dataset en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenar un árbol de regresión
regressor = DecisionTreeRegressor(max_depth=3, random_state=42)
regressor.fit(X_train, y_train)

# Evaluar el modelo
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nResultados del modelo:")
print(f"  - Error Absoluto Medio (MAE): {mae:.2f}")
print(f"  - R2 Score: {r2:.2f}")

# --------------------------------------------------------------------------- 
# 5. Guardar el modelo
# --------------------------------------------------------------------------- 

joblib.dump(regressor, "modelo_casas.pkl")

print("\n✅ Modelo de regresión entrenado y guardado como modelo_casas.pkl")

# --------------------------------------------------------------------------- 
# 6. Visualización del Árbol de Decisión
# --------------------------------------------------------------------------- 

print("\nGenerando visualización del árbol de decisión...")

plt.figure(figsize=(25, 12))
plot_tree(
    regressor, # El arbol ya está limitado a max_depth=3
    feature_names=features,
    filled=True,
    fontsize=8,
    rounded=True
)
plt.title("Árbol de Decisión para Predicción de Precios de Casas (Profundidad 3)")
plt.savefig("arbol_decision.png", dpi=300)
plt.show()

print("\n✅ Diagrama del árbol guardado como arbol_decision.png y mostrado en pantalla.")
