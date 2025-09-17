from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Crear una instancia de la aplicación FastAPI
app = FastAPI()

# Cargar el modelo entrenado
model = joblib.load("modelo_casas.pkl")

# Definir la estructura de los datos de entrada con Pydantic
class CasaFeatures(BaseModel):
    superficie: float
    habitaciones: int
    antiguedad: int
    ubicacion: str  # "urbano" o "rural"

@app.get("/")
def read_root():
    return {"message": "API de predicción de precios de casas"}


@app.post("/predict/")
def predict_price(features: CasaFeatures):
    # Convertir la entrada a un DataFrame de pandas
    input_data = pd.DataFrame([features.dict()])

    # Mapear la ubicación a su valor numérico
    # Según el script de entrenamiento: rural=0, urbano=1
    input_data["ubicacion"] = input_data["ubicacion"].map({"rural": 0, "urbano": 1})

    # Reordenar las columnas para que coincidan con el entrenamiento
    input_data = input_data[["superficie", "habitaciones", "antiguedad", "ubicacion"]]

    # Realizar la predicción
    prediction = model.predict(input_data)

    # Devolver el precio estimado
    return {"precio_estimado": prediction[0]}
