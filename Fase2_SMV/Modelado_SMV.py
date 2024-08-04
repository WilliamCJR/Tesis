import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Datos de ejemplo (reemplaza esto con tu DataFrame real)
data = pd.DataFrame({
    'ANNO': [2021, 2022, 2023, 2024],
    'MES': [1, 2, 3, 12],
    'ID_EMPRESA': [800153993, 830114921, 830122566, 901354361],
    'ID_TECNOLOGIA': [1, 2, 3, 4],
    'ID_MUNICIPIO': [11001, 76001, 5001, 8001],
    'PORCENTAJE_DISPONIBILIDAD': [97.5, 98.0, 97.8, 98.2]
})

# Separar los datos del año 2023
data_2023 = data[data['ANNO'] == 2023]
data_no_2023 = data[data['ANNO'] != 2023]

# Separar características y objetivo
X_no_2023 = data_no_2023[['ANNO', 'MES', 'ID_EMPRESA', 'ID_TECNOLOGIA', 'ID_MUNICIPIO']]
y_no_2023 = data_no_2023['PORCENTAJE_DISPONIBILIDAD']

X_2023 = data_2023[['ANNO', 'MES', 'ID_EMPRESA', 'ID_TECNOLOGIA', 'ID_MUNICIPIO']]
y_2023 = data_2023['PORCENTAJE_DISPONIBILIDAD']

# Definir el preprocesamiento para variables categóricas y numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ('anno', StandardScaler(), ['ANNO']),  # Normalizar el año
        ('mes', OneHotEncoder(handle_unknown='ignore'), ['MES']),  # Codificar el mes
        ('id_empresa', OneHotEncoder(handle_unknown='ignore'), ['ID_EMPRESA']),  # Codificar ID_EMPRESA
        ('id_tecnologia', OneHotEncoder(handle_unknown='ignore'), ['ID_TECNOLOGIA']),  # Codificar ID_TECNOLOGIA
        ('id_municipio', OneHotEncoder(handle_unknown='ignore'), ['ID_MUNICIPIO'])  # Codificar ID_MUNICIPIO
    ])

# Crear un pipeline que incluya preprocesamiento y el modelo SVM
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', SVR(kernel='linear'))  # Modelo SVM con kernel lineal
])

# Dividir los datos no de 2023 en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_no_2023, y_no_2023, test_size=0.2, random_state=42)

# Entrenar el modelo
pipeline.fit(X_train, y_train)

# Evaluar el modelo con datos del año 2023
y_pred_2023 = pipeline.predict(X_2023)
mse_2023 = mean_squared_error(y_2023, y_pred_2023)
print(f"Mean Squared Error (2023): {mse_2023}")

# Imprimir las predicciones para el año 2023
print("Predicciones para 2023:", y_pred_2023)
