import pandas as pd

# Cargar el dataset
dataset = pd.read_csv("C:/Users/losal/Desktop/temporal/DEBER/ec_properties.csv")

# Imprimir las columnas
print("Columnas del dataset:", dataset.columns)

# Imprimir la forma del dataset (filas, columnas)
print("Shape del dataset:", dataset.shape)

# Mostrar las primeras 5 filas
print("Primeras filas del dataset:")
print(dataset.head())
