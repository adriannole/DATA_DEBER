import pandas as pd

import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import	seaborn as sns
from matplotlib.ticker import ScalarFormatter





# Cargar el dataset
dataset = pd.read_csv("C:/Users/losal/Desktop/temporal/DEBER/ec_properties.csv")

# Imprimir las columnas
print("Columnas del dataset:", dataset.columns)

# Imprimir la forma del dataset (filas, columnas)
print("Shape del dataset:", dataset.shape)

# Mostrar las primeras 5 filas
print("Primeras filas del dataset:")
print(dataset.head())



print(dataset.tail())


print(dataset.describe())


print(dataset.isnull().sum())

print(dataset["property_type"].unique())

print(dataset["property_type"].value_counts())



plt.figure(figsize=(10,5))
plt.yscale('log')
plt.gca().yaxis.set_major_formatter(ScalarFormatter())
ax = sns.countplot(data = dataset, x = "property_type")
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha="right")


print (plt.show())


#--------------------------------------------------------------------#




fig= plt.subplots(figsize=(20,18),constrained_layout=True)
grid = gridspec.GridSpec(2, 1, height_ratios=[1, 3])

ax1=plt.subplot(grid[0])
sns.countplot(data=dataset,y="l2",order=dataset["l2"].value_counts().index,ax=ax1,color="g")

ax1.set_yticklabels(ax1.get_yticklabels(),fontsize="medium")
ax1.set_title("Distribucion segun el G.B.A.", fontsize= 'large')

ax2=plt.subplot(grid[1])
sns.countplot(data=dataset,x="l3",order=dataset["l3"].value_counts().index,ax=ax2,color="b")


ax2.set_title("Distribucion segun barrios", fontsize= 'large')
ax2.set_xticklabels(ax2.get_xticklabels(),rotation=90,ha="right")
plt.yticks(fontsize= 11)
ax1.grid()
ax2.grid()
print (plt.show())





#dataset = dataset[((dataset['property_type'] == "Departamento") |(dataset['property_type'] == "Casa") | (dataset['property_type'] == "PH"))  & (dataset['l2'] == "Quito")] 
# print (dataset.shape)

# Importar las librerías necesarias
import pandas as pd
from pandas_profiling import ProfileReport

# Cargar tu dataset (ajusta 'dataset' al nombre real de tu DataFrame)


# Generar el reporte
reporte = ProfileReport(dataset, title="Pandas Profiling Reporte", minimal=True)

# Mostrar el reporte
reporte.to_notebook_iframe()
