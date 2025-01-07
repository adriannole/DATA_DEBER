import pandas as pd

import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import	seaborn as sns
from matplotlib.ticker import ScalarFormatter




# Cargar el dataset
dataset = pd.read_csv("C:/Users/Adrian Nole/Desktop/ec_properties.csv")

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




import pandas as pd
from ydata_profiling import ProfileReport

# Filtrar el dataset
dataset = dataset[
    ((dataset['property_type'] == "Departamento") |
     (dataset['property_type'] == "Casa") |
     (dataset['property_type'] == "PH")) &
    (dataset['l2'] == "Pichincha")
]
print(dataset.shape)

# Verificar que el dataset no esté vacío
if dataset.empty:
    print("El dataset está vacío después de filtrar. Verifica los filtros.")
else:
    # Generar el reporte
    reporte = ProfileReport(dataset, title="YData Profiling Reporte", minimal=True)

# Guardar el reporte en un archivo HTML
    reporte.to_file("reporte.html")
    print("Reporte guardado como 'reporte.html'. Ábrelo en tu navegador para visualizarlo.")
print("Dimensiones del dataset filtrado:", dataset.shape)

sns.scatterplot(data=dataset, x='surface_total', y='surface_covered')
print(plt.grid())
print(plt.show())

dataset.drop(dataset.loc[dataset['surface_covered'] > dataset['surface_total']].index,inplace=True ,axis=0)
sns.scatterplot(data=dataset, x='surface_total', y='surface_covered')
print(plt.grid())
print(plt.show())

dataset.shape
for i in range(dataset.shape[1]):
    print(i,len(pd.unique(dataset.iloc[:,i])))
    
dataset = dataset.drop(["l1","l2","currency","operation_type"],axis=1)
print(dataset.columns)


for i in range(dataset.shape[1]):
    num=len(pd.unique(dataset.iloc[:,i]))
    porcentaje=float(num)/dataset.shape[0]*100
    print("%d, %d, %.1f%%"%(i,num,porcentaje))
    
    
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Asegúrate de cargar correctamente el DataFrame 'dataset' antes
dataset = dataset.drop(["start_date", "end_date", "title", "description"], axis=1)

# Filtrar solo columnas numéricas
dataset_numeric = dataset.select_dtypes(include=[float, int])

# Verificar si hay columnas numéricas
if dataset_numeric.empty:
    print("No hay columnas numéricas en el dataset para calcular la correlación.")
else:
    # Crear el mapa de calor
    plt.figure(figsize=(8, 8))
    sns.heatmap(dataset_numeric.corr(), annot=True, cmap="coolwarm")
    plt.title("Mapa de calor de correlaciones")
    plt.show()
dataset1 = dataset.copy()
dataset1.isnull().sum()

X= dataset1.drop(["price","created_on","lat","lon","property_type","l3"],axis=1)
y=dataset1["price"]
X.head()
print(X.head())  # Imprime las primeras 5 filas de X
X.isnull().sum()


from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import numpy as np

# Seleccionar solo las columnas relevantes
X = dataset[["rooms", "bedrooms", "bathrooms", "surface_total", "surface_covered"]]
y = dataset["price"]

# Imputar los valores faltantes en y (price) usando KNNImputer
imputer_y = KNNImputer(n_neighbors=6)  # Usamos el mismo número de vecinos
y_imputed = imputer_y.fit_transform(y.values.reshape(-1, 1)).flatten()

# Imputación de los valores faltantes en X (si existen)
imputer_X = KNNImputer(n_neighbors=6)
X_imputed = imputer_X.fit_transform(X)

# Modelo de árbol de decisión
tree = DecisionTreeRegressor(max_depth=10, random_state=42)
tree.fit(X_imputed, y_imputed)

# Predicción y evaluación
y_pred = tree.predict(X_imputed)
r2 = metrics.r2_score(y_imputed, y_pred)
RMSE = np.sqrt(mean_squared_error(y_imputed, y_pred))

# Almacenar y mostrar resultados
print("RMSE según la imputación de KNN:", RMSE)


from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import numpy as np

# Seleccionar solo las columnas relevantes de X
X = dataset[["rooms", "bedrooms", "bathrooms", "surface_total", "surface_covered"]]

# Filtrar las filas donde y (price) no tiene valores NaN
dataset_clean = dataset.dropna(subset=["price"])
X_clean = dataset_clean[["rooms", "bedrooms", "bathrooms", "surface_total", "surface_covered"]]
y_clean = dataset_clean["price"]

# Lista de estrategias de imputación
tipos = ['mean', 'median', 'most_frequent', 'constant']

# Almacenar resultados
resultado = []

for t in tipos:
    # Imputar los valores faltantes en X usando SimpleImputer
    imputer_X = SimpleImputer(strategy=t)
    imputer_X.fit(X_clean)
    X_trans = imputer_X.transform(X_clean)
    
    # Crear y entrenar el modelo de árbol de decisión
    tree = DecisionTreeRegressor(max_depth=10, random_state=42)
    tree.fit(X_trans, y_clean)
    
    # Predicción y evaluación
    y_pred = tree.predict(X_trans)
    r2 = metrics.r2_score(y_clean, y_pred)
    RMSE = np.sqrt(mean_squared_error(y_clean, y_pred))
    
    # Almacenar y mostrar resultados
    resultado.append(r2)
    print("La estrategia utilizada--->", t)
    print("RMSE según el tipo de estrategia:", RMSE)
    print("----------------------------------------")


for x in ["Casa","PH", "Departamento"]:
    Q1 = dataset[dataset["property_type"]==x]["price"].quantile(0.25)
    Q3 = dataset[dataset["property_type"]==x]["price"].quantile(0.75)
    IQR = Q3 - Q1
    lim_min = dataset[dataset["property_type"]==x]["price"].quantile(0.01)
    lim_max = Q3 + (IQR*1.5)
    print(x)
    print("el precio maximo es {}, el precio minimo es {} y el IQR {}" .format(lim_max,lim_min,IQR))
    print("-------------------------------------------------------------------")


import matplotlib.pyplot as plt
import seaborn as sns

# Filtrar las propiedades por tipo
dptos = dataset[dataset["property_type"] == "Departamento"]
phs = dataset[dataset["property_type"] == "PH"]
casas = dataset[dataset["property_type"] == "Casa"]

# Filtrar los precios dentro de los rangos establecidos
dptos = dptos[(dptos.price <= 392500.0) & (dptos.price >= 65000.0)]
phs = phs[(phs.price <= 428500.0) & (phs.price >= 75000.0)]
casas = casas[(casas.price <= 706000.0) & (casas.price >= 110000.0)]

# Crear subgráficos
figure, (ax1, ax2, ax3) = plt.subplots(3, constrained_layout=True, figsize=(10, 10))

# Títulos de los gráficos
ax1.set_title("Relacion tipo de propiedad y precio (En miles de Dolares) - Departamento")
ax2.set_title("Relacion tipo de propiedad y precio (En miles de Dolares) - PH")
ax3.set_title("Relacion tipo de propiedad y precio (En miles de Dolares) - Casa")

# Gráficos de caja (boxplots)
sns.boxplot(data=dptos, x="price", y="property_type", ax=ax1)
sns.boxplot(data=phs, x="price", y="property_type", ax=ax2)
sns.boxplot(data=casas, x="price", y="property_type", ax=ax3)

# Agregar las rejillas a cada gráfico
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

# Mostrar el gráfico
plt.show()
