import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


# lecutra de dataset 
df = pd.read_csv('./healthcare-dataset-stroke-data.csv')

# encontrar cantidad de valoeres null en la columna bmi 
print(df['bmi'].isnull().sum())

# eliminar columnas no importantes como id 
df = df.drop(columns=['id'])

# Convertir variables categóricas en numéricas
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# separar data set 
complete_data = df[df['bmi'].notnull()]
missing_data = df[df['bmi'].isnull()]

## Matriz de correlación 
corr_matrix = df.corr()
'''
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matriz de Correlación")
plt.show()
'''
# Filtrar columnas con correlación >= 0.15 con 'bmi'
relevant_features = corr_matrix['bmi'][abs(corr_matrix['bmi']) >= 0.2].index
print("Variables relevantes:", relevant_features)

complete_data.filter(relevant_features)
X = complete_data.drop(columns=['bmi'])
y = complete_data['bmi']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = regressor.predict(X_test)

# Calcular el error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Preparar datos para predicción
X_missing = missing_data.drop(columns=['bmi'])

# Predecir los valores faltantes
predicted_bmi = regressor.predict(X_missing)

print(predicted_bmi)

# Rellenar los valores faltantes
df.loc[df['bmi'].isnull(), 'bmi'] = predicted_bmi

# Revisar la distribución de 'bmi' después de rellenar valores
print(df['bmi'].describe())

# Comparar distribución antes y después
import seaborn as sns
sns.histplot(df['bmi'], kde=True)
plt.title("Distribución de BMI después de imputar valores")
plt.show()
