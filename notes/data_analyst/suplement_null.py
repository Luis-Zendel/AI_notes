# Identificar los valores faltantes
missing_bmi = df_healthcare[df_healthcare['bmi'].isnull()]
complete_bmi = df_healthcare[df_healthcare['bmi'].notnull()]
# Matriz de correlación
corr_matrix = complete_bmi.corr()
print(corr_matrix['bmi'].sort_values(ascending=False))
# Convertir variables categóricas en numéricas
df_encoded = pd.get_dummies(df_healthcare, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_encoded.drop(['id', 'bmi', 'stroke'], axis=1))
from sklearn.model_selection import train_test_split

X = df_encoded[complete_bmi.index].drop(['bmi', 'id', 'stroke'], axis=1)
y = complete_bmi['bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor

# Modelo de regresión
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
X_missing = df_encoded[missing_bmi.index].drop(['bmi', 'id', 'stroke'], axis=1)
predicted_bmi = regressor.predict(X_missing)

# Rellenar los valores nulos
df_healthcare.loc[missing_bmi.index, 'bmi'] = predicted_bmi
