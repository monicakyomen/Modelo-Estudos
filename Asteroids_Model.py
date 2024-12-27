import kagglehub

# Download latest version
path = kagglehub.dataset_download("shrutimehta/nasa-asteroids-classification")

print("Path to dataset files:", path)

import pandas as pd
import os

# Caminho para o arquivo CSV
csv_path = os.path.join(path, 'nasa.csv')

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(csv_path)

# Mostrar as primeiras linhas do DataFrame
print(df.head())



print(df.columns)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Converter a coluna 'Hazardous' para valores numéricos (True = 1, False = 0)
df['Hazardous'] = df['Hazardous'].astype(int)
# Selecionar as variáveis relevantes
features = [
   'Absolute Magnitude', 'Est Dia in KM(max)', 'Perihelion Distance',
   'Aphelion Dist', 'Orbital Period', 'Perihelion Time','Mean Motion', 'Mean Anomaly'
]
X = df[features]
y = df['Hazardous']

# Tratar valores nulos (substituir por mediana como exemplo)
X = X.fillna(X.median())

# Configurar validação cruzada Stratified K-Fold
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Configurar o modelo XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Definir o grid de hiperparâmetros
param_grid = {
   'n_estimators': [100, 200, 300],
   'max_depth': [3, 5, 7],
   'learning_rate': [0.01, 0.1, 0.2],
   'subsample': [0.8, 1.0],
   'colsample_bytree': [0.8, 1.0]
}
# Configurar o GridSearchCV
grid_search = GridSearchCV(
   estimator=xgb_model,
   param_grid=param_grid,
   scoring='accuracy',
   cv=kf,
   verbose=1,
   n_jobs=-1
)
# Executar o GridSearchCV
grid_search.fit(X, y)

# Resultados do GridSearchCV
print("\n--- Melhores Hiperparâmetros ---")
print(grid_search.best_params_)
print(f"Melhor Acurácia: {grid_search.best_score_:.4f}")

# Avaliar o modelo final
best_model = grid_search.best_estimator_

# Importância das variáveis
feature_importances = pd.DataFrame({
   'Feature': features,
   'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)


# Visualizar importâncias
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, color='#FF66B2')
plt.title('Importância das Variáveis - XGBoost')
plt.show()

# Fazendo a Divisão em treino e teste para plotar a matriz de confusão
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ajustar o GridSearchCV com os dados de treino
grid_search.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = best_model.predict(X_test)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotar a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Perigoso', 'Perigoso'])
disp.plot(cmap=plt.cm.pink)
plt.title('Matriz de chapisco')
plt.show()
