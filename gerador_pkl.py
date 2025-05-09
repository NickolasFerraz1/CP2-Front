# -*- coding: utf-8 -*-
"""Case_Ifood_Colearning_Parte4_cp.ipynb"""
 
import numpy as np
import pandas as pd
from pycaret.classification import *
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
 
# Configurações de display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_colwidth', 1000)
 
# ============== Carregamento dos Dados ==============
df = pd.read_csv('data.csv', encoding='utf-8')
df.drop('ID', axis=1, inplace=True, errors='ignore')
df.drop(['Z_CostContact', 'Z_Revenue'], axis=1, inplace=True, errors='ignore')
 
# ============== Tratamento de Dados ==============
# Removendo valores nulos
df.dropna(axis=0, inplace=True)
 
# Ajuste dos tipos de dados
df = df.convert_dtypes()
df.Dt_Customer = pd.to_datetime(df.Dt_Customer)
df.Response = df.Response.astype('bool')
 
# ============== Feature Engineering ==============
# Idade dos clientes
ano_atual = pd.Timestamp.now().year
df['Age'] = ano_atual - df.Year_Birth
df.drop('Year_Birth', axis=1, inplace=True)
 
# Tempo como cliente
dt = pd.Timestamp.now().date()
df['Time_Customer'] = (dt - pd.to_datetime(df['Dt_Customer']).dt.date) / np.timedelta64(1, 'Y')
df.drop('Dt_Customer', axis=1, inplace=True)
 
# Removendo valores incoerentes
index_to_drop = df[df['Marital_Status'].isin(['YOLO', 'Absurd', 'absurd', 'Alone'])].index
df.drop(index_to_drop, inplace=True)
df.reset_index(drop=True, inplace=True)
 
# ============== Preparação do Dataset para Modelagem ==============
df_train_test = df.sample(frac=0.9, random_state=123)
df_valid = df.drop(df_train_test.index)
df_train_test.reset_index(drop=True, inplace=True)
df_valid.reset_index(drop=True, inplace=True)
 
print('Data for Modeling:', df_train_test.shape)
print('Unseen Data For Predictions:', df_valid.shape)
 
# ============== Setup do PyCaret ==============
s = setup(data=df_train_test,
          target='Response',
          fix_imbalance=True,
          remove_outliers=True,
          categorical_features=['Education', 'Marital_Status'],
          session_id=123)
 
# ============== Comparação entre Modelos (sem gráficos) ==============
top_models = compare_models(n_select=5, sort='AUC',
                            include=['lr', 'rf', 'lightgbm', 'gbc'])
 
best_model = top_models[0]
print("\n=== Melhor Modelo Selecionado ===")
print(best_model)
 
# ============== Predições no Conjunto de Validação ==============
predictions = predict_model(best_model, data=df_valid)
 
# ============== Exibição das Métricas ==============
print("\n========== Métricas do Modelo ==========")
print(f"AUC: {round(predictions['Score'].mean(), 4) if 'Score' in predictions.columns else 'N/A'}")
# print(f"Accuracy: {round((predictions['Label'] == df_valid['Response']).mean(), 4)}")
# print(f"Precision: {round(predictions[predictions['Label'] == 1]['Response'].mean(), 4)}")
# print(f"Recall: {round(predictions[predictions['Response'] == 1]['Label'].mean(), 4)}")
# print(f"F1-Score: {round(2 * (predictions[predictions['Label'] == 1]['Response'].mean() * predictions[predictions['Response'] == 1]['Label'].mean()) / (predictions[predictions['Label'] == 1]['Response'].mean() + predictions[predictions['Response'] == 1]['Label'].mean()), 4)}")
 
# ============== Salvando o Modelo Treinado ==============
save_model(best_model, './pickle/pickle_rf_pycaret2')
print("\nModelo salvo em './pickle/pickle_rf_pycaret2.pkl'")

# Salva o conjunto de validação direto na pasta
df_valid.drop('Response', axis=1).to_csv('./pickle/Xtest.csv', index=False)
print("\nConjunto de validação salvo como './pickle/Xtest.csv'")