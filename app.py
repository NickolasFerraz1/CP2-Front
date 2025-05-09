import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
import sys
 
# Configurações iniciais do app
st.set_page_config(page_title='Simulador - Case Ifood',
                   page_icon='./images/logo_fiap.png',
                   layout='wide')
 
st.title('Simulador - Conversão de Vendas')
 
# Load modelo treinado
import os
from pathlib import Path
 
 
# --- Carregamento do Modelo com tratamento de erros ---
from joblib import load
try:
    #mdl_rf = load_model('./pickle/pickle_rf_pycaret2')
    mdl_rf = load('./pickle/pickle_rf_pycaret2.pkl')
   
except Exception as e:
    st.error(f"Erro ao carregar modelo: {str(e)}")
    st.error("Verifique se o arquivo pickle existe e está correto")
    st.stop()
   
# Sidebar com opção CSV ou Online
st.sidebar.image('./images/logo_fiap.png', width=100)
st.sidebar.subheader('Auto ML - Fiap [v2]')
database = st.sidebar.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'), horizontal=True)
 
# Define threshold padrão
threshold = st.sidebar.slider('Definir Threshold (slider)', 0.0, 1.0, 0.5, step=0.01)
 
# Input via prompt de linguagem natural
text_input = st.sidebar.text_input("Defina o threshold com linguagem natural (ex: 'usar 70%')")
match = re.search(r'(\d+)', text_input)
if match:
    t_val = int(match.group(1)) / 100
    if 0 <= t_val <= 1:
        threshold = t_val
        st.sidebar.success(f"Threshold ajustado via texto: {threshold}")
 
# --- MODO CSV ---
if database == 'CSV':
    file = st.sidebar.file_uploader('Upload do CSV', type='csv')
    if file:
        Xtest = pd.read_csv(file)
       
        # Verificar se a coluna 'Response' existe (target)
        if 'Response' in Xtest.columns:
            st.write("Dados carregados com sucesso! Contendo a coluna target 'Response'.")
       
        ypred = predict_model(mdl_rf, data=Xtest, raw_score=True)
 
        st.subheader('📄 Visualização dos Dados e Predições')
 
        with st.expander("Visualizar Dados CSV", expanded=False):
            qtd = st.slider("Quantas linhas mostrar?", 5, Xtest.shape[0], step=10, value=5)
            st.dataframe(Xtest.head(qtd))
 
        with st.expander("Visualizar Predições", expanded=True):
            ypred['final_pred'] = (ypred['prediction_score_1'] >= threshold).astype(int)
            c1, c2 = st.columns(2)
            c1.metric("Clientes com Previsão = 1", ypred['final_pred'].sum())
            c2.metric("Clientes com Previsão = 0", len(ypred) - ypred['final_pred'].sum())
 
            def color_pred(val):
                return 'background-color: lightgreen' if val >= threshold else 'background-color: lightcoral'
 
            tipo = st.radio("Visualização:", ['Completa', 'Somente Previsões'])
            view_df = ypred if tipo == 'Completa' else ypred[['prediction_score_1', 'final_pred']]
            st.dataframe(view_df.style.applymap(color_pred, subset=['prediction_score_1']))
 
        # Analytics Tab
        with st.expander("📊 Análise Comparativa (Analytics)", expanded=True):
            st.write("Comparação entre clientes preditos como 0 e 1")
            y0 = ypred[ypred['final_pred'] == 0]
            y1 = ypred[ypred['final_pred'] == 1]
            tabs = st.tabs(["Boxplot", "Histogramas"])
 
            with tabs[0]:
                feature_cols = [col for col in Xtest.columns if Xtest[col].dtype in [np.float64, np.int64]
                              and col not in ['ID', 'Z_CostContact', 'Z_Revenue', 'Response']]
                for col in feature_cols:
                    fig, ax = plt.subplots()
                    sns.boxplot(data=ypred, x='final_pred', y=col, ax=ax)
                    ax.set_title(f'Boxplot - {col}')
                    st.pyplot(fig)
 
            with tabs[1]:
                for col in feature_cols:
                    fig, ax = plt.subplots()
                    sns.histplot(y0[col], kde=True, color='red', label='Classe 0', stat='density')
                    sns.histplot(y1[col], kde=True, color='green', label='Classe 1', stat='density')
                    ax.set_title(f'Histograma - {col}')
                    ax.legend()
                    st.pyplot(fig)
 
# --- MODO ONLINE ---
else:
    st.subheader('🧾 Inserção Manual de Dados')
 
    # Criar um dicionário com todas as colunas necessárias
    features = {
        # Dados demográficos
        "Year_Birth": st.number_input("Ano de Nascimento", min_value=1900, max_value=2023, value=1980),
        "Education": st.selectbox("Educação", ["Graduation", "PhD", "Master", "Basic", "2n Cycle"]),
        "Marital_Status": st.selectbox("Estado Civil", ["Single", "Married", "Together", "Divorced", "Widow", "Alone"]),
       
        # Dados financeiros
        "Income": st.number_input("Renda Anual", min_value=0, value=50000),
       
        # Dados familiares
        "Kidhome": st.selectbox("Crianças em casa", [0, 1, 2]),
        "Teenhome": st.selectbox("Adolescentes em casa", [0, 1, 2]),
       
        # Comportamento de compra
        "Recency": st.slider("Dias desde última compra", 0, 100, 30),
        "MntWines": st.slider("Gasto com Vinhos", 0, 2000, 200),
        "MntFruits": st.slider("Gasto com Frutas", 0, 500, 50),
        "MntMeatProducts": st.slider("Gasto com Carnes", 0, 2000, 150),
        "MntFishProducts": st.slider("Gasto com Peixes", 0, 500, 50),
        "MntSweetProducts": st.slider("Gasto com Doces", 0, 500, 30),
        "MntGoldProds": st.slider("Gasto com Produtos Premium", 0, 500, 50),
       
        # Campanhas de marketing
        "AcceptedCmp1": st.selectbox("Aceitou Campanha 1", [0, 1]),
        "AcceptedCmp2": st.selectbox("Aceitou Campanha 2", [0, 1]),
        "AcceptedCmp3": st.selectbox("Aceitou Campanha 3", [0, 1]),
        "AcceptedCmp4": st.selectbox("Aceitou Campanha 4", [0, 1]),
        "AcceptedCmp5": st.selectbox("Aceitou Campanha 5", [0, 1]),
       
        # Comportamento de compra detalhado
        "NumDealsPurchases": st.slider("Compras com desconto", 0, 20, 2),
        "NumWebPurchases": st.slider("Compras via website", 0, 20, 4),
        "NumCatalogPurchases": st.slider("Compras via catálogo", 0, 20, 2),
        "NumStorePurchases": st.slider("Compras na loja física", 0, 20, 4),
        "NumWebVisitsMonth": st.slider("Visitas ao site/mês", 0, 20, 6),
       
        # Outras variáveis
        "Complain": st.selectbox("Reclamação nos últimos 2 anos", [0, 1]),
    }
 
    # Criar DataFrame com todas as colunas necessárias
    df_input = pd.DataFrame([features])
   
    # Adicionar colunas calculadas que o modelo pode esperar
    df_input['Age'] = pd.Timestamp.now().year - df_input['Year_Birth']
    df_input['Time_Customer'] = (pd.Timestamp.now() - pd.to_datetime('2020-01-01')).days  # Exemplo
   
    st.write("📦 Dados inseridos:")
    st.dataframe(df_input)
 
    # Predição
    if st.button("🔍 Realizar Predição"):
        try:
            pred_result = predict_model(mdl_rf, data=df_input, raw_score=True)
            score = pred_result['prediction_score_1'][0]
            final_pred = int(score >= threshold)
 
            if final_pred == 1:
                st.success(f"✅ Alta probabilidade de conversão! (Score: {score:.2f})")
                st.balloons()
            else:
                st.error(f"❌ Baixa probabilidade de conversão. (Score: {score:.2f})")
               
            # Mostrar detalhes da predição
            with st.expander("Detalhes da Predição"):
                st.write(f"Probabilidade de conversão: {score:.2%}")
                st.write(f"Threshold utilizado: {threshold:.2%}")
                st.progress(score)
               
        except Exception as e:
            st.error(f"Erro ao fazer predição: {str(e)}")
            st.write("Verifique se todas as colunas necessárias estão presentes e com os tipos corretos.")
 