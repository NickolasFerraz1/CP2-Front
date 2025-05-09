import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model
import sys
from joblib import load
import os
from pathlib import Path

# Configurações iniciais do app
st.set_page_config(page_title='Simulador - Case Ifood',
                   page_icon='./images/logo_fiap.png',
                   layout='wide')

# Sidebar com configurações básicas
st.sidebar.image('./images/logo_fiap.png', width=100)
st.sidebar.subheader('Auto ML - Fiap [v2]')
compatibility_mode = True

# --- Carregamento do Modelo com tratamento de erros ---
if not compatibility_mode:
    try:
        # st.info("Tentando carregar o modelo...") # Removed
        
        # Tenta diferentes métodos de carregamento
        try:
            mdl_rf = load('./pickle/pickle_rf_pycaret2.pkl')
            # st.success("Modelo carregado com sucesso usando joblib.load!") # Removed
        except Exception as e1:
            # st.warning(f"Erro ao carregar com joblib.load: {str(e1)}") # Removed
            
            try:
                # Tenta carregamento alternativo com pickle
                with open('./pickle/pickle_rf_pycaret2.pkl', 'rb') as f:
                    mdl_rf = pickle.load(f)
                # st.success("Modelo carregado com sucesso usando pickle.load!") # Removed
            except Exception as e2:
                # st.warning(f"Erro ao carregar com pickle.load: {str(e2)}") # Removed
                
                try:
                    # Tenta carregamento com pycaret (que pode ter formato diferente)
                    mdl_rf = load_model('./pickle/pickle_rf_pycaret2')
                    # st.success("Modelo carregado com sucesso usando pycaret load_model!") # Removed
                except Exception as e3:
                    st.error(f"Todas as tentativas de carregamento falharam.")
                    st.error(f"Erros: \n1. {str(e1)}\n2. {str(e2)}\n3. {str(e3)}")
                    st.error("O arquivo pickle pode ser incompatível devido a diferenças de versão do Python ou bibliotecas.")
                    st.error("Possíveis soluções: Obtenha o código fonte usado para treinar o modelo ou solicite um modelo exportado com versões compatíveis.")
                    st.stop()
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {str(e)}")
        st.error("Verifique se o arquivo pickle existe e está correto")
        st.stop()
else:
    # st.warning("⚠️ Executando em modo de compatibilidade - O modelo não será carregado e as predições retornarão valores aleatórios apenas para teste da interface.") # Removed
    pass # Mantém o bloco else para clareza, mas sem ação

st.title('Simulador - Conversão de Vendas')
with st.expander('Descrição do App', expanded=False):
    st.markdown("""
        Este simulador utiliza um modelo de machine learning para prever a conversão de vendas com base em dados demográficos, comportamentais e históricos de campanhas.<br>
        Você pode inserir os dados manualmente ou carregar um arquivo CSV para gerar predições e visualizar insights analíticos.
    """, unsafe_allow_html=True)

# Continuar com a escolha de fonte de dados
database = st.sidebar.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'), horizontal=True)

# Estilo CSS customizado para o slider
st.markdown(
    """
    <style>
    /* Altera a cor do track do slider */
    [data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(to right, #FF4B4B 0%, #FF4B4B 50%, #3F3F3F 50%, #3F3F3F 100%);
    }

    /* Altera a cor da bolinha (thumb) do slider */
    [data-testid="stSidebar"] [data-testid="stSlider"] span[role="slider"] {
        background-color: #FF4B4B;
        border: 2px solid white;
    }

    /* Altera a cor do texto acima do slider */
    [data-testid="stSidebar"] [data-testid="stSlider"] .css-1y4p8pa {
        color: #FF4B4B;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define threshold padrão com slider estilizado
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
       
        if not compatibility_mode:
            ypred = predict_model(mdl_rf, data=Xtest, raw_score=True)
        else:
            # Modo de compatibilidade - gera predições aleatórias
            ypred = Xtest.copy()
            ypred['prediction_label'] = np.random.choice([0, 1], size=len(Xtest))
            ypred['prediction_score_1'] = np.random.random(size=len(Xtest))
            ypred['prediction_score_0'] = 1 - ypred['prediction_score_1']
 
        st.subheader('📄 Visualização dos Dados e Predições')
 
        with st.expander("Visualizar Dados CSV", expanded=False):
            col1, col2 = st.columns([1, 1])
            with col1:
                qtd = st.slider("Quantas linhas mostrar?", 5, min(100, Xtest.shape[0]), step=5, value=5)
            with col2:
                if Xtest.shape[1] > 10:
                    show_all_cols = st.checkbox("Mostrar todas as colunas", False)
                    if not show_all_cols:
                        num_cols = st.slider("Quantas colunas mostrar?", 5, min(30, Xtest.shape[1]), step=5, value=10)
                        # Seleciona colunas mais importantes ou interessantes
                        selected_cols = ['ID'] if 'ID' in Xtest.columns else []
                        # Adiciona colunas categóricas e binárias primeiro
                        cat_cols = [col for col in Xtest.columns if Xtest[col].dtype == 'object' 
                                  or (Xtest[col].nunique() <= 5 and col not in selected_cols)]
                        selected_cols.extend(cat_cols[:min(5, len(cat_cols))])
                        # Adiciona colunas numéricas
                        num_cols_list = [col for col in Xtest.columns if col not in selected_cols 
                                       and Xtest[col].dtype in [np.float64, np.int64]]
                        selected_cols.extend(num_cols_list[:num_cols-len(selected_cols)])
                        
                        # Verifica se 'Response' existe e a adiciona se não estiver
                        if 'Response' in Xtest.columns and 'Response' not in selected_cols:
                            selected_cols.append('Response')
                            
                        # Exibe tabela com colunas selecionadas
                        st.dataframe(Xtest[selected_cols].head(qtd))
                    else:
                        st.dataframe(Xtest.head(qtd))
                else:
                    st.dataframe(Xtest.head(qtd))
 
        with st.expander("Visualizar Predições", expanded=True):
            # Layout mais compacto para as métricas
            st.markdown("### 📊 Resumo das Predições")
            
            # Calcular a previsão final baseada no threshold
            ypred['final_pred'] = (ypred['prediction_score_1'] >= threshold).astype(int)
            
            # Layout com mais colunas para melhor uso em telas ultrawide
            metric_cols = st.columns(4)
            metric_cols[0].metric("Positivos (1)", ypred['final_pred'].sum())
            metric_cols[1].metric("Negativos (0)", len(ypred) - ypred['final_pred'].sum())
            
            # Taxa de positivos
            positive_rate = (ypred['final_pred'].sum() / len(ypred)) * 100
            metric_cols[2].metric("Taxa de conversão", f"{positive_rate:.1f}%")
            
            # Score médio
            avg_score = ypred['prediction_score_1'].mean() * 100
            metric_cols[3].metric("Score médio", f"{avg_score:.1f}%")
            
            # Definir opções de visualização
            view_options = st.columns([1, 1, 2])
            with view_options[0]:
                tipo = st.radio("Visualização:", ['Completa', 'Compacta'], index=1)
            with view_options[1]:
                qtd_show = st.slider("Linhas", 5, min(50, len(ypred)), 10, step=5)
            with view_options[2]:
                filter_option = st.radio("Filtrar por:", ['Todos', 'Apenas conversões (1)', 'Apenas não-conversões (0)'], horizontal=True)
            
            # Aplicar filtro se necessário
            if filter_option == 'Apenas conversões (1)':
                view_df = ypred[ypred['final_pred'] == 1]
            elif filter_option == 'Apenas não-conversões (0)':
                view_df = ypred[ypred['final_pred'] == 0]
            else:
                view_df = ypred
            
            # Determinar quais colunas mostrar
            if tipo == 'Completa':
                columns_to_show = view_df.columns
            else:
                # Versão compacta mostra apenas algumas colunas relevantes
                base_cols = ['prediction_score_1', 'final_pred']
                id_col = ['ID'] if 'ID' in view_df.columns else []
                response_col = ['Response'] if 'Response' in view_df.columns else []
                # Adicionar algumas variáveis interessantes
                key_vars = []
                for col in ['Income', 'Age', 'Recency', 'MntWines', 'MntMeatProducts']:
                    if col in view_df.columns:
                        key_vars.append(col)
                columns_to_show = id_col + key_vars[:3] + base_cols + response_col
            
            # Função para colorir as células da predição
            def color_pred(val):
                if isinstance(val, (int, float)) and 0 <= val <= 1:
                    if val >= threshold:
                        return f'background-color: rgba(0, 128, 0, {min(val, 0.8)})'
                    else:
                        return f'background-color: rgba(255, 0, 0, {min(1-val, 0.8)})'
                return ''
            
            # Exibir a tabela com formatação melhorada
            st.dataframe(
                view_df[columns_to_show].head(qtd_show).style.applymap(
                    color_pred, subset=['prediction_score_1']
                ).format({
                    'prediction_score_1': '{:.1%}'
                }),
                height=min(qtd_show * 35 + 38, 450)  # Altura dinâmica baseada no número de linhas
            )
 
        # Analytics Tab
        with st.expander("📊 Análise Comparativa (Analytics)", expanded=True):
            st.write("Comparação entre clientes preditos como 0 e 1")
            y0 = ypred[ypred['final_pred'] == 0]
            y1 = ypred[ypred['final_pred'] == 1]
            
            # Define tamanho padrão para todos os gráficos
            fig_width = 3
            fig_height = 1
            
            # Identifica variáveis numéricas
            feature_cols = [col for col in Xtest.columns if Xtest[col].dtype in [np.float64, np.int64]
                          and col not in ['ID', 'Z_CostContact', 'Z_Revenue', 'Response']]
            
            # Controles para ajustar a visualização
            cols_config = st.columns([2, 1, 1])
            with cols_config[0]:
                max_vars = st.slider("Limitar número de variáveis", 1, len(feature_cols) if feature_cols else 1, 
                                    min(6, len(feature_cols) if feature_cols else 1), 
                                    help="Limite o número de variáveis para melhorar a performance")
            with cols_config[1]:
                sort_by = st.selectbox("Ordenar por", ["Nome", "Importância"], index=0)
            with cols_config[2]:
                show_controls = st.checkbox("Mostrar controles avançados", False)
            
            # --- início da lógica de seleção de features revisada ---
            # feature_cols_for_plotting será a lista final de colunas para as abas Boxplot, Histograma, Densidade.
            # multiselect_options será a lista de opções para o widget multiselect.

            all_available_numerical_features = [col for col in Xtest.columns if Xtest[col].dtype in [np.float64, np.int64]
                                              and col not in ['ID', 'Z_CostContact', 'Z_Revenue', 'Response']]
            
            options_for_multiselect = []
            current_features_for_plotting = []

            if not all_available_numerical_features:
                st.warning("Nenhuma variável numérica disponível para análise.")
                # Define feature_cols_for_plotting como uma lista vazia para evitar erros nas abas
                feature_cols_for_plotting = []
            else:
                if sort_by == "Importância":
                    importance = {}
                    for col in all_available_numerical_features:
                        if len(y0) > 0 and len(y1) > 0:
                            mean_y1 = y1[col].mean() if not y1[col].empty else 0
                            mean_y0 = y0[col].mean() if not y0[col].empty else 0
                            std_y0 = y0[col].std() if not y0[col].empty else 1 # Evitar divisão por zero
                            importance[col] = abs(mean_y1 - mean_y0) / (std_y0 + 1e-6) # Adicionado 1e-6 para evitar divisão por zero
                        else:
                            importance[col] = 0
                    
                    sorted_features = sorted(all_available_numerical_features, key=lambda x: importance.get(x, 0), reverse=True)
                    options_for_multiselect = sorted_features
                    current_features_for_plotting = sorted_features
                else: # Ordenar por Nome
                    sorted_features = sorted(all_available_numerical_features)
                    options_for_multiselect = sorted_features
                    current_features_for_plotting = sorted_features

                if show_controls:
                    st.markdown("#### Configurações de visualização")
                    
                    sizing_cols = st.columns(4)
                    with sizing_cols[0]:
                        fig_width = st.slider("Largura Fig", 2.0, 10.0, 4.0, step=0.5, key="fig_w_adv")
                    with sizing_cols[1]:
                        fig_height = st.slider("Altura Fig", 1.0, 8.0, 2.5, step=0.5, key="fig_h_adv")
                    
                    layout_cols = st.columns(4)
                    with layout_cols[0]:
                        num_columns = st.slider("Colunas Grid", 1, 6, 3, key="num_cols_adv")
                    with layout_cols[1]:
                        dpi_val = st.slider("DPI Fig", 70, 150, 90, step=10, 
                                          help="Valores maiores = texto mais nítido, mas gráficos maiores", key="dpi_adv")
                    with layout_cols[2]:
                        font_size = st.slider("Fonte Fig", 8, 14, 10, key="font_adv")
                    with layout_cols[3]:
                        show_labels = st.checkbox("Rótulos Fig", True, key="labels_adv")

                    # Novo slider para o fator IQR (whis)
                    whis_value = st.slider("Fator IQR p/ Outliers (Whis)", 0.5, 5.0, 1.5, step=0.1, 
                                           key="whis_adv", 
                                           help="Define o alcance das hastes do boxplot em múltiplos do IQR. Valores maiores incluem mais pontos nas hastes, mostrando menos outliers.")

                    selected_features_multiselect = st.multiselect("Selecionar variáveis específicas", 
                                                     options=options_for_multiselect,
                                                     default=[],
                                                     help="Deixe vazio para usar as variáveis determinadas pelas configurações acima (ordenadas e limitadas por 'Limitar número de variáveis').")
                    if selected_features_multiselect:
                        feature_cols_for_plotting = selected_features_multiselect
                    else:
                        # Se nada selecionado no multiselect, usa a lista (ordenada) e aplica max_vars
                        feature_cols_for_plotting = current_features_for_plotting[:max_vars]
                    whis_value = 1.5 # Valor padrão para whis se controles avançados estiverem desabilitados
                else:
                    # Valores padrão otimizados para ultrawide (sem controles avançados)
                    fig_width = 4
                    fig_height = 2.5
                    num_columns = 3
                    dpi_val = 90
                    font_size = 10
                    show_labels = True
                    # Sem controles avançados, aplica max_vars à lista (ordenada)
                    feature_cols_for_plotting = current_features_for_plotting[:max_vars]
                    whis_value = 1.5 # Valor padrão para whis se controles avançados estiverem desabilitados

            # --- fim da lógica de seleção de features revisada ---
            
            tabs = st.tabs(["Automático", "Boxplot", "Histogramas", "Densidade"])
            
            with tabs[0]: # Aba Automático
                # Identificar colunas categóricas e numéricas no dataset
                all_cols = [col for col in Xtest.columns if col not in ['ID', 'Z_CostContact', 'Z_Revenue', 'Response']]
                
                # Detecta tipos de colunas
                categorical_cols = []
                numerical_cols = []
                
                for col in all_cols:
                    # Verifica se a coluna é categórica (objeto, string ou poucos valores únicos)
                    if Xtest[col].dtype == 'object' or Xtest[col].nunique() <= 5:
                        categorical_cols.append(col)
                    # Se for numérica
                    elif Xtest[col].dtype in [np.float64, np.int64]:
                        numerical_cols.append(col)
                
                # Limitar o número de colunas para visualização
                categorical_cols = categorical_cols[:max_vars]
                numerical_cols = numerical_cols[:max_vars]
                
                # Informações sobre os tipos identificados
                st.markdown(f"**Detecção automática:** {len(categorical_cols)} colunas categóricas, {len(numerical_cols)} colunas numéricas")
                
                # Gráficos para colunas categóricas (barras)
                if categorical_cols:
                    st.markdown("#### Colunas Categóricas (Gráficos de Barras)")
                    
                    cols = st.columns(num_columns)
                    col_idx = 0
                    
                    for col in categorical_cols:
                        with cols[col_idx % num_columns]:
                            plt.figure(figsize=(fig_width, fig_height), dpi=dpi_val)
                            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_val)
                            
                            # Contagem de valores para cada classe
                            cat_counts_0 = y0[col].value_counts().sort_index()
                            cat_counts_1 = y1[col].value_counts().sort_index()
                            
                            # Obter todas as categorias únicas
                            all_categories = sorted(list(set(list(cat_counts_0.index) + list(cat_counts_1.index))))
                            
                            # Preparar dados para plot
                            counts_0 = [cat_counts_0.get(cat, 0) for cat in all_categories]
                            counts_1 = [cat_counts_1.get(cat, 0) for cat in all_categories]
                            
                            # Posições das barras
                            x = np.arange(len(all_categories))
                            width = 0.4
                            
                            # Plotar barras para cada classe
                            ax.bar(x - width/2, counts_0, width, label='0', color='red', alpha=0.7)
                            ax.bar(x + width/2, counts_1, width, label='1', color='green', alpha=0.7)
                            
                            # Configurar rótulos
                            if len(all_categories) > 5:  # Rótulos na vertical se houver muitas categorias
                                ax.set_xticks(x)
                                ax.set_xticklabels(all_categories, rotation=90)
                            else:
                                ax.set_xticks(x)
                                ax.set_xticklabels(all_categories)
                                
                            ax.set_title(f'{col}', fontsize=font_size)
                            ax.tick_params(labelsize=font_size-1)
                            ax.legend(fontsize=font_size-2)
                            
                            # Rótulos
                            if show_labels:
                                ax.set_xlabel(col, fontsize=font_size-1)
                                ax.set_ylabel('Contagem', fontsize=font_size-1)
                            else:
                                ax.set_xlabel('')
                                ax.set_ylabel('')
                            
                            plt.tight_layout(pad=1.2)
                            st.pyplot(fig)
                            plt.close(fig)  # Liberar memória
                        col_idx += 1
                
                # Gráficos para colunas numéricas (boxplots)
                if numerical_cols:
                    st.markdown("#### Colunas Numéricas (Boxplots)")
                    
                    cols_auto = st.columns(num_columns) # Usar num_columns definido pela lógica acima
                    col_idx_auto = 0
                    
                    for col_auto in numerical_cols: # numerical_cols é definido dentro desta aba
                        with cols_auto[col_idx_auto % num_columns]:
                            plt.figure(figsize=(fig_width, fig_height), dpi=dpi_val)
                            fig_auto, ax_auto = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_val)
                            
                            # Criar boxplot com whis customizável
                            sns.boxplot(data=ypred, x='final_pred', y=col_auto, ax=ax_auto, whis=whis_value)
                            
                            # Configurar gráfico
                            ax_auto.set_title(f'{col_auto}', fontsize=font_size)
                            ax_auto.tick_params(labelsize=font_size-1)
                            
                            if show_labels:
                                ax_auto.set_xlabel('Previsão', fontsize=font_size-1)
                                ax_auto.set_ylabel(col_auto, fontsize=font_size-1)
                            else:
                                ax_auto.set_xlabel('')
                                ax_auto.set_ylabel('')
                                
                            plt.tight_layout(pad=1.2)
                            st.pyplot(fig_auto)
                            plt.close(fig_auto)
                        col_idx_auto += 1
            
            with tabs[1]: # Aba Boxplot (usa feature_cols_for_plotting)
                if not feature_cols_for_plotting:
                    st.info("Nenhuma variável selecionada ou disponível para Boxplots.")
                else:
                    st.markdown("#### Boxplots das Variáveis Selecionadas")
                    cols_boxplot = st.columns(num_columns)
                    col_idx_boxplot = 0
                    
                    for col_bp in feature_cols_for_plotting:
                        with cols_boxplot[col_idx_boxplot % num_columns]:
                            plt.figure(figsize=(fig_width, fig_height), dpi=dpi_val)
                            fig_bp, ax_bp = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_val)
                            # Usar whis_value para controlar outliers
                            sns.boxplot(data=ypred, x='final_pred', y=col_bp, ax=ax_bp, whis=whis_value)
                            ax_bp.set_title(f'{col_bp}', fontsize=font_size)
                            ax_bp.tick_params(labelsize=font_size-1)
                            
                            if show_labels:
                                ax_bp.set_xlabel('Previsão', fontsize=font_size-1)
                                ax_bp.set_ylabel(col_bp, fontsize=font_size-1)
                            else:
                                ax_bp.set_xlabel('')
                                ax_bp.set_ylabel('')
                                
                            plt.tight_layout(pad=1.2)
                            st.pyplot(fig_bp)
                            plt.close(fig_bp)
                        col_idx_boxplot += 1
 
            with tabs[2]: # Aba Histogramas (usa feature_cols_for_plotting)
                if not feature_cols_for_plotting:
                    st.info("Nenhuma variável selecionada ou disponível para Histogramas.")
                else:
                    st.markdown("#### Histogramas das Variáveis Selecionadas")
                    cols_hist = st.columns(num_columns)
                    col_idx_hist = 0
                    
                    for col_hist_plot in feature_cols_for_plotting: # Renomeada variável de loop para evitar conflito
                        with cols_hist[col_idx_hist % num_columns]:
                            plt.figure(figsize=(fig_width, fig_height), dpi=dpi_val)
                            fig_hist, ax_hist = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_val)
                            # Evitar erro se y0 ou y1 estiverem vazios para a coluna específica
                            if not y0[col_hist_plot].empty:
                                sns.histplot(y0[col_hist_plot], kde=False, color='red', label='0', stat='density', alpha=0.4, ax=ax_hist)
                            if not y1[col_hist_plot].empty:
                                sns.histplot(y1[col_hist_plot], kde=False, color='green', label='1', stat='density', alpha=0.4, ax=ax_hist)
                            ax_hist.set_title(f'{col_hist_plot}', fontsize=font_size)
                            ax_hist.tick_params(labelsize=font_size-1)
                            ax_hist.legend(fontsize=font_size-2)
                            
                            if show_labels:
                                ax_hist.set_xlabel(col_hist_plot, fontsize=font_size-1)
                                ax_hist.set_ylabel('Densidade', fontsize=font_size-1)
                            else:
                                ax_hist.set_xlabel('')
                                ax_hist.set_ylabel('')
                                
                            plt.tight_layout(pad=1.2)
                            st.pyplot(fig_hist)
                            plt.close(fig_hist)
                        col_idx_hist += 1
                    
            with tabs[3]: # Aba Densidade (usa feature_cols_for_plotting)
                if not feature_cols_for_plotting:
                    st.info("Nenhuma variável selecionada ou disponível para Gráficos de Densidade.")
                else:
                    st.markdown("#### Gráficos de Densidade das Variáveis Selecionadas")
                    cols_kde = st.columns(num_columns)
                    col_idx_kde = 0
                    
                    for col_kde_plot in feature_cols_for_plotting: # Renomeada variável de loop
                        with cols_kde[col_idx_kde % num_columns]:
                            plt.figure(figsize=(fig_width, fig_height), dpi=dpi_val)
                            fig_kde, ax_kde = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi_val)
                            # Evitar erro se y0 ou y1 estiverem vazios
                            if not y0[col_kde_plot].empty:
                                sns.kdeplot(y0[col_kde_plot], color='red', label='0', ax=ax_kde)
                            if not y1[col_kde_plot].empty:
                                sns.kdeplot(y1[col_kde_plot], color='green', label='1', ax=ax_kde)
                            ax_kde.set_title(f'{col_kde_plot}', fontsize=font_size)
                            ax_kde.tick_params(labelsize=font_size-1)
                            ax_kde.legend(fontsize=font_size-2)
                            
                            if show_labels:
                                ax_kde.set_xlabel(col_kde_plot, fontsize=font_size-1)
                                ax_kde.set_ylabel('Densidade', fontsize=font_size-1)
                            else:
                                ax_kde.set_xlabel('')
                                ax_kde.set_ylabel('')
                                
                            plt.tight_layout(pad=1.2)
                            st.pyplot(fig_kde)
                            plt.close(fig_kde)
                        col_idx_kde += 1
 
# --- MODO ONLINE ---
else:
    st.subheader('🧾 Inserção Manual de Dados')
 
    # Criando abas mais compactas para cada categoria
    form_tabs = st.tabs(["📊 Demográficos", "💰 Financeiros", "🛒 Compras", "📱 Marketing"])
    
    with form_tabs[0]:
        # Dados demográficos - organizados em colunas
        col1, col2, col3 = st.columns(3)
        with col1:
            features = {}
            features["Year_Birth"] = st.number_input("Ano Nascimento", min_value=1900, max_value=2023, value=1980)
            features["Kidhome"] = st.selectbox("Crianças", [0, 1, 2], help="Número de crianças em casa")
        with col2:
            features["Education"] = st.selectbox("Educação", ["Graduation", "PhD", "Master", "Basic", "2n Cycle"])
            features["Teenhome"] = st.selectbox("Adolescentes", [0, 1, 2], help="Número de adolescentes em casa")
        with col3:
            features["Marital_Status"] = st.selectbox("Estado Civil", ["Single", "Married", "Together", "Divorced", "Widow", "Alone"])
            features["Complain"] = st.selectbox("Reclamações", [0, 1], help="Reclamação nos últimos 2 anos")
       
    with form_tabs[1]:
        # Dados financeiros e comportamentais em colunas
        col1, col2 = st.columns(2)
        with col1:
            features["Income"] = st.number_input("Renda Anual", min_value=0, value=50000)
            features["Recency"] = st.slider("Dias desde última compra", 0, 100, 30)
        
        with col2:
            # Adicionar mais campos financeiros aqui se necessário
            st.markdown("##### Gastos por Categoria")
            features["MntWines"] = st.slider("Vinhos", 0, 2000, 200, step=100)
            features["MntFruits"] = st.slider("Frutas", 0, 500, 50, step=50)
            
    with form_tabs[2]:
        # Gastos com produtos em colunas
        col1, col2, col3 = st.columns(3)
        with col1:
            features["MntMeatProducts"] = st.slider("Carnes", 0, 2000, 150, step=100)
            features["MntFishProducts"] = st.slider("Peixes", 0, 500, 50, step=25)
            
        with col2:
            features["MntSweetProducts"] = st.slider("Doces", 0, 500, 30, step=25)
            features["MntGoldProds"] = st.slider("Premium", 0, 500, 50, step=25)
            
        with col3:
            features["NumDealsPurchases"] = st.slider("Compras c/ desconto", 0, 20, 2)
            features["NumWebPurchases"] = st.slider("Compras online", 0, 20, 4)
            
        # Segunda linha para mais opções de compras
        col4, col5, col6 = st.columns(3)
        with col4:
            features["NumCatalogPurchases"] = st.slider("Compras catálogo", 0, 20, 2)
        with col5:
            features["NumStorePurchases"] = st.slider("Compras loja", 0, 20, 4)
        with col6:
            features["NumWebVisitsMonth"] = st.slider("Visitas site/mês", 0, 20, 6)
       
    with form_tabs[3]:
        # Campanhas de marketing em layout compacto
        st.markdown("##### Campanhas Aceitas")
        cols = st.columns(5)
        for i, col in enumerate(cols):
            camp_key = f"AcceptedCmp{i+1}"
            features[camp_key] = col.selectbox(f"Camp. {i+1}", [0, 1], key=f"camp_{i+1}")
    
    # Criar DataFrame com todas as colunas necessárias
    df_input = pd.DataFrame([features])
   
    # Adicionar colunas calculadas que o modelo pode esperar
    df_input['Age'] = pd.Timestamp.now().year - df_input['Year_Birth']
    df_input['Time_Customer'] = (pd.Timestamp.now() - pd.to_datetime('2020-01-01')).days  # Exemplo
   
    # Resumo dos dados inseridos mais compacto
    st.markdown("##### 📦 Resumo dos Dados")
    display_cols = st.columns(4)
    
    # Resumo dos dados demográficos
    with display_cols[0]:
        st.markdown("**Perfil**")
        st.markdown(f"👤 {df_input['Age'].values[0]} anos, {df_input['Education'].values[0]}")
        st.markdown(f"👪 {df_input['Marital_Status'].values[0]}, Crianças: {df_input['Kidhome'].values[0]}")
        
    # Resumo financeiro
    with display_cols[1]:
        st.markdown("**Financeiro**")
        st.markdown(f"💵 ${df_input['Income'].values[0]:,.0f}")
        st.markdown(f"🍷 Vinhos: ${df_input['MntWines'].values[0]:,.0f}")
        
    # Resumo de gastos
    with display_cols[2]:
        st.markdown("**Gastos**")
        st.markdown(f"🥩 Carnes: ${df_input['MntMeatProducts'].values[0]:,.0f}")
        st.markdown(f"✨ Premium: ${df_input['MntGoldProds'].values[0]:,.0f}")
        
    # Resumo de compras
    with display_cols[3]:
        st.markdown("**Compras**")
        st.markdown(f"🔄 Recência: {df_input['Recency'].values[0]} dias")
        st.markdown(f"🌐 Web: {df_input['NumWebPurchases'].values[0]}, Loja: {df_input['NumStorePurchases'].values[0]}")
 
    # Predição - layout mais compacto
    btn_cols = st.columns([3, 1])
    with btn_cols[0]:
        predict_btn = st.button("🔍 Realizar Predição", use_container_width=True)
        
    if predict_btn:
        try:
            if not compatibility_mode:
                pred_result = predict_model(mdl_rf, data=df_input, raw_score=True)
                score = pred_result['prediction_score_1'][0]
            else:
                # Modo de compatibilidade - gera predição aleatória
                score = np.random.random()
                
            final_pred = int(score >= threshold)
 
            # Layout mais atraente para o resultado da predição
            # Colunas mais compactas para o resultado
            st.markdown("---")
            result_cols = st.columns([1, 3])
            with result_cols[0]:
                if final_pred == 1:
                    st.markdown("### ✅")
                else:
                    st.markdown("### ❌")
                
            with result_cols[1]:
                if final_pred == 1:
                    st.success(f"**Alta probabilidade de conversão!** (Score: {score:.2%})")
                    st.balloons()
                else:
                    st.error(f"**Baixa probabilidade de conversão.** (Score: {score:.2%})")
               
            # Mostrar apenas métricas básicas sem gráficos
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                st.metric("Score", f"{score:.1%}")
                
            with metrics_cols[1]:
                if "Income" in df_input.columns:
                    renda = df_input["Income"].values[0]
                    st.metric("Renda", f"${renda:,.0f}")
                    
            with metrics_cols[2]:
                idade = df_input["Age"].values[0]
                st.metric("Idade", f"{idade}")
                
            with metrics_cols[3]:
                if "Recency" in df_input.columns:
                    recencia = df_input["Recency"].values[0]
                    st.metric("Recência", f"{recencia}d")
                    
            # Informações adicionais em um expander
            with st.expander("Informações adicionais", expanded=False):
                st.write(f"Threshold utilizado: {threshold:.2%}")
                if score >= threshold:
                    st.write("O cliente tem alta probabilidade de conversão.")
                else:
                    st.write("O cliente tem baixa probabilidade de conversão.")
                    
                if compatibility_mode:
                    st.warning("⚠️ Modo compatibilidade - Predição gerada aleatoriamente")
                
        except Exception as e:
            st.error(f"Erro ao fazer predição: {str(e)}")
            st.write("Verifique se todas as colunas necessárias estão presentes e com os tipos corretos.")
 