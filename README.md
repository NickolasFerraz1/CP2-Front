# CP2-Front

## Descrição

Este projeto é uma aplicação web desenvolvida com Streamlit, que utiliza um modelo de Machine Learning treinado previamente para realizar predições com base em entradas do usuário.

A aplicação foi separada em diferentes módulos e pastas para garantir organização e manutenibilidade. Abaixo está a descrição de cada parte do projeto.

---

## Estrutura do Projeto

### `app.py`
Este é o arquivo principal da aplicação. Ele é responsável por:
- Construir a interface do usuário com Streamlit.
- Receber os dados de entrada.
- Carregar o modelo `.pkl` previamente treinado.
- Realizar as predições.
- Exibir os resultados de forma amigável e visual.

### `gerador_pkl.py`
Este script contém o processo de geração do modelo `.pkl`. Ele é baseado em uma etapa anterior de experimentação, onde diversos algoritmos e combinações de hiperparâmetros foram testados.

Após a definição do melhor modelo com base em métricas de avaliação, o código de treino foi refinado para conter **apenas o pipeline final** usado na geração do modelo `.pkl`. Ou seja, ele representa o treinamento consolidado e objetivo, sem as etapas de experimentação.

### `requirements.txt`
Arquivo que contém todas as dependências necessárias para executar a aplicação. O conteúdo deste arquivo deve ser utilizado por ambientes como o Streamlit Cloud ou virtualenvs locais para instalar as bibliotecas corretamente.

### `images/`
Diretório utilizado para armazenar as imagens usadas na interface da aplicação. Isso pode incluir logotipos, ícones ou imagens demonstrativas utilizadas no `app.py`.

### `pickle/`
Pasta onde está armazenado o modelo `.pkl` treinado. Este arquivo é carregado no `app.py` e usado diretamente para fazer predições com base nos dados inseridos.

---

## Executando o projeto

### Localmente
Para rodar a aplicação localmente, execute os seguintes comandos:

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

# (Opcional) Crie e ative um ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Instale as dependências
pip install -r requirements.txt

# Execute a aplicação
streamlit run app.py
