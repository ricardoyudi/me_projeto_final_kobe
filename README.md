# Projeto Kobe: Preditor de Arremessos

## Descrição
Este projeto visa desenvolver uma solução preditiva para arremessos realizados pelo astro da NBA Kobe Bryant durante sua carreira. Utilizando técnicas avançadas de Machine Learning, implementamos dois modelos - regressão e classificação - para prever a probabilidade de acerto de cada arremesso. O projeto segue o framework TDSP da Microsoft, incorporando práticas recomendadas em Engenharia de Machine Learning, como pipelines de processamento de dados, treinamento de modelos, avaliação e monitoramento.

# Estrutura do Projeto (TDSP)

Segue a organização de diretórios adotada neste projeto, alinhada com as práticas recomendadas pelo Team Data Science Process (TDSP) da Microsoft:

```plaintext
projeto-kobe/
│
├── data/
│   ├── raw/                # Dados brutos baixados ou adquiridos
│   └── processed/          # Dados processados e prontos para serem utilizados em modelos
│
├── docs/                   # Documentação relevante do projeto
│
├── models/                 # Modelos treinados, artefatos e versões
│
├── notebooks/              # Jupyter notebooks para exploração de dados e experimentação
│
├── src/                    # Código fonte do projeto
│   ├── data/               # Scripts para download ou geração de dados
│   ├── features/           # Scripts para criação de features
│   ├── models/             # Scripts de treinamento e construção de modelos
│   └── visualization/      # Scripts de visualização
│
├── README.md               # Visão geral do projeto
└── requirements.txt        # Dependências do projeto

## Como Usar
- **Instalação:**
  - Clone o repositório usando `git clone <link-do-repositorio>`.
  - Instale as dependências usando `pip install -r requirements.txt`.

- **Preparação dos Dados:**
  - Execute o script de preparação de dados para processamento em `/src/data/preparation_script.py`.

- **Treinamento de Modelos:**
  - Dois modelos foram treinados: Regressão Logística e Árvore de Decisão.
  - Os scripts de treinamento estão localizados em `/src/models/`.

- **Aplicação e Avaliação dos Modelos:**
  - Uma aplicação web foi desenvolvida com Streamlit para demonstrar a aplicação do modelo em produção.
  - Execute `streamlit run src/app.py` para iniciar a aplicação web.

## Ferramentas Utilizadas
- **PyCaret e Scikit-Learn:** Bibliotecas de Machine Learning para treinamento e avaliação de modelos.
- **MLflow:** Rastreamento de experimentos, monitoramento da saúde do modelo e servindo o modelo final.
- **Streamlit:** Desenvolvimento do dashboard de monitoramento.

## Monitoramento e Estratégia de Retreinamento
Descrição das estratégias para monitorar a saúde do modelo e planos preditivos e reativos para retreinamento.

## Conclusão e Resultados
Análise do desempenho do modelo selecionado em comparação com os dados de desenvolvimento e produção, justificando a aderência do modelo aos novos dados.


