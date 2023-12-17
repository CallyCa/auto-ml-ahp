<div  align="center">
 <h1>
  Análise do perfil dos beneficiários por sexo e raça/cor no ProUNI
 </h1>
</div>

## Objetivo do trabalho

O objetivo deste trabalho é realizar uma análise do perfil dos beneficiários do Programa Universidade para Todos (ProUNI) com foco nas variáveis de **sexo** e **raça/cor**. O ProUNI é um programa do governo brasileiro que oferece bolsas de estudos em instituições privadas de ensino superior. Esta análise visa responder a perguntas como:

- Existe alguma disparidade na concessão de bolsas com base no sexo dos beneficiários?
- Há diferenças na distribuição das bolsas em relação à raça/cor dos beneficiários?

## Metodologia

Neste trabalho, foram utilizados dados do ProUNI para realizar a análise. Os dados foram obtidos de **https://brasil.io/datasets/** e **https://dados.gov.br/home** e carregados em um banco de dados Neo4j para facilitar a análise e as consultas.

A análise de dados foi dividida em várias etapas:

1. **Inserção de Dados**: Os dados brutos foram inseridos em um banco de dados Neo4j. Para isso, foi criada uma classe `InserirDadosNeo4j` que carrega os dados a partir de um arquivo CSV e insere os registros em tabelas correspondentes no banco de dados.

2. **Criação de Relacionamentos**: Os relacionamentos entre os dados foram estabelecidos. Para isso, foi criada a classe `CriarRelacionamentosNeo4j`, que define as conexões entre as entidades no banco de dados.

3. **Análise de Aderência**: Foi realizada uma análise para verificar o quão bem os dados se encaixam em um modelo específico no Neo4j. A classe `AnalisarAderenciaModeloNeo4j` foi criada para calcular a aderência dos dados.

4. **Consulta de Dados**: A classe `ConsultaDadosNeo4j` permite consultar os dados do banco de dados Neo4j e formatá-los em um formato mais legível. Os dados consultados são relativos a beneficiários, instituições de ensino superior, cursos, bolsas, regiões, unidades federativas e municípios.

5. **Análise de Dados**: Com os dados formatados, é possível realizar análises relacionadas ao perfil dos beneficiários com base no sexo e na raça/cor. Diferentes métricas e visualizações podem ser geradas a partir dos dados consultados.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


