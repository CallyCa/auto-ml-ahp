import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

class DataVisualization:
    @staticmethod
    def boxplot_plot(dataframe, column, xlabel=None, ylabel=None, titulo=None):
        """
        Plota um BoxPlot para a coluna especificada do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        :param coluna: Nome da coluna para a qual deseja gerar o BoxPlot.
        :param xlabel: Rótulo do eixo x. Se não fornecido, será 'Valores' por padrão.
        :param ylabel: Rótulo do eixo y. Se não fornecido, será 'Frequência' por padrão.
        :param titulo: Título do gráfico. Se não fornecido, será 'BoxPlot para "{coluna}"' por padrão.
        """
        if (column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column].dtype)):
            plt.figure(figsize=(8, 6))
            plt.boxplot(dataframe[column], vert=False)
            plt.title(titulo.format(coluna=column) if titulo else f'BoxPlot para "{column}"')
            plt.xlabel(xlabel if xlabel else 'Valores')
            plt.ylabel(ylabel if ylabel else 'Frequência')
            plt.show()
        else:
            print(f'A coluna "{column}" não existe no DataFrame ou não é do tipo numérico.')
    
    @staticmethod
    def plotar_histogramas(dataframe, colunas_numericas, xlabel=None, ylabel=None, titulo=None):
        """
        Plota histogramas para as variáveis numéricas especificadas do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        :param colunas_numericas: Lista de nomes das colunas numéricas para as quais deseja gerar os histogramas.
        :param xlabel: Rótulo do eixo x. Se não fornecido, será 'Valores' por padrão.
        :param ylabel: Rótulo do eixo y. Se não fornecido, será 'Frequência' por padrão.
        :param titulo: Título do gráfico. Se não fornecido, será 'Histograma para "{coluna}"' por padrão.
        """
        # Largura proporcional à altura para manter a proporção com o mapa de calor
        largura = 10

        # Certifique-se de que colunas_numericas seja uma lista, mesmo que tenha apenas uma coluna
        if not isinstance(colunas_numericas, list):
            colunas_numericas = [colunas_numericas]

        for coluna in colunas_numericas:
            if coluna in dataframe.columns:
                # Converta a coluna para o tipo numérico (float) se possível
                dataframe[coluna] = pd.to_numeric(dataframe[coluna], errors='coerce')

                if pd.api.types.is_numeric_dtype(dataframe[coluna].dtype):
                    # Crie um novo gráfico para cada coluna com largura proporcional
                    plt.figure(figsize=(largura, largura * 0.75))
                    plt.hist(dataframe[coluna].dropna(), bins=30, color='skyblue', edgecolor='black')
                    plt.title(titulo.format(coluna=coluna) if titulo else f'Histograma para "{coluna}"')
                    plt.xlabel(xlabel if xlabel else 'Valores')
                    plt.ylabel(ylabel if ylabel else 'Frequência')
                    plt.show()
                else:
                    print(f'A coluna "{coluna}" não é do tipo numérico. Tipos de dados encontrados: {dataframe[coluna].dtype}')
    
    @staticmethod
    def plot_bar_chart(dataframe, column, xlabel=None, ylabel=None, titulo=None):
        """
        Plota um gráfico de barras para a coluna especificada do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        :param column: Nome da coluna para a qual deseja gerar o gráfico de barras.
        :param xlabel: Rótulo do eixo x. Se não fornecido, será 'Categorias' por padrão.
        :param ylabel: Rótulo do eixo y. Se não fornecido, será 'Frequência' por padrão.
        :param titulo: Título do gráfico. Se não fornecido, será 'Gráfico de barras para "{coluna}"' por padrão.
        """

        # Verifique se a coluna é uma lista de strings e se existe no DataFrame
        if isinstance(column, list) and all(isinstance(item, str) for item in column) and all(item in dataframe.columns for item in column):
            plt.figure(figsize=(8, 6))
            for col in column:
                plt.bar(dataframe[col].value_counts().index, dataframe[col].value_counts(), label=col)

            plt.title(titulo.format(coluna=', '.join(column)) if titulo else f'Gráfico de barras para "{", ".join(column)}"')
            plt.xlabel(xlabel if xlabel else 'Categorias')
            plt.ylabel(ylabel if ylabel else 'Frequência')
            plt.legend()
            plt.show()
        else:
            print(f'Uma ou mais colunas fornecidas não existem no DataFrame ou não são do tipo string.')
    
    @staticmethod
    def plot_density_plot(dataframe, column, xlabel=None, ylabel=None, titulo=None):
        """
        Plota um gráfico de densidade para a coluna numérica especificada do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        :param column: Nome da coluna numérica para a qual deseja gerar o gráfico de densidade.
        :param xlabel: Rótulo do eixo x. Se não fornecido, será 'Valores' por padrão.
        :param ylabel: Rótulo do eixo y. Se não fornecido, será 'Densidade' por padrão.
        :param titulo: Título do gráfico. Se não fornecido, será 'Gráfico de densidade para "{coluna}"' por padrão.
        """

        # Verifique se a coluna é numérica
        if isinstance(column, str) and column in dataframe.columns and pd.api.types.is_numeric_dtype(dataframe[column].dtype):
            plt.figure(figsize=(8, 6))
            sns.kdeplot(dataframe[column].dropna(), color='skyblue', fill=True)
            plt.title(titulo.format(coluna=column) if titulo else f'Gráfico de densidade para "{column}"')
            plt.xlabel(xlabel if xlabel else 'Valores')
            plt.ylabel(ylabel if ylabel else 'Densidade')
            plt.show()
        else:
            print(f'A coluna "{column}" não existe no DataFrame ou não é do tipo numérico. Tipos de dados encontrados: {dataframe[column].dtype}')

    @staticmethod
    def plotar_dispersao(dataframe, coluna_x, coluna_y, xlabel=None, ylabel=None, titulo=None):
        """
        Plota um Gráfico de Dispersão para visualizar a relação entre duas variáveis.

        :param dataframe: DataFrame a ser visualizado.
        :param coluna_x: Nome da coluna para o eixo x.
        :param coluna_y: Nome da coluna para o eixo y.
        :param xlabel: Rótulo do eixo x. Se não fornecido, será o nome da coluna_x.
        :param ylabel: Rótulo do eixo y. Se não fornecido, será o nome da coluna_y.
        :param titulo: Título do gráfico. Se não fornecido, será 'Gráfico de Dispersão entre "{coluna_x}" e "{coluna_y}"' por padrão.
        """
        if coluna_x in dataframe.columns and coluna_y in dataframe.columns:
            plt.figure(figsize=(10, 6))

            if isinstance(dataframe[coluna_x].dtype, CategoricalDtype):
                sns.scatterplot(x=coluna_x, y=coluna_y, data=dataframe, hue=coluna_x)
            else:
                sns.scatterplot(x=coluna_x, y=coluna_y, data=dataframe)

            plt.title(titulo.format(coluna_x=coluna_x, coluna_y=coluna_y) if titulo else f'Gráfico de Dispersão entre "{coluna_x}" e "{coluna_y}"')
            plt.xlabel(xlabel if xlabel else coluna_x)
            plt.ylabel(ylabel if ylabel else coluna_y)
            plt.show()
        else:
            print(f'Uma ou ambas as colunas especificadas não existem no DataFrame.')

    @staticmethod
    def plotar_heatmap(dataframe):
        """
        Gera um Mapa de Calor (Heatmap) para visualizar a correlação entre as variáveis numéricas do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        """
        # Seleciona apenas as variáveis numéricas
        variaveis_numericas = dataframe.select_dtypes(include='number')

        # Calcula a matriz de correlação
        matriz_correlacao = variaveis_numericas.corr()

        # Configuração do tamanho da figura
        plt.figure(figsize=(12, 8))

        # Cria o heatmap utilizando Seaborn
        sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', fmt=".2f")

        # Adiciona título ao mapa de calor
        plt.title('Mapa de Calor - Correlação entre Variáveis Numéricas')

        # Exibe o mapa de calor
        plt.show()

    @staticmethod
    def plotar_grafico_linhas(dataframe, coluna_x, coluna_y):
        """
        Plota um gráfico de linhas para duas colunas específicas do DataFrame.

        :param dataframe: DataFrame a ser visualizado.
        :param coluna_x: Nome da coluna para o eixo x.
        :param coluna_y: Nome da coluna para o eixo y.
        """
        if coluna_x in dataframe.columns and coluna_y in dataframe.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(dataframe[coluna_x], dataframe[coluna_y], marker='o', linestyle='-')
            plt.title(f'Gráfico de Linhas para {coluna_y} em relação a {coluna_x}')
            plt.xlabel(coluna_x)
            plt.ylabel(coluna_y)
            plt.grid(True)
            plt.show()
        else:
            print(f'Uma ou ambas as colunas ({coluna_x}, {coluna_y}) não existem no DataFrame.')
            
    @staticmethod
    def plotar_tabela_contingencia(dataframe, *colunas_categoricas):
        """
        Plota uma tabela de contingência para analisar a relação entre variáveis categóricas.

        :param dataframe: DataFrame a ser visualizado.
        :param colunas_categoricas: Nomes das colunas categóricas.
        """
        if len(colunas_categoricas) < 2:
            print("Forneça pelo menos duas colunas categóricas para criar uma tabela de contingência.")
            return

        tabela_contingencia = pd.crosstab(dataframe[colunas_categoricas[0]], dataframe[colunas_categoricas[1]])
        print("Tabela de Contingência:")
        print(tabela_contingencia)

        # Teste qui-quadrado
        chi2, p, _, _ = chi2_contingency(tabela_contingencia)
        print(f"\nResultado do Teste Qui-Quadrado:")
        print(f"Chi2: {chi2}")
        print(f"P-valor: {p}")

        # Plotar heatmap da tabela de contingência
        plt.figure(figsize=(10, 6))
        sns.heatmap(tabela_contingencia, annot=True, cmap='coolwarm', fmt="d")
        plt.title(f'Relação entre {", ".join(colunas_categoricas)}')
        plt.show()
        
    @staticmethod
    def plotar_scatterplot_com_cores_codificadas_por_classe(dataframe, coluna_x, coluna_y, xlabel=None, ylabel=None, titulo=None, encoded_labels=None):
        """
        Plota um Scatterplot com cores codificadas por classe para visualizar a separação entre classes.

        :param dataframe: DataFrame a ser visualizado.
        :param coluna_x: Nome da coluna para o eixo x.
        :param coluna_y: Nome da coluna para o eixo y.
        :param xlabel: Rótulo do eixo x.
        :param ylabel: Rótulo do eixo y.
        :param titulo: Título do gráfico.
        :param encoded_labels: Labels codificadas para colorir o gráfico.
        """
        if coluna_x in dataframe.columns and coluna_y in dataframe.columns:
            plt.figure(figsize=(10, 6))
            
            if encoded_labels is not None:
                sns.scatterplot(x=coluna_x, y=coluna_y, data=dataframe, hue=encoded_labels)
            else:
                sns.scatterplot(x=coluna_x, y=coluna_y, data=dataframe, hue=coluna_x)
            
            plt.title(titulo if titulo else f'Scatterplot entre "{coluna_x}" e "{coluna_y}"')
            plt.xlabel(xlabel if xlabel else coluna_x)
            plt.ylabel(ylabel if ylabel else coluna_y)
            plt.show()
        else:
            print(f'Uma ou ambas as colunas especificadas não existem no DataFrame.')


    @staticmethod
    def comparar_dimensionalidade(dataframe, labels):
        """
        Compara as reduções de dimensionalidade usando PCA, LDA e t-SNE.

        :param dataframe: DataFrame a ser visualizado.
        :param labels: Rótulos de classe.
        """
        # Aplica o LabelEncoder se as classes não estiverem codificadas numericamente
        encoded_labels = labels if isinstance(labels[0], (int, np.integer)) else LabelEncoder().fit_transform(labels)

        # Reduz a dimensionalidade usando PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(dataframe)

        # Reduz a dimensionalidade usando LDA
        lda = LDA(n_components=2)
        lda_result = lda.fit_transform(dataframe, encoded_labels)

        # Reduz a dimensionalidade usando t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(dataframe)

        # Cria um DataFrame para armazenar os resultados
        resultados = pd.DataFrame({
            'PCA_1': pca_result[:, 0],
            'PCA_2': pca_result[:, 1],
            'LDA_1': lda_result[:, 0],
            'LDA_2': lda_result[:, 1],
            't-SNE_1': tsne_result[:, 0],
            't-SNE_2': tsne_result[:, 1],
            'Classe': encoded_labels
        })

        # Plota os resultados
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.scatterplot(x='PCA_1', y='PCA_2', hue='Classe', data=resultados, palette='viridis', legend='full')
        plt.title('PCA')

        plt.subplot(1, 3, 2)
        sns.scatterplot(x='LDA_1', y='LDA_2', hue='Classe', data=resultados, palette='viridis', legend='full')
        plt.title('LDA')

        plt.subplot(1, 3, 3)
        sns.scatterplot(x='t-SNE_1', y='t-SNE_2', hue='Classe', data=resultados, palette='viridis', legend='full')
        plt.title('t-SNE')

        plt.tight_layout()
        plt.show()