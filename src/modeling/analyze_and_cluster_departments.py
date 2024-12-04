import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

def analyze_and_cluster_departments(df, optimal_clusters=3):
    """
    Analiza la relación entre departamentos y estados nutricionales,
    y realiza clustering directamente sobre los datos originales.
    
    Parámetros:
    - df (DataFrame): Dataset con columnas 'Dpto_*' y 'Estado_nutricional_*'.
    - optimal_clusters (int): Número de clusters para el análisis de agrupamiento.

    Genera visualizaciones para el análisis directo y el clustering.
    """
    # Identificar columnas de departamentos y estados nutricionales
    dpto_columns = df.filter(like='Dpto_EESS_').columns.tolist()
    estado_nutricional_columns = df.filter(like='Estado_nutricional_').columns.tolist()
    
    # Crear columna combinada para el estado nutricional con el valor más frecuente por fila
    df['Estado_nutricional'] = df[estado_nutricional_columns].idxmax(axis=1).str.replace('Estado_nutricional_', '')
    
    # Crear una columna para el departamento basado en el que tiene valor 1
    df['Departamento'] = df[dpto_columns].idxmax(axis=1).str.replace('Dpto_EESS_', '')
    
    # Tabular la relación entre Departamento y Estado Nutricional
    department_nutrition = pd.crosstab(df['Departamento'], df['Estado_nutricional'], normalize='index') * 100
    print("\nDistribución porcentual de estado nutricional por departamento:")
    print(department_nutrition)
    
    # Análisis de correlación
    print("\nAnálisis de correlación entre departamentos y estados nutricionales:")
    correlation_matrix = department_nutrition.corr()
    print(correlation_matrix)
    
    # Visualización de la matriz de correlación
    figsize = (14, 10)  # Verifica que el tamaño sea válido
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Correlación'}, vmin=-1, vmax=1)
    plt.title('Matriz de Correlación entre Departamentos y Estado Nutricional')
    plt.tight_layout()
    plt.show()
    
    # Visualización: Heatmap de la distribución de Estado Nutricional por Departamento
    plt.figure(figsize=figsize)
    sns.heatmap(department_nutrition, annot=True, fmt=".1f", cmap="viridis", cbar_kws={'label': 'Porcentaje (%)'})
    plt.title('Distribución de Estado Nutricional por Departamento')
    plt.ylabel('Departamento')
    plt.xlabel('Estado Nutricional')
    plt.tight_layout()
    plt.show()
    
    # Aplicar KMeans directamente a los datos
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    department_nutrition['Cluster'] = kmeans.fit_predict(department_nutrition)
    
    print("\nDistribución por cluster de departamentos:")
    print(department_nutrition)
    
    # Visualización de Clustering (usando un gráfico de dispersión 2D para los dos primeros estados nutricionales)
    plt.figure(figsize=figsize)
    sns.scatterplot(
        x=department_nutrition.iloc[:, 0],  # Primer estado nutricional
        y=department_nutrition.iloc[:, 1],  # Segundo estado nutricional
        hue=department_nutrition['Cluster'],
        palette="viridis",
        s=100
    )
    plt.title('Clusters de Departamentos por Estado Nutricional (2D)')
    plt.xlabel(department_nutrition.columns[0])  # Etiqueta para el eje x
    plt.ylabel(department_nutrition.columns[1])  # Etiqueta para el eje y
    plt.legend(title='Cluster', loc='best')
    plt.tight_layout()
    plt.show()

    # Si quieres visualizar el clustering en 3D (utilizando 3 dimensiones de estado nutricional)
    if department_nutrition.shape[1] >= 3:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(
            department_nutrition.iloc[:, 0], 
            department_nutrition.iloc[:, 1], 
            department_nutrition.iloc[:, 2], 
            c=department_nutrition['Cluster'], 
            cmap="viridis", s=100
        )
        ax.set_xlabel(department_nutrition.columns[0])
        ax.set_ylabel(department_nutrition.columns[1])
        ax.set_zlabel(department_nutrition.columns[2])
        plt.title('Clusters de Departamentos por Estado Nutricional (3D)')
        plt.show()

# Uso
# Asegúrate de cargar el DataFrame antes de llamar a esta función, por ejemplo:
# df = pd.read_csv('tu_archivo.csv')
# analyze_and_cluster_departments(df, optimal_clusters=4)

