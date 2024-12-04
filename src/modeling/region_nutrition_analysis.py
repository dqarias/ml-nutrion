import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def geographical_nutrition_clustering(df):
    """
    Realiza clustering geográfico para analizar la relación entre características nutricionales y ubicación geográfica.
    """
    # Verificar si hay columnas geográficas en el dataset
    if 'Region' not in df.columns or 'Provincia' not in df.columns or 'Distrito' not in df.columns:
        print("Faltan columnas geográficas en el dataset")
        return None

    # Filtrar las columnas relevantes para el análisis nutricional y geográfico
    estado_nutricional_columns = df.filter(like='Estado_nutricional').columns.tolist()
    df['Estado_nutricional'] = df[estado_nutricional_columns].idxmax(axis=1)

    # Seleccionar las variables para clustering (nutricionales y geográficas)
    clustering_features = df[['Peso', 'Hemoglobina', 'BMI', 'AlturaREN', 'Region', 'Provincia', 'Distrito']].copy()

    # Codificar las variables geográficas como numéricas (si es necesario)
    clustering_features['Region'] = clustering_features['Region'].astype('category').cat.codes
    clustering_features['Provincia'] = clustering_features['Provincia'].astype('category').cat.codes
    clustering_features['Distrito'] = clustering_features['Distrito'].astype('category').cat.codes

    # Escalar las características nutricionales y geográficas
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clustering_features[['Peso', 'Hemoglobina', 'BMI', 'AlturaREN', 'Region', 'Provincia', 'Distrito']])

    # Elegir el número óptimo de clusters usando el método del codo
    wcss = []
    for i in range(1, 11):  # Probar con 1 a 10 clusters
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

    # Graficar el método del codo
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Método del codo para segmentación geográfica')
    plt.xlabel('Número de clusters')
    plt.ylabel('WCSS')
    plt.show()

    # Elegir el número de clusters basado en el gráfico (por ejemplo, 3 clusters)
    optimal_clusters = 3
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)

    # Agregar los clusters al DataFrame original
    df['Cluster'] = clusters

    ### **2. Resumen estadístico por cluster**
    cluster_analysis = df.groupby('Cluster').agg({
        'Peso': ['mean', 'std'],
        'Hemoglobina': ['mean', 'std'],
        'BMI': ['mean', 'std'],
        'AlturaREN': ['mean', 'std'],
        'Region': lambda x: x.mode().iloc[0],
        'Provincia': lambda x: x.mode().iloc[0],
        'Distrito': lambda x: x.mode().iloc[0],
    }).reset_index()

    print("Resumen por cluster:")
    print(cluster_analysis)

    ### **3. Visualización de resultados geográficos**
    # Mostrar la distribución de los clusters por geografía
    geografía_cluster = df.groupby(['Region', 'Provincia', 'Distrito', 'Cluster']).size().reset_index(name='Count')
    
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Region', hue='Cluster', data=geografía_cluster)
    plt.title('Distribución de Clusters por Región')
    plt.xlabel('Región')
    plt.ylabel('Cantidad de casos')
    plt.legend(title='Cluster')
    plt.show()

    # Si tienes un mapa de las regiones, puedes superponer la segmentación de clusters en el mapa.

    # Retornar el DataFrame con los clusters
    return df
