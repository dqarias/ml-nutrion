import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def status_nutrition_clustering(df, n_clusters=4):
    # Seleccionar las columnas relevantes para el clustering
    nutrition_columns = [
        'Estado_nutricional_Desnutricion leve', 'Estado_nutricional_Desnutricion moderada', 
        'Estado_nutricional_Desnutricion severa', 'Estado_nutricional_Normal', 
        'Estado_nutricional_Obesidad', 'Estado_nutricional_Riesgo de sobrepeso', 
        'Estado_nutricional_Sobrepeso'
    ]
    
    # Asegúrate de que las columnas necesarias existan en el DataFrame
    if not all(col in df.columns for col in nutrition_columns):
        raise ValueError("El DataFrame no contiene todas las columnas necesarias para el clustering.")
    
    # Extraer los datos de nutrición
    nutrition_data = df[nutrition_columns]
    
    # Verifica que los datos no estén vacíos
    if nutrition_data.empty:
        raise ValueError("Los datos de nutrición están vacíos.")
    
    # Normalizar los datos para evitar que las variables con valores más grandes dominen el clustering
    scaler = StandardScaler()
    nutrition_data_scaled = scaler.fit_transform(nutrition_data)
    
    # Verifica los datos normalizados
    print("Datos normalizados (primeras 5 filas):")
    print(nutrition_data_scaled[:5])
    
    # Aplicar el algoritmo de KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(nutrition_data_scaled)
    
    # Verifica los resultados
    print("Distribución de clusters:")
    print(df['Cluster'].value_counts())
    
    # Verifica los centros de los clusters
    print("Centros de los clusters:")
    print(kmeans.cluster_centers_)
    
    # Realizar PCA para reducir a 2 dimensiones
    pca = PCA(n_components=2)
    nutrition_data_pca = pca.fit_transform(nutrition_data_scaled)

    # Imprimir las coordenadas de los puntos en el espacio reducido
    print("\nCoordenadas de los puntos en el espacio PCA (primeras 5 filas):")
    print(nutrition_data_pca[:5])
    
    # Imprimir los clusters asignados y sus coordenadas en PCA
    print("\nClusters asignados y sus coordenadas en el espacio PCA:")
    pca_df = pd.DataFrame(nutrition_data_pca, columns=['Componente_Principal_1', 'Componente_Principal_2'])
    pca_df['Cluster'] = df['Cluster']
    print(pca_df.head())
    
    # Graficar los clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(nutrition_data_pca[:, 0], nutrition_data_pca[:, 1], c=df['Cluster'], cmap='viridis', s=10)
    plt.title('Clustering de Estado Nutricional')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.colorbar(label='Cluster')
    plt.show()

    return df

# Ejemplo de uso
# Suponiendo que 'data' es el DataFrame que contiene tus datos
# data = pd.read_csv('tu_archivo.csv')
# df_clustered = status_nutrition_clustering(data, n_clusters=4)
# print(df_clustered[['Cluster'] + nutrition_columns].head())
