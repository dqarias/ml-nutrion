import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def height_nutrition_clustering(df, optimal_clusters=3):
    """
    Realiza clustering para analizar la relación entre altura geográfica (msnm) y estado nutricional,
    e incluye un análisis de varianza (ANOVA) para validar las diferencias entre los clusters.

    Parámetros:
    - df (DataFrame): Dataset con las columnas 'AlturaREN' y 'Estado_nutricional' (o relacionadas).
    - optimal_clusters (int): Número de clusters a utilizar para el modelo KMeans.

    Retorna:
    - DataFrame: Dataset original con una nueva columna 'Cluster' indicando el cluster asignado.
    """
    # Verificar columnas del DataFrame
    print("Columnas del DataFrame:", df.columns)

    # Filtrar columnas relacionadas con 'Estado_nutricional'
    estado_nutricional_columns = df.filter(like='Estado_nutricional').columns.tolist()

    if not estado_nutricional_columns:
        raise ValueError("No se encontraron columnas relacionadas con 'Estado_nutricional'.")

    # Crear columna combinada 'Estado_nutricional' con el valor más frecuente por fila
    df['Estado_nutricional'] = df[estado_nutricional_columns].idxmax(axis=1)

    # Eliminar valores nulos en las columnas clave
    df = df.dropna(subset=['AlturaREN', 'Estado_nutricional'])

    # Seleccionar características para clustering
    clustering_features = df[['AlturaREN']].copy()
    clustering_features['Estado_nutricional'] = df['Estado_nutricional'].astype('category').cat.codes

    # Escalar las características antes de clustering
    scaler = StandardScaler()
    clustering_features_scaled = scaler.fit_transform(clustering_features)

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(clustering_features_scaled)

    # **Análisis por clusters**
    cluster_analysis = df.groupby('Cluster').agg({
        'AlturaREN': ['mean', 'std', 'min', 'max'],
        'Estado_nutricional': lambda x: x.value_counts(normalize=True).to_dict()
    }).reset_index()

    print("\nResumen por cluster:")
    print(cluster_analysis)

    # **ANOVA para validar diferencias en AlturaREN entre los clusters**
    f_stat, p_value = stats.f_oneway(*[df[df['Cluster'] == i]['AlturaREN'] for i in range(optimal_clusters)])

    print("\nResultados de ANOVA para AlturaREN por Cluster:")
    print(f"Estadístico F: {f_stat}")
    print(f"Valor p: {p_value}")

    if p_value < 0.05:
        print("Existen diferencias significativas en la altura geográfica entre los clusters.")
    else:
        print("No existen diferencias significativas en la altura geográfica entre los clusters.")

    # **Categorizar altura geográfica (msnm)**
    df['Altura_categoria'] = pd.cut(
        df['AlturaREN'], bins=[0, 1500, 3000, 4500], labels=['Baja', 'Media', 'Alta']
    )

    # **Distribución de estado nutricional por categoría de altura**
    altura_nutricion = df.groupby('Altura_categoria')['Estado_nutricional'].value_counts(normalize=True).unstack()
    print("\nDistribución de estado nutricional por categoría de altura:")
    print(altura_nutricion)

    # Visualizaciones
    visualize_results(df, altura_nutricion)

    # Retornar el DataFrame actualizado
    return df

def visualize_results(df, altura_nutricion):
    """
    Genera visualizaciones para explorar los resultados del clustering.
    """
    # Visualización 1: Estado nutricional por categoría de altura
    altura_nutricion.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('Distribución del Estado Nutricional por Categoría de Altura')
    plt.ylabel('Porcentaje')
    plt.xlabel('Altura (msnm)')
    plt.legend(title='Estado Nutricional', loc='best')
    plt.tight_layout()
    plt.show()

    # Visualización 2: Gráfico de dispersión (Altura vs Estado Nutricional)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='AlturaREN', y='Estado_nutricional', data=df, hue='Cluster', palette='viridis')
    plt.title('Relación entre Altura y Estado Nutricional por Clusters')
    plt.xlabel('Altura (msnm)')
    plt.ylabel('Estado Nutricional')
    plt.legend(title='Cluster', loc='best')
    plt.tight_layout()
    plt.show()

    # Visualización 3: Pairplot para explorar los clusters
    sns.pairplot(df[['AlturaREN', 'Estado_nutricional', 'Cluster']], hue='Cluster', palette='viridis')
    plt.tight_layout()
    plt.show()
