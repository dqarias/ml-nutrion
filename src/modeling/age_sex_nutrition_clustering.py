import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def age_sex_nutrition_clustering(df, optimal_clusters=3):
    """
    Realiza clustering para analizar la relación entre edad, sexo y estado nutricional,
    e incluye un análisis de varianza (ANOVA) para validar las diferencias entre los clusters.

    Parámetros:
    - df (DataFrame): Dataset con las columnas 'EdadMeses', 'Sexo_F', 'Sexo_M', 'Estado_nutricional' (o relacionadas).
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

    # Verificar que 'EdadMeses' esté presente
    if 'EdadMeses' not in df.columns:
        raise ValueError("Falta la columna 'EdadMeses' en el DataFrame.")

    # Crear columna 'Sexo' basada en las columnas 'Sexo_F' y 'Sexo_M'
    if 'Sexo_F' in df.columns and 'Sexo_M' in df.columns:
        df['Sexo'] = df['Sexo_M'].apply(lambda x: 'M' if x == 1 else 'F')
    else:
        raise ValueError("Faltan las columnas 'Sexo_F' o 'Sexo_M' en el DataFrame.")

    # Eliminar columnas 'Sexo_F' y 'Sexo_M' si ya no las necesitamos
    df = df.drop(columns=['Sexo_F', 'Sexo_M'])

    # Eliminar valores nulos en las columnas clave (EdadMeses, Sexo, Estado_nutricional)
    df = df.dropna(subset=['EdadMeses', 'Sexo', 'Estado_nutricional'])

    # Seleccionar características para clustering
    clustering_features = df[['EdadMeses', 'Sexo']].copy()

    # Convertir 'Sexo' a variable numérica (codificar Sexo como 0 para 'M' y 1 para 'F', por ejemplo)
    clustering_features['Sexo'] = clustering_features['Sexo'].map({'M': 0, 'F': 1})

    # Escalar las características antes de clustering
    scaler = StandardScaler()
    clustering_features_scaled = scaler.fit_transform(clustering_features)

    # Aplicar KMeans
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(clustering_features_scaled)

    # **Análisis por clusters**
    cluster_analysis = df.groupby('Cluster').agg({
        'EdadMeses': ['mean', 'std', 'min', 'max'],
        'Sexo': lambda x: x.value_counts(normalize=True).to_dict(),
        'Estado_nutricional': lambda x: x.value_counts(normalize=True).to_dict()
    }).reset_index()

    print("\nResumen por cluster:")
    print(cluster_analysis)

    # **ANOVA para validar diferencias en EdadMeses entre los clusters**
    f_stat, p_value = stats.f_oneway(*[df[df['Cluster'] == i]['EdadMeses'] for i in range(optimal_clusters)])

    print("\nResultados de ANOVA para EdadMeses por Cluster:")
    print(f"Estadístico F: {f_stat}")
    print(f"Valor p: {p_value}")

    if p_value < 0.05:
        print("Existen diferencias significativas en la edad entre los clusters.")
    else:
        print("No existen diferencias significativas en la edad entre los clusters.")

    # **Distribución de estado nutricional por sexo**
    estado_nutricion_sex = df.groupby('Sexo')['Estado_nutricional'].value_counts(normalize=True).unstack()
    print("\nDistribución de estado nutricional por sexo:")
    print(estado_nutricion_sex)

    # Visualizaciones
    visualize_results(df, estado_nutricion_sex)

    # Retornar el DataFrame actualizado
    return df

def visualize_results(df, estado_nutricion_sex):
    """
    Genera visualizaciones para explorar los resultados del clustering.
    """
    # Visualización 1: Estado nutricional por sexo
    estado_nutricion_sex.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
    plt.title('Distribución del Estado Nutricional por Sexo')
    plt.ylabel('Porcentaje')
    plt.xlabel('Sexo')
    plt.legend(title='Estado Nutricional', loc='best')
    plt.tight_layout()
    plt.show()

    # Visualización 2: Gráfico de dispersión (EdadMeses vs Sexo)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='EdadMeses', y='Sexo', data=df, hue='Cluster', palette='viridis')
    plt.title('Relación entre Edad y Sexo por Clusters')
    plt.xlabel('Edad en Meses')
    plt.ylabel('Sexo')
    plt.legend(title='Cluster', loc='best')
    plt.tight_layout()
    plt.show()

    # Visualización 3: Pairplot para explorar los clusters
    sns.pairplot(df[['EdadMeses', 'Sexo', 'Estado_nutricional', 'Cluster']], hue='Cluster', palette='viridis')
    plt.tight_layout()
    plt.show()

    # Visualización 4: EdadMeses vs Estado Nutricional
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Estado_nutricional', y='EdadMeses', data=df, palette='viridis')
    plt.title('EdadMeses vs Estado Nutricional')
    plt.xlabel('Estado Nutricional')
    plt.ylabel('Edad en Meses')
    plt.tight_layout()
    plt.show()

    # Visualización 5: Sexo vs Estado Nutricional
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Sexo', hue='Estado_nutricional', data=df, palette='viridis')
    plt.title('Distribución de Estado Nutricional por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Número de Casos')
    plt.tight_layout()
    plt.show()

    # Visualización 6: Edad y Sexo vs Estado Nutricional
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='EdadMeses', y='Sexo', hue='Estado_nutricional', data=df, palette='viridis')
    plt.title('Edad y Sexo vs Estado Nutricional')
    plt.xlabel('Edad en Meses')
    plt.ylabel('Sexo')
    plt.legend(title='Estado Nutricional', loc='best')
    plt.tight_layout()
    plt.show()
