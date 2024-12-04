import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Función para el análisis exploratorio inicial
def exploratory_analysis(df):
    print("=== Análisis Exploratorio Inicial ===\n")
    
    # 1. Análisis de distribución de variables clave
    print("\nDistribución de variables clave:")
    key_vars = ['Peso', 'Talla', 'EdadMeses', 'Sexo', 'BMI', 'Estado_nutricional', 'Dx_anemia']
    
    for var in key_vars:
        if var in df.columns:
            if df[var].dtype in ['float64', 'int64']:
                # Histogramas para variables numéricas
                plt.figure(figsize=(6, 4))
                sns.histplot(df[var], kde=True, bins=30)
                plt.title(f"Distribución de {var}")
                plt.xlabel(var)
                plt.ylabel("Frecuencia")
                plt.show()
            else:
                # Gráficos de barras para variables categóricas
                plt.figure(figsize=(6, 4))
                df[var].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
                plt.title(f"Distribución de {var}")
                plt.xlabel(var)
                plt.ylabel("Frecuencia")
                plt.show()

    # 2. Revisión de desequilibrio en clases
    print("\nDesequilibrio en clases de 'Estado_nutricional' y 'Dx_anemia':")
    for col in ['Estado_nutricional', 'Dx_anemia']:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(normalize=True) * 100)

    # 3. Porcentaje de datos faltantes
    print("\nPorcentaje de datos faltantes por columna:")
    
    # Verificación explícita de valores nulos
    missing_data = df.isnull().sum() / len(df) * 100  # Calculamos el porcentaje de nulos
    missing_data_sorted = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if missing_data_sorted.empty:
        print("No se encontraron datos faltantes en las columnas.")
    else:
        print(missing_data_sorted)
    
    # Visualización de datos faltantes
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Mapa de Datos Faltantes")
    plt.show()

    # 4. Cobertura geográfica
    print("\nCobertura geográfica (conteo de registros por región o microred):")
    for col in ['Diresa', 'Red', 'Microred']:
        if col in df.columns:
            print(f"\nDistribución de registros en {col}:")
            print(df[col].value_counts())
    
    print("\nAnálisis exploratorio completado.")

# Uso de la función con el DataFrame
# Reemplaza 'tu_archivo.csv' con la ruta al archivo que deseas cargar
df = pd.read_csv("tu_archivo.csv")

# Llamada a la función de análisis
exploratory_analysis(df)
