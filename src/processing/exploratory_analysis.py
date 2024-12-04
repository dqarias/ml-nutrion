import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
                plt.show()
            else:
                # Gráficos de barras para variables categóricas
                plt.figure(figsize=(6, 4))
                df[var].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
                plt.title(f"Distribución de {var}")
                plt.show()

    # 2. Revisión de desequilibrio en clases
    print("\nDesequilibrio en clases de 'Estado_nutricional' y 'Dx_anemia':")
    for col in ['Estado_nutricional', 'Dx_anemia']:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts(normalize=True) * 100)

    # 3. Porcentaje de datos faltantes
    print("\nPorcentaje de datos faltantes por columna:")
    
    # Reemplazar cadenas vacías y espacios en blanco por NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    # Calcular los datos faltantes considerando tanto NaN como celdas vacías
    missing_data = df.isnull().sum() / len(df) * 100
    empty_data = (df == '').sum() / len(df) * 100  # Detectar cadenas vacías
    total_missing = missing_data + empty_data  # Sumar ambos tipos de "faltantes"
    
    # Ordenar columnas con datos faltantes y mostrar el resultado
    total_missing_sorted = total_missing[total_missing > 0].sort_values(ascending=False)
    print(total_missing_sorted)
    
    # Visualización de datos faltantes
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Mapa de Datos Faltantes")
    plt.show()

    # 4. Cobertura geográfica
    print("\nCobertura geográfica (conteo de registros por región o microred):")
    for col in ['Diresa', 'Red', 'Microred']:
        if col in df.columns:
            print(f"\n{col}:")
            print(df[col].value_counts())

    print("\nFin del análisis exploratorio.")
