import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_missing_by_region(df, region_col):
    print(f"=== Análisis de datos faltantes por región ({region_col}) ===\n")
    
    # Reemplazar cadenas vacías y espacios en blanco por NaN
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    
    # Calcular porcentaje de valores nulos por región y columna
    missing_by_region = df.groupby(region_col).apply(
        lambda group: group.isnull().mean() * 100
    )
    
    # Mostrar los resultados
    print(missing_by_region)
    
    # Visualizar los datos faltantes por región
    plt.figure(figsize=(12, 8))
    sns.heatmap(missing_by_region.T, cmap="YlGnBu", annot=True, fmt=".1f", cbar=True)
    plt.title(f"Porcentaje de Datos Faltantes por Región ({region_col})")
    plt.xlabel("Región")
    plt.ylabel("Columnas")
    plt.show()
    
    return missing_by_region