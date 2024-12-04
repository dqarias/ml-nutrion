import pandas as pd

#### Selección de Data

def remove_columns(df, columns_to_remove):
    """Elimina las columnas especificadas en columns_to_remove."""
    return df.drop(columns=columns_to_remove, errors='ignore')

def remove_unique_value_columns(df):
    """Elimina columnas del DataFrame que contienen un único valor."""
    cols_to_remove = [col for col in df.columns if df[col].nunique() == 1]
    print(f"Columnas eliminadas por contener valores únicos: {cols_to_remove}")
    return df.drop(columns=cols_to_remove)

def remove_redundant_columns(df):
    """Elimina una de las columnas redundantes con los mismos datos."""
    redundant_cols = set()
    columns = list(df.columns)

    for i, col1 in enumerate(columns):
        if col1 in redundant_cols:
            continue
        for col2 in columns[i + 1:]:
            if col2 in redundant_cols:
                continue
            if df[col1].equals(df[col2]):
                redundant_cols.add(col2)  # Solo elimina col2 para cada duplicado
    return df.drop(columns=list(redundant_cols))


#### Preprocesamiento de Data

def remove_high_null_columns(df, threshold=0.2):
    """Elimina columnas con más del 20% de datos nulos."""
    # Reemplazar celdas vacías (espacios en blanco o cadenas vacías) por NaN
    df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    
    # Identificar columnas con un porcentaje de nulos mayor al umbral
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > threshold]
    
    # Imprimir para depuración
    for col in cols_to_drop:
        null_percentage = df[col].isnull().mean()
        print(f"Eliminando columna: {col} - Porcentaje de nulos: {null_percentage:.2%}")
    
    # Eliminar las columnas
    return df.drop(columns=cols_to_drop)

def remove_outliers(df):
    """ Elimina valores atípicos de las columnas 'Peso', 'Talla' y 'AlturaREN' """
    # Convertir columnas a numéricas (forzando errores a NaN)
    df['Peso'] = pd.to_numeric(df['Peso'], errors='coerce')
    df['Talla'] = pd.to_numeric(df['Talla'], errors='coerce')
    df['AlturaREN'] = pd.to_numeric(df['AlturaREN'], errors='coerce')  # Asegurarse de que AlturaREN sea numérico

    # Eliminar filas con valores NaN
    df = df.dropna(subset=['Peso', 'Talla', 'AlturaREN'])

    # Límites predefinidos
    min_peso, max_peso = 0.4, 40  # 400 gramos a 40 kg
    min_talla, max_talla = 24, 140  # 24 cm a 140 cm
    min_altura, max_altura = 0, 4500  # 0 a 4500 (AlturaREN)

    # Calcular los percentiles (IQR) para peso y talla
    Q1_peso, Q3_peso = df['Peso'].quantile(0.25), df['Peso'].quantile(0.75)
    IQR_peso = Q3_peso - Q1_peso
    lower_bound_peso = max(Q1_peso - 1.5 * IQR_peso, min_peso)
    upper_bound_peso = min(Q3_peso + 1.5 * IQR_peso, max_peso)

    Q1_talla, Q3_talla = df['Talla'].quantile(0.25), df['Talla'].quantile(0.75)
    IQR_talla = Q3_talla - Q1_talla
    lower_bound_talla = max(Q1_talla - 1.5 * IQR_talla, min_talla)
    upper_bound_talla = min(Q3_talla + 1.5 * IQR_talla, max_talla)

    # Filtrar los datos eliminando valores fuera de los rangos válidos
    df_filtered = df[
        (df['Peso'] >= lower_bound_peso) & (df['Peso'] <= upper_bound_peso) &
        (df['Talla'] >= lower_bound_talla) & (df['Talla'] <= upper_bound_talla) &
        (df['AlturaREN'] >= min_altura) & (df['AlturaREN'] <= max_altura)
    ]

    print(f"Valores atípicos eliminados:")
    print(f"- Peso: valores fuera de {lower_bound_peso:.2f} y {upper_bound_peso:.2f}")
    print(f"- Talla: valores fuera de {lower_bound_talla:.2f} y {upper_bound_talla:.2f}")
    print(f"- AlturaREN: valores fuera de {min_altura} y {max_altura}")

    return df_filtered

def impute_social_programs(df):
    """Imputa valores faltantes en columnas de programas sociales con 0."""
    # Asegurar que valores vacíos sean tratados como NaN
    df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    
    # Columnas categóricas específicas
    social_programs = ['Juntos', 'SIS', 'Qaliwarma']
    for col in social_programs:
        if col in df.columns:
            # Convertir a numérico si es necesario y llenar NaN con 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def clean_data(df):
    """Limpia el DataFrame según los criterios específicos para el análisis."""
    
    #### Selección de Data ####
    # Eliminar columnas por criterio de selección
    columns_to_remove = ['UbigeoPN', 'DepartamentoPN', 'ProvinciaPN', 'DistritoPN', 'CentroPobladoPN', 'UbigeoREN', 'Renipress', 'FechaAtencion', 'FechaHemoglobina', 'FechaNacimiento'] 
    df = remove_columns(df, columns_to_remove)
    
    # Eliminar columnas con valores únicos
    df = remove_unique_value_columns(df)
    
    # Eliminar columnas con valores redundantes
    df = remove_redundant_columns(df)
    
    #### Preprocesamiento de Data ####
    
    # Eliminar columnas con el 20% de nulos
    df = remove_high_null_columns(df)
    
    # Eliminar outliers, valores atípicos de peso, talla y altura
    df = remove_outliers(df)
    
    # Imputar valores 0 en programas sociales
    df = impute_social_programs(df)
    
    print("Datos limpiados y preprocesados exitosamente.")
    return df
