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
    """Elimina valores atípicos solo de la columna 'AlturaREN'."""
    # Convertir la columna a numérica (forzando errores a NaN)
    df['AlturaREN'] = pd.to_numeric(df['AlturaREN'], errors='coerce')

    # Eliminar filas con valores NaN en 'AlturaREN'
    df = df.dropna(subset=['AlturaREN'])

    # Límites predefinidos para 'AlturaREN'
    min_altura, max_altura = 0, 4500  # 0 a 4500 (AlturaREN)

    # Filtrar los datos eliminando valores fuera del rango válido
    df_filtered = df[(df['AlturaREN'] >= min_altura) & (df['AlturaREN'] <= max_altura)]

    print(f"Valores atípicos eliminados:")
    print(f"- AlturaREN: valores fuera de {min_altura} y {max_altura}")

    return df_filtered

# def impute_social_programs(df):
#     """Imputa valores faltantes en columnas de programas sociales con 0."""
#     # Asegurar que valores vacíos sean tratados como NaN
#     df.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
    
#     # Columnas categóricas específicas
#     social_programs = ['Juntos', 'SIS', 'Qaliwarma']
#     for col in social_programs:
#         if col in df.columns:
#             # Convertir a numérico si es necesario y llenar NaN con 0
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
#     return df


def clean_data(df):
    """Limpia el DataFrame según los criterios específicos para el análisis."""
    
    #### Selección de Data ####
    # Eliminar columnas por criterio de selección
    columns_to_remove = [
        'EESS', 'Renipress', 'Fur', 'red', 'microred', 'tipo_embarazo', 
        'Ubigeo', 'Departamento', 'Provincia', 'Distrito', 'Localidad', 
        'Altitud_Loc', 'FechaHemoglobina', 'atencion_fecha', 
        'fecha_nacimiento', 'UbigeoREN'
    ]
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
    #df = impute_social_programs(df)
    
    print("Datos limpiados y preprocesados exitosamente.")
    return df
