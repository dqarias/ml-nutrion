import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Funciones auxiliares para la limpieza de datos

def remove_columns(df, columns_to_remove):
    """Elimina las columnas especificadas en columns_to_remove."""
    print(f"Eliminando columnas específicas: {columns_to_remove}")
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
    print(f"Columnas redundantes eliminadas: {redundant_cols}")
    return df.drop(columns=list(redundant_cols))


def remove_high_null_columns(df, threshold=0.2):
    """Elimina columnas con más del 20% de datos nulos."""
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)  # Reemplaza espacios vacíos por NaN
    cols_to_drop = [col for col in df.columns if df[col].isnull().mean() > threshold]
    
    print(f"Columnas eliminadas por alto porcentaje de nulos (> {threshold * 100}%): {cols_to_drop}")
    return df.drop(columns=cols_to_drop)

def remove_outliers(df):
    """Elimina valores atípicos de las columnas 'Peso', 'Talla' y 'AlturaREN'."""
    # Convertir columnas a numéricas
    for col in ['Peso', 'Talla', 'AlturaREN']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Límites predefinidos
    min_peso, max_peso = 0.4, 40  # kg
    min_talla, max_talla = 24, 140  # cm
    min_altura, max_altura = 0, 4500  # msnm

    # Filtrar los datos eliminando valores fuera de los rangos válidos
    filtered_df = df[
        (df['Peso'].between(min_peso, max_peso)) &
        (df['Talla'].between(min_talla, max_talla)) &
        (df['AlturaREN'].between(min_altura, max_altura))
    ]
    
    print(f"Valores atípicos eliminados para 'Peso', 'Talla', 'AlturaREN'.")
    return filtered_df

def impute_social_programs(df):
    """Imputa valores faltantes en columnas de programas sociales con 0."""
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    social_programs = ['Juntos', 'SIS', 'Qaliwarma']
    for col in social_programs:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    print(f"Valores imputados en columnas de programas sociales: {social_programs}")
    return df


# Función principal de limpieza y preprocesamiento
def clean_data(df):
    """Limpia el DataFrame según los criterios específicos para el análisis."""
    # Selección de Data
    columns_to_remove = ['UbigeoPN', 'DepartamentoPN', 'ProvinciaPN', 'DistritoPN', 
                         'CentroPobladoPN', 'UbigeoREN', 'Renipress', 
                         'FechaAtencion', 'FechaHemoglobina', 'FechaNacimiento'] 
    df = remove_columns(df, columns_to_remove)
    
    # Eliminar columnas redundantes y con valores únicos
    df = remove_redundant_columns(df)
    df = remove_unique_value_columns(df)
    
    # Preprocesamiento de Data
    df = remove_high_null_columns(df)
    df = remove_outliers(df)
    df = impute_social_programs(df)
    
    print("Datos limpiados y preprocesados exitosamente.")
    return df


# Preprocesamiento adicional (normalización y codificación)
def preprocess_data(df):
    """Aplica normalización y codificación a los datos."""
    numeric_features = ['Peso', 'Talla', 'AlturaREN', 'Hemoglobina', 'EdadMeses']
    categorical_features = ['Diresa', 'Red', 'Microred', 'Estado_nutricional']

    # Pipelines para datos numéricos y categóricos
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combinación de transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    print("Preprocesamiento aplicado.")
    return preprocessor


# Ejecución completa
def process_pipeline(df):
    """Aplica el flujo completo de limpieza y preprocesamiento."""
    print("Iniciando limpieza de datos...")
    cleaned_df = clean_data(df)
    
    print("Aplicando preprocesamiento adicional...")
    preprocessor = preprocess_data(cleaned_df)
    
    return cleaned_df, preprocessor
