import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def transform_data(df):
    """
    Transforma un DataFrame aplicando imputación, escalado y codificación categórica.
    Mantiene las columnas binarias y otras específicas sin transformar.
    """
    # Definir las columnas categóricas, numéricas, y las columnas a mantener sin cambios
    categorical_columns = ['Sexo', 'Dpto_EESS', 'Dx_anemia', 'Estado_nutricional']
    numerical_columns = ['Peso', 'Talla', 'Hemoglobina']
    altura_column = ['AlturaREN']
    edad_column = ['EdadMeses']
    unchanged_columns = ['Qaliwarma', 'Juntos', 'SIS', 'Cred', 'Suplementacion', 'Consejeria', 'Sesion']  # Mantener estas columnas sin cambios

    # Transformaciones para columnas numéricas (excepto las excluidas)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Imputación de valores faltantes
        ('scaler', MinMaxScaler())  # Normalización entre 0 y 1
    ])

    # Transformaciones para la columna 'AlturaREN' (mantener en rango original)
    altura_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))  # Solo imputación
    ])

    # Transformaciones para la columna 'EdadMeses' (mantener sin cambios)
    edad_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean'))  # Solo imputación
    ])

    # Transformaciones para columnas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación con el valor más frecuente
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Codificación One-Hot
    ])

    # Combinamos las transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('edad', edad_transformer, edad_column),  # 'EdadMeses'
            ('altura', altura_transformer, altura_column),  # 'AlturaREN'
            ('cat', categorical_transformer, categorical_columns),
            ('unchanged', 'passthrough', unchanged_columns)  # Pasar sin cambios
        ])

    # Aplicar las transformaciones al DataFrame
    df_transformed = preprocessor.fit_transform(df)

    # Obtener nombres de las columnas transformadas
    numerical_columns_normalized = numerical_columns
    edad_columns = edad_column
    altura_columns = altura_column
    unchanged_columns_kept = unchanged_columns
    categorical_columns_encoded = preprocessor.transformers_[3][1].named_steps['encoder'].get_feature_names_out(categorical_columns)

    # Combinar todos los nombres de las columnas
    all_columns = numerical_columns_normalized + edad_columns + altura_columns + list(categorical_columns_encoded) + unchanged_columns_kept

    # Convertir el resultado a un DataFrame con los nombres de las columnas
    transformed_df = pd.DataFrame(df_transformed, columns=all_columns)

    # Verificar la transformación
    print(f"Forma del DataFrame transformado: {transformed_df.shape}")
    return transformed_df

# Ejemplo de uso
# df = pd.read_csv("tu_archivo.csv")  # Carga tu DataFrame
# df_transformed = transform_data(df)
# print(df_transformed.head())


