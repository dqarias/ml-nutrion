import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def transform_data(df):
    # Crear un pipeline para las transformaciones de los datos
    # Definir las columnas categóricas y numéricas
    categorical_columns = ['Sexo', 'Dx_anemia', 'Diresa', 'Red', 'Microred', 'EESS', 'Prov_EESS', 'Dist_EESS', 'Pais']
    numerical_columns = ['EdadMeses', 'Peso', 'Talla', 'Hemoglobina', 'AlturaREN']

    # Transformaciones para columnas numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),  # Imputación de valores faltantes
        ('scaler', StandardScaler())  # Escalado de las columnas numéricas
    ])

    # Transformaciones para columnas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Imputación de valores faltantes con el más frecuente
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Codificación One-Hot para categorías (sin matriz dispersa)
    ])

    # Combinamos las transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns)
        ])

    # Aplicamos las transformaciones al dataframe
    df_transformed = preprocessor.fit_transform(df)

    # Verificar la forma del dataframe transformado
    print(f"Forma del dataframe transformado: {df_transformed.shape}")

    # Convertimos el resultado a un DataFrame
    transformed_df = pd.DataFrame(df_transformed)

    # Obtener los nombres de las columnas después de la codificación One-Hot
    cat_columns = preprocessor.transformers_[1][1].named_steps['encoder'].get_feature_names_out(categorical_columns)

    # Asegurarnos de que el número de columnas coincida con la forma del DataFrame transformado
    all_columns = numerical_columns + list(cat_columns)

    # Verificar que el número de columnas coincida
    print(f"Número de columnas: {len(all_columns)}")
    print(f"Número de columnas en el DataFrame transformado: {df_transformed.shape[1]}")

    # Si el número de columnas es correcto, asignamos los nombres
    if len(all_columns) == df_transformed.shape[1]:
        transformed_df.columns = all_columns
    else:
        raise ValueError("El número de columnas no coincide con los nombres de las columnas esperados")

    # Devolver el dataframe transformado
    return transformed_df
