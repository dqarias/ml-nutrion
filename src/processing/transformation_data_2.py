import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def transform_data_2(df):
    """
    Transforma un DataFrame aplicando imputación, escalado y codificación categórica.
    Mantiene las columnas binarias y otras específicas sin transformar.
    """
    # Identificar columnas originales de estado nutricional
    original_estado_nutricional_cols = [col for col in df.columns if col.startswith('Estado_nutricional_')]
    
    # Columnas a transformar
    categorical_columns = [col for col in df.columns if col.startswith(('Dpto_EESS_', 'Sexo_'))]
    numerical_columns = ['Peso', 'Talla', 'Hemoglobina', 'AlturaREN', 'EdadMeses', 'BMI']
    unchanged_columns = ['Qaliwarma', 'Juntos', 'SIS', 'Cred', 'Suplementacion', 'Consejeria', 'Sesion']
    
    # Transformaciones
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', MinMaxScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_columns),
            ('cat', categorical_transformer, categorical_columns),
            ('unchanged', 'passthrough', unchanged_columns)
        ])
    
    try:
        # Preparar datos para transformación
        X = df[numerical_columns + categorical_columns + unchanged_columns]
        
        # Transformar
        df_transformed = preprocessor.fit_transform(X)
        
        # Nombres de columnas
        num_cols = numerical_columns
        cat_cols = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_columns)
        
        # Combinar columnas
        final_columns = num_cols + list(cat_cols) + unchanged_columns
        
        # Crear DataFrame transformado
        transformed_df = pd.DataFrame(
            df_transformed, 
            columns=final_columns, 
            index=df.index
        )
        
        # Añadir de vuelta las columnas de estado nutricional originales
        for col in original_estado_nutricional_cols:
            transformed_df[col] = df[col]
        
        print(f"Forma del DataFrame transformado: {transformed_df.shape}")
        print("\nColumnas transformadas:")
        print(transformed_df.columns.tolist())
        
        return transformed_df
    
    except Exception as e:
        print(f"Error en la transformación: {e}")
        return df