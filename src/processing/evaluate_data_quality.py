import pandas as pd

# Función para eliminar outliers en las columnas 'Peso', 'Talla' y 'AlturaREN'
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


# Función para evaluar la calidad de los datos
def evaluate_data_quality(df):
    # Verificar registros duplicados
    duplicates = df.duplicated().sum()
    print(f"Registros duplicados encontrados: {duplicates}")

    # Evaluar inconsistencias en peso, talla y altura
    df_cleaned = remove_outliers(df)
    print(f"Registros con inconsistencias eliminados: {df.shape[0] - df_cleaned.shape[0]}")

    # Verificar distribución de categorías
    print("Distribución del Estado Nutricional:")
    if 'Estado Nutricional' in df.columns:
        print(df['Estado Nutricional'].value_counts())
    else:
        print("Columna 'Estado Nutricional' no encontrada en el DataFrame.")
    
    print("Distribución del Diagnóstico de Anemia:")
    if 'Dx_anemia' in df.columns:
        print(df['Dx_anemia'].value_counts())
    else:
        print("Columna 'Dx_anemia' no encontrada en el DataFrame.")

    # Identificar registros con datos faltantes y nulos
    missing_data = df.isnull().sum()
    print(f"Datos faltantes por columna: {missing_data}")
    
    # Evaluar consistencia de las edades en relación a peso y talla
    evaluate_age_weight_consistency(df)

    # Verificar sesgos en los datos, como grupos desproporcionados
    evaluate_data_bias(df)

    # Verificación final de la limpieza de datos
    print("Evaluación de calidad de los datos completada.")


# Evaluar la consistencia de las edades en relación al peso y la talla
def evaluate_age_weight_consistency(df):
    # Supongamos que las edades deben ser entre 0 y 18 años
    # Verificar si hay niños con peso o talla fuera de los rangos esperados
    print("Evaluando consistencia de la edad con peso y talla:")
    inconsistent_data = df[(df['EdadMeses'] < 0) | (df['EdadMeses'] > 18)]
    print(f"Niños con edad inconsistente: {inconsistent_data.shape[0]}")
    
    # Evaluar si hay relaciones inconsistentes entre edad, peso y talla
    for index, row in inconsistent_data.iterrows():
        if row['EdadMeses'] <= 5 and (row['Peso'] < 1.5 or row['Talla'] < 60):
            print(f"Inconsistencia encontrada en registro con índice {index}: Niños menores de 5 años con peso o talla anormal.")

    print("Evaluación de consistencia completada.")


# Evaluar sesgos en los datos
def evaluate_data_bias(df):
    # Evaluamos la distribución de diferentes grupos, como mujeres gestantes y niños menores de 5 años
    print("Evaluando sesgos en los datos:")
    print("Distribución de grupos por Estado Nutricional y Edad:")
    print(df.groupby(['Estado Nutricional', 'EdadMeses']).size())
    
    # Supongamos que queremos evaluar si las mujeres gestantes están sobre-representadas
    # y si los niños menores de 5 años están subrepresentados.
    print("Evaluando grupos subrepresentados o sobre representados...")
    pregnant_women = df[df['Estado Nutricional'] == 'Gestante']
    children_under_5 = df[df['EdadMeses'] < 5]
    
    print(f"Mujeres gestantes: {pregnant_women.shape[0]} registros.")
    print(f"Niños menores de 5 años: {children_under_5.shape[0]} registros.")

    print("Evaluación de sesgos completada.")