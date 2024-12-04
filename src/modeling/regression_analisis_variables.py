import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, f_oneway

def regression_analisis_variables(df, estado_nutricional_columnas):
    """
    Función para calcular la correlación entre la variable 'AlturaREN' y las variables de estado nutricional.
    También realiza un ANOVA para evaluar si hay diferencias significativas entre las categorías del estado nutricional.
    """
    # Verificar que 'AlturaREN' esté en el DataFrame
    if 'AlturaREN' not in df.columns:
        print("Error: La columna 'AlturaREN' no está en el DataFrame.")
        return
    
    # Correlación de Spearman entre AlturaREN y las categorías de estado nutricional
    for estado_nutricional in estado_nutricional_columnas:
        if estado_nutricional not in df.columns:
            print(f"Error: La columna '{estado_nutricional}' no está en el DataFrame.")
            continue

        # Correlación de Spearman (utilizada para variables categóricas)
        corr, _ = spearmanr(df['AlturaREN'], df[estado_nutricional])
        print(f"Correlación de Spearman entre 'AlturaREN' y '{estado_nutricional}': {corr:.3f}")

        # Realizar ANOVA entre las categorías de estado nutricional y la variable AlturaREN
        categorias = df[estado_nutricional].unique()
        grupos = [df[df[estado_nutricional] == categoria]['AlturaREN'] for categoria in categorias]
        f_stat, p_value = f_oneway(*grupos)
        print(f"ANOVA para '{estado_nutricional}': Estadístico F = {f_stat:.2f}, Valor p = {p_value:.4f}")

        if p_value < 0.05:
            print(f"Diferencias significativas en 'AlturaREN' según '{estado_nutricional}'\n")
        else:
            print(f"No hay diferencias significativas en 'AlturaREN' según '{estado_nutricional}'\n")

# # Ejemplo de uso:
# if __name__ == "__main__":
#     # Cargar el DataFrame (por ejemplo, desde un archivo CSV)
#     df = pd.read_csv('datos_nutricionales.csv')

#     # Lista de columnas relacionadas con el estado nutricional
#     estado_nutricional_columnas = [
#         'Estado_nutricional_Desnutricion leve',
#         'Estado_nutricional_Desnutricion moderada',
#         'Estado_nutricional_Desnutricion severa',
#         'Estado_nutricional_Normal',
#         'Estado_nutricional_Obesidad',
#         'Estado_nutricional_Riesgo de sobrepeso',
#         'Estado_nutricional_Sobrepeso'
#     ]

#     # Llamar a la función para calcular la correlación y hacer el análisis de ANOVA
#     correlacion_altura_estado(df, estado_nutricional_columnas)

