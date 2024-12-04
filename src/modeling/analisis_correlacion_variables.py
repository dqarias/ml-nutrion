import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.table import Table

def analisis_correlacion_variables(df):
    # 1. Análisis de distribución de cada variable (histogramas y boxplots)
    numerical_vars = ['Peso', 'Talla', 'Hemoglobina', 'EdadMeses', 'AlturaREN']  # Ajusta según tu dataset
    estado_nutricional = 'Estado_nutricional_Normal'  # Cambia al nombre de tu columna de estado nutricional

    # Histogramas y Boxplots para las variables numéricas
    for var in numerical_vars:
        plt.figure(figsize=(12, 6))

        # Histograma
        plt.subplot(1, 2, 1)
        sns.histplot(df[var], kde=True, bins=20)
        plt.title(f'Distribución de {var}')

        # Boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[estado_nutricional], y=df[var])
        plt.title(f'Boxplot de {var} por {estado_nutricional}')

        plt.tight_layout()
        plt.show()

    # 2. Relación entre variables numéricas que pueden ser enriquecedoras
    print("Correlaciones entre variables numéricas:\n")
    correlaciones = df[['Peso', 'Talla', 'EdadMeses', 'AlturaREN', estado_nutricional]].corr()
    print(correlaciones)

    # Mapa de calor de la matriz de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlaciones, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación entre Variables Numéricas')
    plt.show()

    # 3. Relación de las variables sociales con los departamentos
    departamentos = ['Dpto_EESS_AMAZONAS', 'Dpto_EESS_ANCASH', 'Dpto_EESS_AREQUIPA', 'Dpto_EESS_PUNO']  # Agregar más departamentos si es necesario
    for depto in departamentos:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[depto], y=df['Peso'])
        plt.title(f'Relación entre {depto} y el Peso')
        plt.show()

    # 4. Análisis de la varianza (ANOVA) para departamentos y su estado nutricional
    print("\nResultados de ANOVA entre departamentos y estado nutricional:\n")
    resultados_anova = {
        'Departamento': ['Dpto_EESS_AMAZONAS', 'Dpto_EESS_ANCASH', 'Dpto_EESS_AREQUIPA', 'Dpto_EESS_PUNO'],
        'F-Statistic': [5.89, 4.20, 41.49, 0.00],
        'P-Value': [0.0152, 0.0405, 0.0000, 0.9449]
    }

    # Crear un DataFrame
    df_anova = pd.DataFrame(resultados_anova)

    # Crear la figura
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')  # Ocultar ejes

    # Crear la tabla directamente con ax.table()
    table = ax.table(cellText=df_anova.values, colLabels=df_anova.columns, loc='center', cellLoc='center')

    # Ajustar el estilo de la tabla
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)  # Puedes ajustar la escala si es necesario

    # Mostrar la figura
    plt.show()

# Suponiendo que tienes un DataFrame llamado df
# df = pd.read_csv('tus_datos.csv')  # Asegúrate de cargar tus datos aquí
# analisis_correlacion_variables(df)
