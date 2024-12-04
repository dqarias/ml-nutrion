import pandas as pd
import os
from processing.load_data import load_data
from processing.clean_data import clean_data
from processing.save_data import save_data
from processing.combine_csv import combine_csv_files
from processing.transformation_data import transform_data
from processing.nutritional_status import process_nutritional_status  # Importar la función
from processing.exploratory_analysis import exploratory_analysis
from processing.evaluate_data_quality import evaluate_data_quality
from modeling.analyze_missing_by_region import analyze_missing_by_region
from modeling.status_nutrition_analysis import status_nutrition_analysis
from modeling.height_nutrition_analysis import height_nutrition_clustering
from modeling.region_nutrition_analysis import geographical_nutrition_clustering
from modeling.age_sex_nutrition_clustering import age_sex_nutrition_clustering
from modeling.analyze_and_cluster_departments import analyze_and_cluster_departments
from modeling.anemia_department_analysis import anemia_department_analysis
from modeling.nutricional_department_analysis import nutricional_department_analysis
from modeling.estado_nutricional_department_analysis import estado_nutricional_department_analysis
from modeling.estado_nutricional_programas_analysis import estado_nutricional_programas_analysis
from modeling.estado_nutricional_obesidad_sobrepeso_analysis import estado_nutricional_obesidad_sobrepeso_analysis
from modeling.anemia_programas_analysis import anemia_programas_analysis
from modeling.anemia_analysis import anemia_analysis
from modeling.identificar_correlacion import identificar_correlacion
from modeling.analisis_correlacion_variables import analisis_correlacion_variables
from modeling.regression_analisis_variables import regression_analisis_variables
from modeling.status_nutrition_predictive import status_nutrition_predictive
from modeling.status_nutrition_predictive_alg import status_nutrition_predictive_alg
from modeling.status_nutrition_clustering import status_nutrition_clustering


def main():
    # Obtener la ruta del proyecto actual
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Rutas de los archivos de datos (usar rutas relativas al proyecto)
    raw_data_folder = os.path.join(project_root, "data", "raw", "data_children")
    combined_data_path = os.path.join(project_root, "data", "processed", "children_combined.csv")
    processed_clean_data_path = os.path.join(project_root, "data", "processed", "children_clean.csv")
    processed_transform_data_path = os.path.join(project_root, "data", "processed", "children_transform.csv")
    bmi_boys_file = os.path.join(project_root, "data", "external", "bmi_boys.csv")  # Ruta del archivo BMI para niños
    bmi_girls_file = os.path.join(project_root, "data", "external", "bmi_girls.csv")  # Ruta del archivo BMI para niñas

    # # Combinar los archivos CSV
    # print("Combinando archivos CSV...")
    # combine_csv_files(raw_data_folder, combined_data_path)

    # Cargar los datos combinados
    print("\nCargando datos...")
    #df = load_data(combined_data_path)
    #### Priuebas
    df = load_data(processed_clean_data_path)
    #####
    
    if df is not None:
        # print("Primeras filas de los datos:")
        # print(df.head())

        # # # Limpiar datos
        # print("\nLimpiando datos...")
        # cleaned_df = clean_data(df)
        # print("Primeras filas de los datos limpiados:")
        # print(cleaned_df.head())

        # # # Cargar los archivos BMI como DataFrames
        # print("\nCargando archivos de referencia BMI...")
        # bmi_boys = pd.read_csv(bmi_boys_file)
        # bmi_girls = pd.read_csv(bmi_girls_file)
        # print("Primeras filas de bmi_boys:")
        # print(bmi_boys.head())
        # print("Primeras filas de bmi_girls:")
        # print(bmi_girls.head())

        # # # # Calcular estado nutricional
        # print("\nCalculando estado nutricional...")
        # nutritional_df = process_nutritional_status(cleaned_df, bmi_boys, bmi_girls)
        # print("Primeras filas de los datos con estado nutricional:")
        # print(nutritional_df.head())
        
        # # #  # Guardar datos limpios
        # print("\nGuardando datos limpios...")
        # save_data(nutritional_df, processed_clean_data_path)

        # Transformar los datos (aplicar One-Hot Encoding, Label Encoding, y Escalado)
        print("\nTransformando datos...")
        ##### Prueba
        #exploratory_analysis(df)
        #evaluate_data_quality(df)
        #analyze_missing_by_region(df, 'Dpto_EESS')
        transformed_df = transform_data(df)
        #####
        
        #transformed_df = transform_data(cleaned_df)
        print("Primeras filas de los datos transformados:")
        print(transformed_df.head())

        # Guardar datos transformados
        print("\nGuardando datos transformados...")
        save_data(transformed_df, processed_transform_data_path)
        
        # Realizar el análisis de Machine Learning
        print("\nAnalizando datos con Machine Learning...")
        #status_nutrition_analysis(transformed_df)   
        
        #height_nutrition_clustering(transformed_df)
        
        #age_sex_nutrition_clustering(transformed_df)
        
        #analyze_and_cluster_departments(transformed_df)
        
        # geographical_nutrition_clustering(transformed_df)
        
        # anemia_department_analysis(transformed_df)
        # nutricional_department_analysis(transformed_df)
        
        #estado_nutricional_department_analysis(transformed_df)
        #estado_nutricional_programas_analysis(transformed_df)
        #estado_nutricional_obesidad_sobrepeso_analysis(transformed_df)
        #anemia_analysis(transformed_df)
        #anemia_programas_analysis(transformed_df)
        #identificar_correlacion(transformed_df)
        #analisis_correlacion_variables(transformed_df)
        #analisis_correlacion_variables(transformed_df)
        
        
        estado_nutricional_columnas = [
        'Estado_nutricional_Desnutricion leve',
        'Estado_nutricional_Desnutricion moderada',
        'Estado_nutricional_Desnutricion severa',
        'Estado_nutricional_Normal',
        'Estado_nutricional_Obesidad',
        'Estado_nutricional_Riesgo de sobrepeso',
        'Estado_nutricional_Sobrepeso'
    ]
        # Llamar a la función para calcular la correlación y hacer el análisis de ANOVA
        #regression_analisis_variables(df, estado_nutricional_columnas)
        #status_nutrition_predictive(transformed_df)
        #status_nutrition_predictive_alg(transformed_df)
        status_nutrition_clustering(transformed_df)
        

if __name__ == "__main__":
    main()

