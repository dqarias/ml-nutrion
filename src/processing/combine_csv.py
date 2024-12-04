import os
import glob
import pandas as pd

def combine_csv_files(source_folder, output_file):
    """Combina todos los archivos CSV en la carpeta especificada en un solo archivo CSV."""
    # Obtener la lista de archivos CSV en la carpeta
    csv_files = glob.glob(os.path.join(source_folder, "*.csv"))
    
    # Verificar qué archivos se encontraron
    print(f"Archivos CSV encontrados: {csv_files}")
    
    if not csv_files:
        print("No se encontraron archivos CSV en la carpeta especificada.")
        return
    
    # Lista para almacenar los DataFrames
    dfs = []
    
    # Leer cada archivo CSV y agregarlo a la lista
    for file in csv_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # Concatenar todos los DataFrames en uno solo
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Guardar el DataFrame combinado en un archivo CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Archivos CSV combinados guardados en: {output_file}")
