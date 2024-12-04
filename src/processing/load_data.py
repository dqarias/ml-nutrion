import pandas as pd

def load_data(file_path):
    """Carga los datos desde un archivo CSV."""
    try:
        df = pd.read_csv(file_path)
        print("Datos cargados exitosamente.")
        return df
    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en la ruta {file_path}")
    except Exception as e:
        print(f"Error al cargar los datos: {e}")