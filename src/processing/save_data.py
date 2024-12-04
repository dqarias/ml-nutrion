def save_data(df, file_path):
    """Guarda el DataFrame limpio en un archivo CSV."""
    try:
        df.to_csv(file_path, index=False)
        print("Datos guardados exitosamente.")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")