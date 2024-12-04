import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def status_nutrition_analysis(df):
    # Filtrar todas las columnas que contienen 'Estado_nutricional'
    estado_nutricional_columns = df.filter(like='Estado_nutricional').columns.tolist()

    # Verifica si las columnas que estamos seleccionando son correctas
    print("Estado Nutricional Columns:", estado_nutricional_columns)

    # Crear la columna objetivo 'Estado_nutricional', eligiendo la columna que tenga un valor de 1
    df['Estado_nutricional'] = df[estado_nutricional_columns].idxmax(axis=1)

    # Ahora, 'Estado_nutricional' es la etiqueta, y las características serán el resto de las columnas
    X = df.drop(columns=estado_nutricional_columns + ['Estado_nutricional'])
    y = df['Estado_nutricional']

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inicializar y entrenar un modelo de Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Hacer predicciones
    y_pred = model.predict(X_test)

    # Evaluar el modelo
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # Imprimir resultados
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)

    return model
