import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

def anemia_programas_analysis(df):
    """
    Analiza la relación entre los programas sociales y los diagnósticos de anemia
    utilizando un modelo de Random Forest.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de entrada. Debe incluir columnas 
        relacionadas con diagnóstico de anemia y programas sociales.

    Retorna:
        RandomForestClassifier: Modelo entrenado.
    """
    # Filtrar columnas relacionadas con diagnóstico de anemia y programas sociales
    anemia_columns = df.filter(like='Dx_anemia').columns.tolist()
    programas_columns = ['Qaliwarma', 'Juntos', 'SIS', 'Cred', 'Suplementacion', 'Consejeria', 'Sesion']

    # Crear la columna objetivo 'Dx_anemia' seleccionando la columna con valor 1
    df['Dx_anemia'] = df[anemia_columns].idxmax(axis=1)

    # Filtrar registros con algún diagnóstico de anemia
    df_anemia = df[df['Dx_anemia'].isin(['Dx_anemia_Anemia Leve', 
                                         'Dx_anemia_Anemia Moderada', 
                                         'Dx_anemia_Anemia Severa'])]

    # Preparar las características (X) y el objetivo (y)
    X = df_anemia[programas_columns]
    y = df_anemia['Dx_anemia']

    # Manejo de valores nulos
    X.fillna(0, inplace=True)

    # Codificar etiquetas del objetivo
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inicializar y entrenar el modelo
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

    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # Gráfica de importancia de características
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
    plt.title("Importancia de las Características (Programas Sociales)")
    plt.xlabel("Importancia")
    plt.ylabel("Programa")
    plt.show()

    # Identificar los programas sociales más relacionados con los diagnósticos de anemia
    top_programas = feature_importances.head(5).index.tolist()
    print("\nProgramas más relacionados con los diagnósticos de anemia:")
    for i, programa in enumerate(top_programas, 1):
        print(f"{i}. {programa}")

    # Distribución de anemia por programas sociales
    anemia_distribution = df_anemia.groupby('Dx_anemia')[programas_columns].sum().T
    anemia_distribution.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='coolwarm')
    plt.title("Distribución de Anemia por Programas Sociales")
    plt.ylabel("Frecuencia")
    plt.xlabel("Programa")
    plt.legend(title="Diagnóstico de Anemia")
    plt.tight_layout()
    plt.show()

    return model