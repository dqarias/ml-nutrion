import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def anemia_department_analysis(df):
    # Filtrar todas las columnas relacionadas con Dx_anemia y departamentos
    anemia_columns = df.filter(like='Dx_anemia').columns.tolist()
    department_columns = df.filter(like='Dpto_EESS').columns.tolist()

    # Crear la columna objetivo 'Dx_anemia', seleccionando la columna con valor 1
    df['Dx_anemia'] = df[anemia_columns].idxmax(axis=1)

    # Las características serán las columnas de departamentos
    X = df[department_columns]
    y = df['Dx_anemia']

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

    # Visualizar la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.show()

    # Gráfica de importancia de características
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
    plt.title("Importancia de las Características (Departamentos)")
    plt.xlabel("Importancia")
    plt.ylabel("Departamento")
    plt.show()

    # Identificar los departamentos más relacionados con los diagnósticos de anemia
    top_departments = feature_importances.head(5).index.tolist()
    print("\nDepartamentos más relacionados con los diagnósticos de anemia:")
    for i, dept in enumerate(top_departments, 1):
        print(f"{i}. {dept}")

    # Distribución de los diagnósticos por departamentos
    anemia_distribution = df.groupby('Dx_anemia')[department_columns].sum().T
    anemia_distribution.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='coolwarm')
    plt.title("Distribución de Diagnósticos de Anemia por Departamento")
    plt.ylabel("Frecuencia")
    plt.xlabel("Departamento")
    plt.legend(title="Diagnóstico de Anemia")
    plt.tight_layout()
    plt.show()

    return model
