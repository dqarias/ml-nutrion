import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample

def plot_classification_report(report, title="Classification Report", cmap="coolwarm"):
    """
    Plots a classification report as a table without borders, with improved spacing and font style.
    
    Args:
        report (dict): The classification report as a dictionary.
        title (str): Title of the plot.
    """
    # Convert the classification report dictionary into a DataFrame
    df = pd.DataFrame(report).transpose()

    # Plot the table
    fig, ax = plt.subplots(figsize=(12, len(df) * 0.7))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table with a cleaner style
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc='center',
        cellLoc='center',
    )
    
    # Styling the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    
    # Remove cell borders
    for cell in table.get_celld().values():
        cell.set_linewidth(0)
    
    # Add title
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.show()

def status_nutrition_predictive(df):
    """
    Analiza datos de estado nutricional y factores predictivos utilizando Random Forest.
    Identifica patrones para varios estados nutricionales y múltiples departamentos.

    Args:
        df (pd.DataFrame): Conjunto de datos con columnas relevantes para la predicción.

    Returns:
        rf (RandomForestClassifier): Modelo entrenado.
        results (dict): Métricas y análisis del modelo.
    """
    # Definir columnas relevantes
    programas_columns = ['Qaliwarma', 'Juntos', 'SIS', 'Cred', 'Suplementacion', 'Consejeria', 'Sesion']
    numeric_columns = ['Peso', 'Talla', 'Hemoglobina', 'EdadMeses']

    # Filtrar columnas de estado nutricional
    estado_nutricional_cols = [col for col in df.columns if col.startswith('Estado_nutricional_')]
    if not estado_nutricional_cols:
        print("No se encontró ninguna columna que comience con 'Estado_nutricional_'")
        return

    print(f"Columnas seleccionadas para 'Estado_nutricional': {estado_nutricional_cols}")

    # Crear columna objetivo combinada
    df['Estado_nutricional'] = df[estado_nutricional_cols].idxmax(axis=1).str.replace('Estado_nutricional_', '')
    le_estado_nutricional = LabelEncoder()
    df['Estado_nutricional_encoded'] = le_estado_nutricional.fit_transform(df['Estado_nutricional'])

    # Filtrar columnas de departamentos
    dpto_eess_cols = [col for col in df.columns if col.startswith('Dpto_EESS_')]
    if not dpto_eess_cols:
        print("No se encontró ninguna columna que comience con 'Dpto_EESS_'")
        return

    print(f"Columnas seleccionadas para 'Dpto_EESS': {dpto_eess_cols}")

    # Crear columna de departamentos codificada
    df['Dpto_EESS'] = df[dpto_eess_cols].idxmax(axis=1).str.replace('Dpto_EESS_', '')
    le_dpto = LabelEncoder()
    df['Dpto_EESS_encoded'] = le_dpto.fit_transform(df['Dpto_EESS'])

    # Preparar características y etiquetas
    X = df[programas_columns + numeric_columns + ['Dpto_EESS_encoded']]
    y = df['Estado_nutricional_encoded']

    # Manejo del desbalance de clases
    df_balanced = pd.concat([X, y], axis=1)
    majority_class = df_balanced[df_balanced['Estado_nutricional_encoded'] == y.mode()[0]]
    minority_classes = df_balanced[df_balanced['Estado_nutricional_encoded'] != y.mode()[0]]

    balanced_classes = [
        resample(
            group,
            replace=True,
            n_samples=len(majority_class),
            random_state=42
        )
        for _, group in minority_classes.groupby('Estado_nutricional_encoded')
    ]
    df_balanced = pd.concat([majority_class] + balanced_classes)
    X = df_balanced.drop(columns=['Estado_nutricional_encoded'])
    y = df_balanced['Estado_nutricional_encoded']

    # Escalar características
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    # Entrenar modelo
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predicciones
    y_pred = rf.predict(X_test)

    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_names = le_estado_nutricional.classes_
    class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print("Classification Report:")
    print(pd.DataFrame(class_report).transpose())  # Mostrar el informe en forma de tabla

    # Visualizar el informe de clasificación con la función personalizada
    plot_classification_report(class_report, title="Nutrition Status Classification Report")

    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Importancia de características
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    # Visualización de importancia de características
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances.values, y=feature_importances.index, palette="viridis")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    return rf, {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report,
        'feature_importances': feature_importances
    }
