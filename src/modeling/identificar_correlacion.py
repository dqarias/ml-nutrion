import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Función para mostrar matriz de confusión
def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap='Blues'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# Función para mostrar el reporte de clasificación
def plot_classification_report(report, title="Classification Report", cmap="coolwarm"):
    lines = report.split('\n')
    classes = []
    plot_data = []
    support = []
    
    for line in lines[2:(len(lines) - 5)]:
        row_data = line.split()
        if len(row_data) < 2: 
            continue
        classes.append(" ".join(row_data[:-4]))
        plot_data.append([float(x) for x in row_data[-4:-1]])
        support.append(int(row_data[-1]))
    
    df = pd.DataFrame(plot_data, columns=["Precision", "Recall", "F1-Score"], index=classes)
    df["Support"] = support
    
    fig, ax = plt.subplots(figsize=(12, len(classes) * 0.7))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=classes,
        loc='center',
        cellLoc='center',
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    
    for cell in table.get_celld().values():
        cell.set_linewidth(0)
    
    plt.title(title, fontsize=16, fontweight="bold", pad=20)
    plt.show()

# Función de análisis de estado nutricional con RandomForest
def identificar_correlacion(df, train_sizes=[0.1, 0.5, 1.0]):
    programas_columns = ['Qaliwarma', 'Juntos', 'SIS', 'Cred', 'Suplementacion', 'Consejeria', 'Sesion']
    numeric_columns = ['Peso', 'Talla', 'Hemoglobina', 'EdadMeses']
    
    nutricional_columns = df.filter(like='Estado_nutricional').columns.tolist()

    # Etiquetar el estado nutricional
    le_estado_nutricional = LabelEncoder()
    df['Estado_nutricional_encoded'] = le_estado_nutricional.fit_transform(df[nutricional_columns].idxmax(axis=1))
    X = df[programas_columns + numeric_columns]
    y = df['Estado_nutricional_encoded']
    
    # Escalado de las características numéricas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    results = {}
    feature_importances = None  # Variable para almacenar la importancia de las características

    for size in train_sizes:
        if size == 1.0:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, train_size=size, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=le_estado_nutricional.classes_)
        
        # Guardamos los resultados
        results[size] = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report
        }
        
        print(f"\n--- Evaluación con tamaño de entrenamiento {int(size*100)}% ---")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)
        
        # Graficar matriz de confusión
        plot_confusion_matrix(conf_matrix, labels=le_estado_nutricional.classes_, 
                              title=f'Matriz de Confusión ({int(size*100)}% Entrenamiento)')
        
        # Graficar classification report
        plot_classification_report(class_report, title=f"Classification Report ({int(size*100)}% Entrenamiento)")
        
        # Guardamos la importancia de las características
        feature_importances = rf.feature_importances_

    # Mostrar la importancia de las características
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    # Graficar la importancia de las características
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title("Importancia de las Características")
    plt.tight_layout()
    plt.show()
    
    return results, feature_importance_df
