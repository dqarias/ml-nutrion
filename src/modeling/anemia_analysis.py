import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def plot_confusion_matrix(cm, labels, title='Confusion Matrix', cmap='Blues'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_classification_report(report, title="Classification Report", cmap="coolwarm"):
    """
    Plots a classification report as a table without borders, with improved spacing and font style.
    
    Args:
        report (str): The classification report as a string.
        title (str): Title of the plot.
    """
    # Convert the classification report string into a DataFrame
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
    
    # Create DataFrame for plotting
    df = pd.DataFrame(plot_data, columns=["Precision", "Recall", "F1-Score"], index=classes)
    df["Support"] = support
    
    # Plot the table
    fig, ax = plt.subplots(figsize=(12, len(classes) * 0.7))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table with a cleaner style
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        rowLabels=classes,
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


def anemia_analysis(df, train_sizes=[0.1, 0.5, 1.0]):
    # Define columns
    programas_columns = ['Qaliwarma', 'Juntos', 'SIS', 'Cred', 'Suplementacion', 'Consejeria', 'Sesion']
    numeric_columns = ['Peso', 'Talla', 'Hemoglobina', 'EdadMeses']
    
    # Preparar datos
    le_dx_anemia = LabelEncoder()
    df['Dx_anemia_encoded'] = le_dx_anemia.fit_transform(df['Dx_anemia'])  # Changed to Dx_anemia
    X = df[programas_columns + numeric_columns]
    y = df['Dx_anemia_encoded']  # Changed to Dx_anemia_encoded
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    results = {}
    
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
        class_report = classification_report(y_test, y_pred, target_names=le_dx_anemia.classes_)  # Changed to Dx_anemia
        
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
        # plot_confusion_matrix(conf_matrix, labels=le_dx_anemia.classes_, 
        #                       title=f'Matriz de Confusión ({int(size*100)}% Entrenamiento)')
        
        # Graficar classification report
        plot_classification_report(class_report, title=f"Classification Report ({int(size*100)}% Entrenamiento)")
    
    return results
