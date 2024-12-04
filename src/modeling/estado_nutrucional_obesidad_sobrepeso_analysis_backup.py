import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def estado_nutricional_obesidad_sobrepeso_analysis(df):
    # Define columns
    programas_columns = ['Qaliwarma', 'Juntos', 'SIS', 'Cred', 'Suplementacion', 'Consejeria', 'Sesion']
    numeric_columns = ['Peso', 'Talla', 'Hemoglobina', 'EdadMeses']
    
    # Filtrar las columnas que comienzan con 'Estado_nutricional_'
    estado_nutricional_cols = [col for col in df.columns if col.startswith('Estado_nutricional_')]
    
    if estado_nutricional_cols:
        # Usamos la primera columna que coincide con 'Estado_nutricional_'
        estado_nutricional_col = estado_nutricional_cols[0]
        print(f"Columna seleccionada para 'Estado_nutricional': {estado_nutricional_col}")
        
        # Codificar la columna seleccionada
        le_estado_nutricional = LabelEncoder()
        df['Estado_nutricional_encoded'] = le_estado_nutricional.fit_transform(df[estado_nutricional_col])
    else:
        print("No se encontró ninguna columna que comience con 'Estado_nutricional_'")
        return
    
    # Codificar otras variables categóricas
    le_dpto = LabelEncoder()
    df['Dpto_EESS_encoded'] = le_dpto.fit_transform(df['Dpto_EESS'])
    
    # Preparar características
    X = df[programas_columns + numeric_columns + ['Dpto_EESS_encoded']]
    y = df['Estado_nutricional_encoded']
    
    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Modelo Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Predicciones y métricas
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Imprimir métricas
    print("Analizando datos con Machine Learning...")
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    # Generar informe de clasificación
    class_report_str = classification_report(
        y_test, 
        y_pred, 
        target_names=le_estado_nutricional.classes_
    )
    print("Classification Report:")
    print(class_report_str)
    
    # Importancia de características
    feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    # Visualización de importancia de características
    plt.figure(figsize=(10, 6))
    feature_import_df = pd.DataFrame({
        'Feature': feature_importances.index,
        'Importance': feature_importances.values
    }).sort_values('Importance', ascending=True)
    
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feature_import_df, 
        palette="viridis"
    )
    plt.title("Feature Importance in Nutritional Status Prediction")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
    
    return rf, {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': class_report_str,
        'feature_importances': feature_importances
    }
