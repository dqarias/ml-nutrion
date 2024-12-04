import pandas as pd

# Calcular BMI
def calculate_bmi(row):
    if row["Talla"] > 0:
        return row["Peso"] / ((row["Talla"] / 100) ** 2)
    return None

# Determinar estado nutricional
def determine_nutritional_status(bmi, age_months, sex, bmi_boys, bmi_girls):
    bmi_table = bmi_boys if sex == "M" else bmi_girls
    ref = bmi_table[bmi_table["Month"] == age_months]

    if ref.empty:
        return "Sin referencia"
    
    sd3_neg, sd2_neg, sd1_neg, sd0, sd1, sd2, sd3 = (
        ref.iloc[0]["SD3neg"],
        ref.iloc[0]["SD2neg"],
        ref.iloc[0]["SD1neg"],
        ref.iloc[0]["SD0"],
        ref.iloc[0]["SD1"],
        ref.iloc[0]["SD2"],
        ref.iloc[0]["SD3"],
    )
    if bmi < sd3_neg:
        return "Desnutricion severa"
    elif sd3_neg <= bmi < sd2_neg:
        return "Desnutricion moderada"
    elif sd2_neg <= bmi < sd1_neg:
        return "Desnutricion leve"
    elif sd1_neg <= bmi <= sd1:
        return "Normal"
    elif sd1 < bmi <= sd2:
        return "Riesgo de sobrepeso"
    elif sd2 < bmi <= sd3:
        return "Sobrepeso"
    else:
        return "Obesidad"

# Procesar y añadir estado nutricional
def process_nutritional_status(children, bmi_boys, bmi_girls):
    children["BMI"] = children.apply(calculate_bmi, axis=1)
    children["Estado_nutricional"] = children.apply(
        lambda row: determine_nutritional_status(
            row["BMI"], row["EdadMeses"], row["Sexo"], bmi_boys, bmi_girls
        ),
        axis=1,
    )
    return children