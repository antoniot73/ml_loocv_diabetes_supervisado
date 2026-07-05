# Práctica: Análisis, implementación y prueba de máquinas de aprendizaje supervisado mediante LOOCV

**Repositorio GitHub:**  
https://github.com/antoniot73/ml_loocv_diabetes_supervisado

**URL Binder:**  
https://mybinder.org/v2/gh/antoniot73/ml_loocv_diabetes_supervisado/main?filepath=notebooks/practica_loocv_diabetes_supervisado.ipynb

**GitHub Page:**  
https://antoniot73.github.io/ml_loocv_diabetes_supervisado/notebooks/practica_loocv_diabetes_supervisado.html

**Dataset utilizado:**  
https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database


---

## Instituto Internacional de Aguascalientes

**Maestría en Inteligencia Artificial para la Transformación Digital**  
**Programa:** Aprendizaje Inteligente  
**Alumno:** Antonio Nicolás Toro González  
**Tutor:** Dr. Francisco Javier Luna Rosas

---

## Descripción

Este repositorio contiene la práctica **“Análisis, implementación y prueba de máquinas de aprendizaje supervisado”**, desarrollada mediante la técnica de validación cruzada **Leave-One-Out Cross-Validation (LOOCV)** sobre el dataset **Pima Indians Diabetes Database**.

Se comparan tres algoritmos de clasificación supervisada:

- **Random Forest**
- **Máquina de Soporte Vectorial (SVM Lineal)**
- **Red Neuronal Multicapa (MLP)**

El objetivo es analizar el comportamiento de cada modelo predictivo en un problema de clasificación binaria asociado a la detección de diabetes, utilizando métricas derivadas de la matriz de confusión.

---

## Objetivo general

Implementar, evaluar y comparar máquinas de aprendizaje supervisado mediante **LOOCV**, utilizando el dataset de diabetes como caso de clasificación binaria.

---

## Objetivos específicos

- Comprender el funcionamiento de LOOCV en modelos supervisados.
- Cargar y validar el dataset de diabetes.
- Implementar Random Forest, SVM Lineal y Red Neuronal MLP.
- Evaluar los modelos mediante métricas de clasificación solicitadas en la práctica.
- Comparar el desempeño de los modelos mediante tablas, matrices de confusión y gráficas.
- Interpretar los resultados considerando el impacto de falsos positivos y falsos negativos.

---

## Dataset

**Nombre:** Pima Indians Diabetes Database  
**Archivo utilizado:** `diabetes.csv`  
**Tipo de problema:** Clasificación binaria  
**Registros:** 768  
**Variables predictoras:** 8  
**Variable objetivo:** `Outcome`

### Variable objetivo

- `Outcome`
  - `0`: paciente sin diabetes
  - `1`: paciente con diabetes

### Variables predictoras utilizadas

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`

---

## Técnica de validación

La evaluación se realizó mediante **Leave-One-Out Cross-Validation (LOOCV)**.

Como el dataset contiene **768 observaciones**, cada modelo fue entrenado **768 veces**. En cada iteración, LOOCV entrena el modelo con **767 registros** y evalúa su predicción sobre **1 registro reservado como prueba**, de modo que cada observación se usa exactamente una vez como dato de prueba.

---

## Tecnologías utilizadas

- Python 3.12+
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- pathlib
- logging

---

## Modelos implementados

### Random Forest

Random Forest es un ensamble de árboles de decisión que combina múltiples clasificadores para mejorar la estabilidad predictiva. En esta práctica se utilizó como modelo robusto para clasificación binaria y para obtener importancia de variables.

Componentes utilizados:

- `RandomForestClassifier`
- `fit()`
- `predict()`
- importancia de variables

---

### SVM Lineal

La Máquina de Soporte Vectorial con frontera lineal busca separar las clases mediante un hiperplano óptimo. Se implementó dentro de un pipeline con escalamiento para evitar que las diferencias de escala entre variables afectaran el ajuste del modelo.

Componentes utilizados:

- `StandardScaler`
- `LinearSVC`
- `Pipeline`
- `fit()`
- `predict()`

---

### Red Neuronal Multicapa (MLP)

La Red Neuronal Multicapa permite modelar relaciones no lineales entre las variables predictoras y la clase objetivo. Se implementó en una configuración compacta para mantener un costo computacional razonable dentro de LOOCV.

Componentes utilizados:

- `StandardScaler`
- `MLPClassifier`
- `Pipeline`
- `fit()`
- `predict()`

---

## Evaluación

Los modelos se comparan mediante las métricas solicitadas en la práctica:

- Precisión Global
- Error Global
- Precisión Positiva (PP)
- Precisión Negativa (PN)
- Falsos Positivos (FP)
- Falsos Negativos (FN)
- Asertividad Positiva (AP)
- Asertividad Negativa (NP)

Además, se generan:

- Matrices de confusión
- Comparación gráfica de métricas
- Importancia de variables para Random Forest

---

## Resultados principales

| Modelo | Precisión Global | Error Global | PP | PN | FP | FN | AP | NP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.7591 | 0.2409 | 0.7836 | 0.7460 | 127 | 58 | 0.6231 | 0.8654 |
| SVM Lineal | 0.7552 | 0.2448 | 0.7201 | 0.7740 | 113 | 75 | 0.6307 | 0.8377 |
| Red Neuronal MLP | 0.7396 | 0.2604 | 0.6231 | 0.8020 | 99 | 101 | 0.6278 | 0.7988 |

### Interpretación general

Bajo la configuración utilizada, **Random Forest** obtuvo el mejor desempeño general, alcanzando la mayor **Precisión Global** y la mayor **Precisión Positiva**. También presentó el menor número de **Falsos Negativos**, lo que resulta relevante en un problema médico porque reduce el riesgo de clasificar como sano a un paciente con diabetes.

---

## Archivos generados

El notebook y el pipeline generan salidas en:

```text
outputs/
├── graficas/
└── tablas/
```

Incluye:

- distribución de clases del dataset
- matrices de confusión para Random Forest, SVM Lineal y MLP
- gráfica comparativa de métricas LOOCV
- importancia de variables Random Forest
- tabla de resumen del dataset
- tabla de métricas LOOCV
- tabla de predicciones LOOCV
- tabla de importancia de variables

---

## Estructura del repositorio

```text
ml_loocv_diabetes_supervisado/
│
├── data/
│   └── diabetes.csv
│
├── notebooks/
│   ├── practica_loocv_diabetes_supervisado.ipynb
│   └── practica_loocv_diabetes_supervisado.html
│
├── outputs/
│   ├── graficas/
│   │   ├── comparacion_metricas_loocv.png
│   │   ├── distribucion_clases_dataset.png
│   │   ├── importancia_variables_random_forest.png
│   │   ├── matriz_confusion_random_forest.png
│   │   ├── matriz_confusion_red_neuronal_mlp.png
│   │   └── matriz_confusion_svm_lineal.png
│   │
│   └── tablas/
│       ├── importancia_variables_random_forest.csv
│       ├── metricas_loocv_modelos.csv
│       ├── predicciones_loocv_cache.csv
│       └── resumen_dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── metrics.py
│   ├── modeling.py
│   ├── reporting.py
│   ├── visualization.py
│   └── main.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Instalación

Crear entorno virtual:

```bash
python -m venv .venv
```

Activar entorno en Windows:

```bash
.venv\Scripts\activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Ejecución local

Ejecutar el pipeline completo:

```bash
python src/main.py
```

Abrir Jupyter:

```bash
jupyter notebook
```

Ejecutar el notebook:

```bash
notebooks/practica_loocv_diabetes_supervisado.ipynb
```

Generar HTML desde el notebook:

```bash
jupyter nbconvert --to html notebooks/practica_loocv_diabetes_supervisado.ipynb --output practica_loocv_diabetes_supervisado.html
```

---

## Ejecución en Binder

Pendiente de configurar cuando el repositorio esté publicado.

---

## Publicación en GitHub Pages

Pendiente de publicar cuando el reporte HTML esté disponible en el repositorio remoto.

---

## Reproducibilidad

La práctica utiliza:

- semilla aleatoria fija
- rutas relativas con `pathlib`
- validación de columnas obligatorias
- revisión de valores faltantes
- pipelines con escalamiento interno
- evaluación LOOCV
- exportación automática de tablas y gráficas
- logging para trazabilidad de la ejecución

Esto permite ejecutar el proyecto de forma reproducible en entorno local y facilita su publicación posterior en GitHub, Binder o GitHub Pages.

---

## Referencias

Raschka, S., & Mirjalili, V. (2019). *Python machine learning* (3rd ed.). Packt Publishing.

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

---

## Autor

Práctica académica desarrollada para la asignatura **Aprendizaje Inteligente**, utilizando un pipeline reproducible basado en programación estructurada, modularización funcional y buenas prácticas de ingeniería de software.
