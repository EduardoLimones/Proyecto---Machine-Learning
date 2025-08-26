# Modelo Predictivo de Salarios en el Sector de Datos para Europa

Este proyecto tiene como objetivo analizar los factores clave que influyen en los salarios del sector de datos y construir un modelo de machine learning capaz de predecir una remuneración competitiva para profesionales en el mercado europeo.

## 📜 Descripción del Proyecto

A través de un análisis exhaustivo de un dataset global de ofertas de empleo, este proyecto sigue un flujo de trabajo completo de ciencia de datos: desde la adquisición y limpieza de los datos, pasando por un profundo análisis exploratorio (EDA) y una robusta ingeniería de características, hasta el entrenamiento y la evaluación de múltiples modelos de regresión para encontrar el predictor más preciso.

El resultado final es un modelo de `RandomForestRegressor` optimizado que explica aproximadamente el **87.7%** de la varianza salarial y una aplicación interactiva desarrollada con Streamlit para realizar predicciones en tiempo real.

## 📋 Tabla de Contenidos

1.  [Estructura del Repositorio](#-estructura-del-repositorio)
2.  [Fuente de Datos](#-fuente-de-datos)
3.  [Metodología](#-metodología)
    - [Limpieza y Análisis Exploratorio (EDA)](#1-limpieza-y-análisis-exploratorio-eda)
    - [Ingeniería de Características (Feature Engineering)](#2-ingeniería-de-características-feature-engineering)
    - [Entrenamiento y Evaluación del Modelo](#3-entrenamiento-y-evaluación-del-modelo)
4.  [Resultados](#-resultados)
5.  [Tecnologías Utilizadas](#-tecnologías-utilizadas)
6.  [Cómo Ejecutar el Proyecto](#-cómo-ejecutar-el-proyecto)
7.  [Autor](#-autor)

## 📁 Estructura del Repositorio

El repositorio está organizado de la siguiente manera para facilitar la reproducibilidad y comprensión del proyecto:
```
.
├── app_streamlit/
│   ├── demo_modelo.py
│   ├── mi_modelo_salarios.pkl
│   └── salarios_codificado.csv
│
├── data/
│   ├── procesed/
│   │   └── salarios_codificado.csv
│   ├── raw/
│   │   └── ai_job_dataset.csv
│   ├── test/
│   │   └── test.csv
│   └── train/
│       └── train.csv
│
├── docs/
│   └── Presentación.pdf
│
├── graficos/
│   ├── feature_importance.png
│   └── ... 
│
├── models/
│   └── mi_modelo_salarios.pkl
│
├── notebooks/
│   ├── 01_Fuentes.ipynb
│   ├── 02_LimpiezaEDA.ipynb
│   └── 03_Entrenamiento_Evaluacion.ipynb
│
└── README.md
´´´
## 📊 Fuente de Datos

Los datos para este análisis fueron obtenidos de la plataforma Kaggle.

-   **Nombre del Dataset:** Global AI Job Market & Salary Trends 2025
-   **Enlace:** [Kaggle Dataset](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025)
-   **Descripción:** El conjunto de datos contiene información sobre salarios reportados en diversos roles dentro del campo de la ciencia de datos a nivel mundial.

## 🛠️ Metodología

El proyecto se dividió en tres fases principales, cada una documentada en su respectivo notebook.

### 1. Limpieza y Análisis Exploratorio (EDA)

En esta fase, se realizó un análisis profundo para entender las distribuciones y relaciones en los datos. Los hallazgos más importantes fueron:
-   **Sesgo Salarial:** La distribución de salarios está fuertemente sesgada a la derecha, con una mayoría de salarios en el rango bajo-medio.
-   **Impacto de la Experiencia:** Se confirmó una correlación positiva muy fuerte entre el nivel de experiencia (`EN`, `MI`, `SE`, `EX`) y el salario.
-   **Influencia del Tamaño de la Empresa:** Las empresas de mayor tamaño (`M`, `L`) tienden a ofrecer salarios más competitivos.

### 2. Ingeniería de Características (Feature Engineering)

Para preparar los datos para el modelado, se aplicaron las siguientes transformaciones:
-   **Filtro Geográfico:** El análisis se acotó exclusivamente al **mercado europeo** para aumentar la relevancia de las predicciones.
-   **Codificación Ordinal:** Variables como `experience_level` y `company_size` fueron codificadas numéricamente para preservar su orden jerárquico.
-   **One-Hot Encoding:** Variables categóricas como `company_location`, `industry` y `job_title` (top 20) fueron convertidas a formato dummy.
-   **Procesamiento de Texto:** Se extrajeron las **20 habilidades técnicas más comunes** de la columna `required_skills` y se crearon nuevas columnas binarias para indicar su presencia en cada oferta.
-   **Creación de Variables:** Se generó la variable `is_international` para capturar si la residencia del empleado y la ubicación de la empresa son diferentes.

### 3. Entrenamiento y Evaluación del Modelo

Se adoptó una estrategia competitiva para encontrar el mejor modelo de regresión:
-   **Algoritmos Probados:** `RandomForestRegressor`, `XGBoost` y `LightGBM`.
-   **Optimización:** Se utilizó `RandomizedSearchCV` con validación cruzada para encontrar los mejores hiperparámetros de forma eficiente.
-   **Métrica de Evaluación:** El **coeficiente de determinación (R²)** fue la métrica principal para seleccionar el modelo ganador.

## 🏆 Resultados

El modelo con mejor rendimiento fue el **RandomForestRegressor**, que, tras ser evaluado en un conjunto de datos de prueba completamente nuevo, obtuvo las siguientes métricas:

| Métrica                         | Valor         | Interpretación                                                 |
| ------------------------------- | ------------- | -------------------------------------------------------------- |
| **Coeficiente de Determinación (R²)** | `0.877`       | El modelo explica el **87.7%** de la variabilidad de los salarios. |
| **Error Absoluto Medio (MAE)** | `10,796.61 €` | En promedio, la predicción del modelo se desvía en ~€10,800 del salario real. |

Estos resultados indican un alto poder predictivo y una precisión notable, considerando la dispersión de los salarios en el sector.

## 🚀 Tecnologías Utilizadas

-   **Lenguaje:** Python 3.10
-   **Librerías Principales:**
    -   `Pandas` y `NumPy` para la manipulación de datos.
    -   `Matplotlib` y `Seaborn` para la visualización de datos.
    -   `Scikit-learn` para el preprocesamiento, entrenamiento y evaluación de modelos.
    -   `XGBoost` y `LightGBM` para modelos de boosting.
    -   `Streamlit` para la creación de la demo interactiva.
    -   `Pickle` para la serialización del modelo.

## 💻 Cómo Ejecutar el Proyecto

Para replicar este proyecto o ejecutar la demo en tu máquina local, sigue estos pasos:

1.  **Clonar el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/nombre-del-repositorio.git](https://github.com/tu-usuario/nombre-del-repositorio.git)
    cd nombre-del-repositorio
    ```

2.  **Crear un entorno virtual (recomendado):**
    ```bash
    python -m venv env
    source env/bin/activate  # En Windows: env\Scripts\activate
    ```

3.  **Instalar las dependencias:**
    Asegúrate de tener un archivo `requirements.txt` con el siguiente contenido y luego ejecuta el comando `pip`:
    
    *Contenido de `requirements.txt`:*
    ```
    numpy
    pandas
    matplotlib
    seaborn
    scikit-learn
    xgboost
    lightgbm
    streamlit
    ```

    *Comando de instalación:*
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ejecutar la demo interactiva:**
    Una vez instaladas las dependencias, inicia la aplicación de Streamlit:
    ```bash
    streamlit run demo_modelo.py
    ```
    Se abrirá una nueva pestaña en tu navegador con la aplicación para realizar predicciones.

5.  **Explorar los Notebooks:**
    Puedes abrir y ejecutar los notebooks (`.ipynb`) utilizando Jupyter Notebook o Jupyter Lab para ver el proceso de análisis y modelado en detalle.

## 👤 Autor

Desarrollado por **Eduardo José Limones Contreras**.

-   **LinkedIn:** [Perfil de LinkedIn](https://www.linkedin.com/in/eduardo-jos%C3%A9-limones-contreras-b1348677/)
-   **Correo:** eduardo.limones.contreras@gmail.com
