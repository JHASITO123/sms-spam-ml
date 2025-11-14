Clasificación de SMS Spam con Machine Learning y Análisis de Tópicos

Proyecto de Procesamiento de Lenguaje Natural (PLN)
Autor: Tu Nombre
Tecnologías: Python, Scikit-Learn, NLTK, Gensim, TF-IDF, LDA

Descripción del Proyecto

Este proyecto desarrolla un sistema completo de Machine Learning para una empresa de telecomunicaciones.
El objetivo es doble:

Clasificar mensajes SMS como:

ham → mensaje legítimo

spam → mensaje no deseado

Analizar el contenido del spam para descubrir temas o patrones comunes mediante un modelo de tópicos (LDA).

Este proyecto sigue todo el flujo profesional de un pipeline de Procesamiento de Lenguaje Natural (PLN): análisis exploratorio, limpieza, vectorización, modelado, evaluación y análisis de tópicos.

Dataset

SMS Spam Collection Dataset
Fuente: Kaggle
https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

El archivo utilizado es:

data/spam.csv


Este dataset contiene miles de SMS reales clasificados como spam o ham.
Requiere lectura con encoding='latin-1'.

Flujo del Proyecto
1. Análisis Exploratorio (EDA)

Carga del dataset con pandas.

Renombramiento de columnas (v1 → etiqueta, v2 → texto).

Eliminación de columnas irrelevantes.

Visualización del desbalance de clases usando seaborn.countplot.

Análisis del problema del desbalance de clases y por qué la accuracy no es una métrica confiable.

2. Preprocesamiento del Texto (NLTK)

Se construyó una función preprocess_text() que realiza:

Conversión a minúsculas

Eliminación de puntuación y números mediante expresiones regulares

Tokenización

Eliminación de stopwords

Lematización con WordNet

El resultado es una nueva columna:

texto_limpio

3. Vectorización con TF-IDF

Se utilizó TfidfVectorizer para transformar el texto en vectores numéricos.
Posteriormente se realizó la división del dataset:

80% para entrenamiento

20% para prueba

4. Entrenamiento de Modelos

Se entrenaron dos modelos de clasificación:

Multinomial Naive Bayes

Logistic Regression

Para ambos se generaron:

Reporte de clasificación (precision, recall, f1-score)

Matriz de confusión

El proyecto compara ambos modelos utilizando el F1-score específicamente para la clase spam.

5. Selección del Mejor Modelo

El sistema selecciona automáticamente el modelo con mejor F1-score en la clase spam.
Este criterio permite evaluar con precisión qué tan bien el modelo identifica mensajes no deseados sin depender de la accuracy.

6. Análisis de Tópicos (LDA con gensim)

Tras clasificar los mensajes:

Se filtraron únicamente aquellos clasificados como spam.

Se tokenizaron nuevamente.

Se creó un diccionario y un corpus con gensim.

Se entrenó un modelo LDA con cuatro tópicos.

Este análisis permite entender las categorías principales dentro del spam.

Interpretación de los Tópicos Detectados

El modelo LDA identificó temas frecuentes en los mensajes spam.
Dependiendo de la ejecución, los tópicos detectados suelen agruparse en:

Premios y loterías

Finanzas, préstamos o fraudes bancarios

Suscripciones móviles y servicios de pago

Promociones comerciales y descuentos

Conclusiones

El dataset presenta un desbalance significativo entre ham y spam.

La métrica accuracy no es adecuada para este caso; es preferible utilizar precision, recall y F1-score.

El modelo con mejor desempeño para la clase spam fue:
Naive Bayes o Logistic Regression (según los resultados obtenidos).

El análisis de tópicos permitió identificar patrones comunes y categorías dominantes dentro del spam.

Estructura del Repositorio
sms-spam-ml/
├─ data/
│  └─ spam.csv
├─ sms_spam_pln.ipynb
├─ README.md
└─ requirements.txt

Instalación

Ejecutar en un entorno virtual o directamente:

pip install -r requirements.txt

Ejecución

Abrir el notebook:

sms_spam_pln.ipynb


y ejecutar las celdas en orden.

Dependencias

Las dependencias se encuentran en:

requirements.txt
