# <h1 align="center">_SD_ANALYTICS_</h1>

<p align="center">
  <img src="images/SD_analytics.png"  height="400">
<p align="center">

## Índice

1. [Descripción](#descripción)
2. [Características](#características)
3. [Requisitos](#requisitos)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Colaboradores](#colaboradores)
6. [Tecnologías](#tecnologías)

## Descripción

Este proyecto busca utilizando la Ciencia de Datos, identificar características que influyan en la deserción de clientes(Churning), así también con los datos obtenidos ver si se puede obtener revelaciones de cómo optimizar el servicio existente..

Objetivo del proyecto: Desarrollé un flujo de trabajo completo de MLOps, automatizado y reproducible, para predecir la pérdida de clientes, abarcando todas las etapas desde la ingesta de datos hasta la implementación y predicción en una aplicación web interactiva.

Pila tecnológica: Utilicé ZenML para gestionar todo el pipeline de principio a fin, WandB para rastrear experimentos, BentoML para empaquetar el modelo listo para producción, y Streamlit para crear una interfaz web sencilla.
Estructura de archivos: La carpeta src/ contenía lógica reutilizable e independiente del framework, mientras que steps/ incluía funciones decoradas que servían como interfaz del pipeline, facilitando la integración con ZenML.

Diseño modular: La separación entre la lógica principal en src/ y la interfaz de pipeline en steps/ permitió mayor reutilización, prueba y mantenimiento del código, asegurando un proceso completo de MLOps de principio a fin.

## Características

- Análisis Inicial del set de datos.
- Visualización Análisis Exploratorio.
- Preprocesamiento.
- Ingeniería de Datos.
- Predicciones y modelado de datos.

## Requisitos

- Python 3.x
- pandas
- matplotlib
- seaborn
- numpy
- jupyter 


## Estructura del Proyecto

- Proyecto ML Customer Churn/
├── .venv/
├── .zen/
├── data/
├── materializer/
├── pipelines/
├── src/
├── steps/
└── wandb/

## Enlaces

Dashboard: 

Deploy: 

## Stack de tecnologías y herramientas

|  Librería/herramienta    |   Logo                                    | Descripción                                                                                                           |
|----------------------|-----------------------------------------|----------------------------------------------|
| **Pandas**   |      <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/Pandas_logo.svg/1200px-Pandas_logo.svg.png" width="100">   | Librería de Python para manipulación y análisis de datos.|
| **Matplotlib**|<img src="https://matplotlib.org/_static/logo_light.svg" width="100">| Librería usada para la generación de gráficos en dos dimensiones.|
|**Seaborn**|<img src="https://seaborn.pydata.org/_images/logo-tall-lightbg.svg" width="100"> | Librería de Python creada sobre matplotlib, usada para crear gráficos estadísticos.|
| **Jupyter**|<img src="https://jupyter.org/assets/homepage/main-logo.svg" width="65">| Software gratuito, estándares abiertos y servicios web para informática interactiva en todos los lenguajes de programación.|
| **Visual Studio Code**|<img src="https://static-00.iconduck.com/assets.00/visual-studio-code-icon-512x506-2fdb6ar6.png" width="70">| Editor de código fuente.|
| **Colaboratory con Python**|<img src="https://colab.research.google.com/img/colab_favicon_256px.png" width="60">| Plataforma de Google basada en Jupyter Notebooks, junto con las librerías de Python para análisis de datos como Pandas y Matplotlib.|
| **Power BI**|<img src="https://cdn-dynmedia-1.microsoft.com/is/image/microsoftcorp/Analysts_PBI?resMode=sharp2&op_usm=1.5,0.65,15,0&wid=2000&qlt=99&fmt=png-alpha&fit=constrain" width="100">| Herramienta líder en el mercado para crear visualizaciones de datos avanzadas, informes interactivos y paneles de control.|
| **Canva**|<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Canva_Logo.svg/250px-Canva_Logo.svg.png" width="100">| Plataforma de diseño gráfico y composición de imágenes.|
| **Power Point**|<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/Microsoft_PowerPoint_2013-2019_logo.svg/610px-Microsoft_PowerPoint_2013-2019_logo.svg.png" width="100">| Microsoft PowerPoint (PPT) es un software de ofimática diseñado para realizar presentación de diapositivas.|
| **Python**|<img src="https://seeklogo.com/images/P/python-logo-A32636CAA3-seeklogo.com.png" width="50">| Lenguaje de programación utilizado para análisis de datos y desarrollo de aplicaciones.|
| **GitHub**|<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" width="100">| Plataforma de desarrollo colaborativo para proyectos de software.|
| **Draw.io**|<img src="https://store-images.s-microsoft.com/image/apps.14142.2ec57164-ba5b-42ee-8253-ade72ca74c8d.e8efbcc8-e27f-4682-9321-eb111eb3bc68.950e1c22-0dd9-4079-807b-145ecdb6df9c.png" width="100">| Software de diagramas en línea gratuito para crear diagramas de flujo, diagramas de procesos, organigramas, UML, ER y diagramas de red.|
| **Google Drive**|<img src="https://upload.wikimedia.org/wikipedia/commons/1/12/Google_Drive_icon_%282020%29.svg" width="100"> | Servicio de alojamiento y sincronización de archivos.|
| **Streamlit** | <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png" width="100"> | Streamlit es una herramienta de código abierto diseñada para crear aplicaciones web interactivas y visualizaciones de datos de manera rápida y sencilla utilizando Python.|



## Colaboradores

|                         | Nombre   |   Rol                    | GitHub & LinkedIn                                                                                                                                                                                          |
| ----------------------------- | -------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <img width="60" height="60" src="https://github.com/Sergius-DS.png" alt="Sergius-DS" /> | Sergio Rivera Bustamante | Data Scientist | [![Github](https://skillicons.dev/icons?i=github)](https://github.com/Sergius-DS) [![Linkedin](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/sergio-rivera-bustamante-6642b836/)                         |
|                               |

