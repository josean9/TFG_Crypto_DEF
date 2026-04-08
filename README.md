# 🚀 TFG: Análisis y Predicción en el Mercado de Criptomonedas (Bitcoin & Ethereum)

## 📌 Introducción y Propósito del Proyecto
Este Proyecto de Fin de Grado (TFG) tiene como objetivo principal estudiar, comprender y predecir el comportamiento del mercado de criptomonedas (centrado en Bitcoin y Ethereum), el cual se caracteriza por su alta complejidad y extrema volatilidad. 

El proyecto no se limita a un clásico análisis de series de precios, sino que busca obtener una visión de 360 grados. Para ello, cruzamos la acción del precio con métricas de comportamiento técnico, indicadores macroeconómicos globales (como la inflación o los tipos de interés de la FED) y métricas puramente psicológicas sobre el sentimiento del mercado (Fear & Greed Index). Todo este ecosistema de datos alimenta algoritmos de Machine Learning y Deep Learning con la finalidad de extraer patrones subyacentes y generar pronósticos fiables.

---

## 🧠 Estructura y Flujo de Trabajo (De mayor a menor complejidad)

### 1. Modelado Predictivo de Series Temporales (Deep Learning - LSTM)
El objetivo cumbre del proyecto es la predicción de precios y movimientos de mercado utilizando Redes Neuronales Recurrentes, en particular arquitecturas **LSTM (Long Short-Term Memory)**. 
- **¿Por qué LSTM?** Al tratarse de series temporales financieras secuenciales y ruidosas, una capa LSTM es capaz de "recordar" dependencias a largo plazo y "olvidar" ruido a corto plazo, adecuándose perfectamente a los ciclos del mercado cripto.
- **Refinamiento de la Señal:** Previo a la entrada a la red neuronal, aplicamos técnicas de procesamiento de señales como el **Filtro de Kalman**, lo que nos permite aislar de manera más efectiva la verdadera tendencia del precio, descartando anomalías repentinas.

### 2. Segmentación y Detección de Regímenes de Mercado (Clustering K-Means)
Para poder modelar mejor el mercado, primero tenemos que entender en qué fase se encuentra. Utilizando un algoritmo de aprendizaje no supervisado (**K-Means Clustering**), analizamos el histórico multidimensional para catalogar ventanas climáticas del mercado.
- Esto nos permite agrupar días, semanas o meses en distintos "perfiles o regímenes de mercado" (ej: mercado alcista sobrecomprado, acumulación lateral con bajo interés, pánico con caída macro, etc.), dando contextos de entrenamiento y validación mucho más precisos.

### 3. Ingeniería de Características (Feature Engineering) e Indicadores
Un modelo de Machine Learning es tan bueno como las abstracciones de datos que consume. Aquí transformamos la simple información de precios en marcadores matemáticos y estadísticos valiosos:
- **Análisis Técnico en Datos:** Codificamos matemáticamente el momentum de los activos incorporando medias móviles, **RSI** (Índice de Fuerza Relativa), **MACD** y **Estocásticos**.
- **Análisis de Sentimiento Integral:** Integramos variables cualitativas como el *Fear and Greed Index*, pasándolos por transformaciones como **One-Hot Encoding** para que las redes neuronales interpreten de manera booleana el pánico y la euforia.
- **Dominancia y Macro:** Incluimos la proporción del mercado que domina BTC y ETH (Dominance), combinándola con la evolución de la tasa de inflación y tipos de interés, captando tanto rotaciones internas del capital cripto como flujos sistémicos de capital global.

### 4. Análisis de Relaciones (Matriz de Correlaciones) y Agrupación Temporal
Buscando siempre un dataset robusto estadísticamente:
- Evaluamos de forma intensiva cómo se comportan nuestros indicadores unos con respecto a otros y, sobre todo, frente a nuestro objetivo principal (ej: el precio de cierre de Ethereum o Bitcoin). Analizamos **matrices de correlación** con gran cuidado de no eliminar variables esenciales, ajustando dinámicamente nuestra selección de features.
- Efectuamos resampleos y remuestreos de los datos (pasando de cierres diarios a visualizaciones agrupadas de fin de semana/frecuencia semanal o mensual), lo cual es de gran valor para evitar el inmenso ruido intradía.

### 5. Arquitectura de Extracción e Integración de Datos (Data Pipeline)
Es la base sobre la que reposa toda la maquinaria predictiva. Ha constituido la primera gran fase de consolidación y limpieza:
- Consiste en reunir toda la data de distintas procedencias (OHLCV de exchanges, métricas on-chain, datos tradicionales) logrando alinear todos los históricos de manera simétrica por fecha.
- Se resuelve el desafío de datos faltantes (NAs), unificación de formatos cíclicos e imputación adecuada de variables estructurales (como las tasas macro actualizadas mensualmente emparejadas frente a precios diarios).

---

## 🛠 En Resumen
A través de **TFG_Crypto_DEF**, planteamos un end-to-end de Data Science puro aplicado a finanzas descentralizadas. Pasamos por la ingesta de las series continuas en bruto, generamos variables complejas (feature engineering exhaustiva), y finalizamos aportando analítica de machine learning y predicciones de deep learning adaptadas al complejo entorno del mercado criptográfico.


┌─────────────────────────────────────────────────────────────┐
│                     CAPA 1: DATOS                           │
│  Dataset histórico (2800 días)                              │
│  + Noticias scrapeadas (Reddit, Twitter, fuentes macro)     │
└──────────────────┬──────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
┌───────────────┐    ┌─────────────────────┐
│  CAPA 2A:     │    │  CAPA 2B:           │
│  Transformer  │    │  Base vectorial     │
│  MIMO         │    │  (ChromaDB/FAISS)   │
│               │    │                     │
│  Input: 30d   │    │  Noticias + papers  │
│  Output:      │    │  + datos macro      │
│  [+1d,+2d,+3d]│    │  embeddings         │
│  + confianza  │    │                     │
└───────┬───────┘    └──────────┬──────────┘
        │                       │
        └──────────┬────────────┘
                   ↓
┌─────────────────────────────────────────────────────────────┐
│                     CAPA 3: LLM + RAG                       │
│                                                             │
│  Prompt construido dinámicamente:                           │
│  - Señal del Transformer: "predice +2.1% con 58% confianza" │
│  - Contexto recuperado: noticias relevantes últimos 7 días  │
│  - Datos macro actuales: fed_rate, inflación, dominancias   │
│  - Analogías históricas: "situación similar a Oct 2020"     │
│                                                             │
│  → LLM sintetiza todo y da análisis narrativo               │
└─────────────────────────────────────────────────────────────┘