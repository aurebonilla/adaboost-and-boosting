# AdaBoost and Boosting

Este repositorio contiene implementaciones y experimentos relacionados con algoritmos de boosting, incluyendo AdaBoost y variantes personalizadas, usando datasets como MNIST. Aquí se incluyen ejemplos prácticos, evaluaciones y visualizaciones de rendimiento.

## Características principales

- **Implementación de AdaBoost desde cero:**
  - Entrenamiento utilizando clasificadores simples como Decision Stumps.
  - Mecanismo de ajuste y predicción multiclase mediante clasificadores débiles.

- **Uso de librerías avanzadas:**
  - `sklearn.ensemble.AdaBoostClassifier` para comparaciones.
  - Entrenamiento y evaluación de modelos de aprendizaje profundo (MLP y CNN) usando `Keras` y `TensorFlow`.

- **Evaluación exhaustiva:**
  - Visualización de resultados como tasas de acierto y tiempos de entrenamiento.
  - Comparación de distintos parámetros como el número de clasificadores débiles (T) y características máximas (A).

## Contenido del repositorio

### Archivos principales

- `code.py`: Contiene todas las implementaciones, incluyendo:
  - AdaBoost desde cero.
  - Entrenamiento y visualización para modelos MLP y CNN.
  - Visualizaciones detalladas de precisión y tiempos.

### Funciones clave

- **AdaBoost desde cero:**
  - `Adaboost` y `Adaboost_sin`: Implementación básica de boosting.
  - `DecisionStump`: Clasificador débil.

- **Tareas específicas:**
  - `tarea2a`: Evaluación básica usando AdaBoost.
  - `tarea2b`: Análisis de precisión y tiempos de entrenamiento variando hiperparámetros.
  - `tarea2d` y `tarea2e`: Implementaciones y evaluaciones de redes neuronales MLP y CNN.
  - `tarea1c` y `tarea1d`: Visualización y evaluación de modelos personalizados de AdaBoost.

- **Funciones auxiliares:**
  - `load_MNIST_for_adaboost`: Carga y preprocesamiento del dataset MNIST.
  - `plot_results_2b`, `plot_results_1c`: Visualización de resultados experimentales.

## Requisitos

- Python 3.7+
- Librerías:
  - `numpy`
  - `matplotlib`
  - `sklearn`
  - `keras`
  - `tensorflow`

Para instalar todas las dependencias necesarias:

```bash
pip install -r requirements.txt
```

## Uso

1. Clona el repositorio:

   ```bash
   git clone https://github.com/tuusuario/adaboost-and-boosting.git
   cd adaboost-and-boosting
   ```

2. Ejecuta el script principal:

   ```bash
   python code.py
   ```

3. Modifica el archivo `main()` en `code.py` para ejecutar tareas específicas.

## Visualizaciones

El código incluye visualizaciones detalladas de tasas de precisión y tiempos de entrenamiento. Ejemplo:

- Tasas de precisión variando T y A.
- Gráficos comparativos entre AdaBoost y redes neuronales profundas.


---

¡Gracias por explorar el proyecto! Si encuentras útil este repositorio, no olvides darle una estrella ⭐ en GitHub.

