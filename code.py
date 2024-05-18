import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import keras
from sklearn.metrics import accuracy_score
from tensorflow import keras
import logging, os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import keras
from keras.datasets import mnist
from keras.optimizers import RMSprop, Adam


logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 


######################################
#              TAREA 2A              #
######################################
def tarea2a(X_train, Y_train, X_test, Y_test):
    # Utilizamos el AdaBoostClassifier de sklearn
    adaboost = AdaBoostClassifier()
    start = time.time()
    adaboost.fit(X_train, Y_train)

    # Realizo las predicciones
    test_predictions = adaboost.predict(X_test)

    end=time.time()
    # Calculo las tasas de acierto
    train_accuracy = accuracy_score(Y_test, test_predictions)*100

    # Printeo la información
    print(f"Tasas acierto (train) y tiempo: {train_accuracy:.2f}%, {end-start:.3f} s.")

######################################
#              TAREA 2B              #
######################################
def train_2b(T_values, A_values, X_train, Y_train, X_test, Y_test):
    # Inicializo listas para almacenar resultados de precisión y tiempos de entrenamiento
    test_accuracies = []
    train_times = []

    # Itero sobre los valores de T y A
    for T in T_values:
        for A in A_values:
            # Creo un clasificador AdaBoost con un árbol de decisión como base y configura los parámetros
            DTC = DecisionTreeClassifier(max_features=A)
            adaboost = AdaBoostClassifier(estimator=DTC, n_estimators=T)
            # Registro el tiempo de inicio, entreno el clasificador y registro el tiempo de finalización
            start_time = time.time()
            adaboost.fit(X_train, Y_train)
            end_time = time.time()

            # Realizo predicciones en el conjunto de prueba
            test_predictions = adaboost.predict(X_test)

            # Calculo y almaceno la precisión de las predicciones
            test_accuracy = accuracy_score(Y_test, test_predictions)
            test_accuracies.append(test_accuracy)

            # Almaceno el tiempo total de entrenamiento
            train_times.append(end_time - start_time)

    return np.array(test_accuracies).reshape(len(T_values), len(A_values)), np.array(train_times).reshape(len(T_values), len(A_values))


def plot_results_2b(T_values, A_values, test_accuracies, train_times):
    # Reestructuro los resultados para adecuarlos a la visualización
    test_accuracies = test_accuracies.reshape(len(T_values), len(A_values))
    train_times = train_times.reshape(len(T_values), len(A_values))

    # Defino una paleta de colores combinando azules y verdes
    color_palette = ['#000080', '#0000FF', '#4169E1', '#1E90FF', '#00BFFF',
                     '#006400', '#008000', '#228B22', '#32CD32', '#00FA9A']

    # Creo la figura y el primer eje para la precisión
    fig, ax1 = plt.subplots()

    # Grafico la precisión para cada valor de A alternando colores azules
    for i, A in enumerate(A_values):
        ax1.plot(T_values, test_accuracies[:, i], label=f'Tasas de Acierto A={A}', color=color_palette[i % len(color_palette)])
    ax1.set_xlabel('T')
    ax1.set_ylabel('Tasas de Acierto')
    ax1.legend(loc='upper left')

    # Creo un segundo eje para graficar los tiempos de entrenamiento
    ax2 = ax1.twinx()

    # Grafico el tiempo de entrenamiento para cada valor de A alternando colores azules
    for i, A in enumerate(A_values):
        ax2.bar(T_values, train_times[:, i], label=f'Tiempo de entrenamiento A={A}', alpha=0.5, color=color_palette[i % len(color_palette)])
    ax2.set_ylabel('Tiempo de entrenamiento (s)')
    ax2.legend(loc='upper right')

    # Ajusto el layout y muestro la figura
    fig.tight_layout()
    ax1.set_xticks(T_values)
    ax2.set_xticks(T_values)
    plt.show()


def grafic_2b():
    # Defino los valores de T y A para el experimento
    T_values = [10, 15, 20, 25, 30]
    A_values = [10, 15, 20, 25, 30]

    # Cargo los datos de MNIST para el entrenamiento de AdaBoost
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Llamo a la función de entrenamiento y almaceno los resultados
    test_accuracies, train_times = train_2b(T_values, A_values, X_train, Y_train, X_test, Y_test)
    
    # Llamo a la función de visualización para mostrar los resultados
    plot_results_2b(T_values, A_values, test_accuracies, train_times)



######################################
#              TAREA 2D              #
######################################
def tarea2d():
    # Cargo los datos de MNIST
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Preprocesamiento de los datos
    X_train = X_train.reshape(60000, 784).astype('float32') / 255
    X_test = X_test.reshape(10000, 784).astype('float32') / 255
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    # Construir el modelo MLP
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # Compilar el modelo
    model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

    # Entrenar el modelo
    start = time.time() # Empiezo contador del tiempo
    model.fit(X_train, Y_train,
            batch_size=128,
            epochs=10,
            verbose=1,
            validation_data=(X_test, Y_test))

    # Evaluar el modelo
    score = model.evaluate(X_test, Y_test, verbose=0)
    end = time.time() # Finalizo contador del tiempo
    print('Time:', end-start ,'s')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]*100, '%')
    

######################################
#              TAREA 2E              #
######################################
def tarea2e():
    # Cargo los datos de MNIST
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Preprocesamiento de los datos
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    # Construir el modelo CNN
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) # Aplanar los mapas de características para las capas densas
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # Compilar el modelo
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))

    # Evaluar el modelo
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1]*100, '%')

######################################
#              TAREA 1A              #
######################################
class DecisionStump:
    ## Constructor de clase, con número de características
    def __init__(self, n_features):
        self.feature_idx = np.random.randint(0, n_features)
        self.threshold = np.random.uniform(0, 1)
        self.polarity = np.random.choice([-1, 1])  # Elegir entre -1 o 1
 
    ## Método para obtener una predicción con el clasificador débil
    def predict(self, X):
        # Realiza predicciones basadas en el umbral y la polaridad seleccionados
        feature_values = X[:, self.feature_idx]
        predictions = np.ones(X.shape[0])

         # Dependiendo de la polaridad, asigna -1 a las predicciones que no cumplen la condición
        if self.polarity == -1:
            predictions[feature_values >= self.threshold] = -1
        else:
            predictions[feature_values < self.threshold] = -1

        return predictions #Devuelvo la información
    
class Adaboost:
    # Inicializo los parámetros del algoritmo Adaboost
    def __init__(self, T=5, A=20):
        self.T = T  # Número de clasificadores débiles a entrenar
        self.A = A  # Número de intentos por clasificador
        self.classifiers = []  # Lista para almacenar clasificadores débiles
        self.alphas = []  # Lista para almacenar alfas correspondientes
        self.fit_time = None #Para guardar el tiempo de ejecución
 
    ## Método para entrenar un clasificador fuerte a partir de clasificadores débiles mediante Adaboost
    def fit(self, X, Y, verbose=False):
        # Entrena el modelo Adaboost
        n_samples, n_features = X.shape
        self.weights = np.ones(n_samples) / n_samples # Inicializo los pesos de las muestras

        start_time = time.time() # Comienzo a medir el tiempo de ejecución

        for t in range(self.T):
            best_classifier = None
            best_error = np.inf

            for a in range(self.A):
                # Crear un nuevo clasificador débil aleatorio
                classifier = DecisionStump(n_features)
                predictions = classifier.predict(X)
                error = np.sum(self.weights * (predictions != Y))

                # Actualizo el mejor clasificador si encuentra uno con menor error
                if error < best_error:
                    best_error = error
                    best_classifier = classifier

            # Calculo el valor de alfa para el clasificador actual
            alpha = 0.5 * np.log((1 - best_error) / (best_error + 1e-10))
            self.alphas.append(alpha)  # Guardo el alpha correspondiente al clasificador actual

            # Asigno el valor de alpha al clasificador y actualizo los pesos de las muestras
            best_classifier.alpha = alpha
            predictions = best_classifier.predict(X)
            self.weights *= np.exp(-alpha * Y * predictions)
            self.weights /= np.sum(self.weights)
            self.classifiers.append(best_classifier)

            ######################################
            #              TAREA 1B              #
            ######################################
            # Imprimo información del clasificador si el modo verbose está activo
            if verbose:
                if best_classifier.polarity == 1:
                    print(f"Añadido clasificador {1+t}: {best_classifier.feature_idx}, {best_classifier.threshold:.4f}, +1, {best_error:.6f}")
                else:
                    print(f"Añadido clasificador {1+t}: {best_classifier.feature_idx}, {best_classifier.threshold:.4f}, {best_classifier.polarity}, {best_error:.6f}")

            end_time = time.time() # Finalizo el contador del tiempo tiempo
            self.fit_time=end_time - start_time  # Almaceno el tiempo total de ajuste

    # Realiza predicciones con el modelo Adaboost
    def predict(self, X):
        H_X = np.zeros(len(X))

        # Sumo las predicciones de todos los clasificadores ponderadas por sus alphas
        for classifier, alpha in zip(self.classifiers, self.alphas):
            predictions = alpha * classifier.predict(X)
            H_X += predictions

        return np.sign(H_X) # Devuelvo la clase final basada en el signo de la suma


######################################
#              TAREA 1B              #
######################################
def show_result(X_train, Y_train, X_test, Y_test):
    # Printeo enunciado de la práctica
    print(f"Entrenando clasificador Adaboost para el dígito 9, T=20, A=10")
    print(f"Entrenando clasificadores de umbral (con dimensión, umbral, dirección y error):")
   
    # Inicializo y entreno el modelo Adaboost con los datos de entrenamiento
    adaboost=Adaboost(T=20, A=10)
    adaboost.fit(X_train, Y_train, True)  # El argumento 'True' activa el modo verbose para mostrar la información

    # Realizo predicciones tanto en el conjunto de entrenamiento como en el de prueba
    train_predictions = adaboost.predict(X_train)
    test_predictions = adaboost.predict(X_test)

    # Calculo la precisión de las predicciones en ambos conjuntos
    train_accuracy = np.mean(train_predictions == Y_train) * 100
    test_accuracy = np.mean(test_predictions == Y_test) * 100

    # Imprimo las tasas de acierto en los conjuntos de entrenamiento y prueba, y el tiempo de ajuste
    print(f"Tasas acierto (train, test) y tiempo: {train_accuracy:.2f}%, {test_accuracy:.2f}%, {adaboost.fit_time:.3f} s.")

######################################
#              TAREA 1C              #
######################################
def train_1c(T_values, A_values, X_train, Y_train, X_test, Y_test):
    # Inicializo listas vacías para almacenar los resultados
    train_accuracies = []
    test_accuracies = []
    train_times = []

    # Itero sobre los valores dados para T y A
    for T in T_values:
        for A in A_values:
            # Creo y entreno el clasificador AdaBoost con los parámetros T y A
            adaboost = Adaboost(T, A)
            adaboost.fit(X_train, Y_train)
            
            # Realizo predicciones en los conjuntos de entrenamiento y prueba
            train_predictions = adaboost.predict(X_train)
            test_predictions = adaboost.predict(X_test)

            # Calcula la precisión (tasa de acierto) en ambos conjuntos
            train_accuracy = accuracy_score(Y_train, train_predictions)
            test_accuracy = accuracy_score(Y_test, test_predictions)

            # Almaceno las precisión y tiempos de entrenamiento para su posterior análisis
            train_accuracies.append(train_accuracy)
            train_times.append(adaboost.fit_time)

    return np.array(train_accuracies), np.array(train_times)

def plot_results_1c(T_values, A_values, train_accuracies, train_times):
    # Reestructuro los resultados para adecuarlos a la visualización
    train_accuracies = train_accuracies.reshape(len(T_values), len(A_values))
    train_times = train_times.reshape(len(T_values), len(A_values))

    # Defino una paleta de colores combinando azules y verdes
    color_palette = ['#000080', '#0000FF', '#4169E1', '#1E90FF', '#00BFFF',
                     '#006400', '#008000', '#228B22', '#32CD32', '#00FA9A']

    # Creo la figura y el primer eje para la precisión
    fig, ax1 = plt.subplots()

    # Grafico la precisión para cada valor de A alternando colores azules
    for i, A in enumerate(A_values):
        ax1.plot(T_values, train_accuracies[:, i], label=f'Tasas de Acierto A={A}', color=color_palette[i % len(color_palette)])
    ax1.set_xlabel('T')
    ax1.set_ylabel('Tasas de Acierto')
    ax1.legend(loc='upper left')

    # Creo un segundo eje para graficar los tiempos de entrenamiento
    ax2 = ax1.twinx()

    # Grafico el tiempo de entrenamiento para cada valor de A alternando colores azules
    for i, A in enumerate(A_values):
        ax2.bar(T_values, train_times[:, i], label=f'Tiempo de entrenamiento A={A}', alpha=0.5, color=color_palette[i % len(color_palette)])
    ax2.set_ylabel('Tiempo de entrenamiento (s)')
    ax2.legend(loc='upper right')

    # Ajusto el layout y muestro la figura
    fig.tight_layout()
    ax1.set_xticks(T_values)
    ax2.set_xticks(T_values)
    plt.show()

def grafic_1c():
    # Defino los valores para T y A a utilizar en el entrenamiento 
    T_values = [10, 15, 20, 25]
    A_values = [10, 15, 20, 25]

    # Cargo los datos del dataset MNIST para entrenar AdaBoost
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Convierto las clases a -1 y 1, preparándolas para AdaBoost
    Y_train = np.where(Y_train == 0, 1, -1)
    Y_test = np.where(Y_test == 0, 1, -1)

    # Entreno el modelo y obtengo las precisiones y tiempos de entrenamiento
    train_accuracies, train_times = train_1c(T_values, A_values, X_train, Y_train, X_test, Y_test)
    
    # Visualizo los resultados del entrenamiento
    plot_results_1c(T_values, A_values, train_accuracies, train_times)

"""
def graficas(X_train, Y_train, X_test, Y_test):
    T_values = range(1, 21)  # Por ejemplo, de 1 a 20
    A_values = range(1, 21)  # Por ejemplo, de 1 a 20

    training_times = []
    accuracies = [] 

    for T in T_values:
        for A in A_values:
            adaboost = Adaboost(T=T, A=A)
            
            start_time = time.time()
            adaboost.fit(X_train, Y_train)
            training_time = time.time() - start_time
            
            predictions = adaboost.predict(X_test)
            accuracy = np.mean(predictions == Y_test) * 100
            
            training_times.append(training_time)
            accuracies.append(accuracy)

    # Convertir las listas en arrays de NumPy y cambiar su forma
    training_times = np.array(training_times).reshape(len(T_values), len(A_values))
    accuracies = np.array(accuracies).reshape(len(T_values), len(A_values))

    # Graficar tiempo de entrenamiento para cada valor de A
    plt.figure(figsize=(10, 5))
    for i in range(len(A_values)):
        plt.plot(T_values, training_times[:, i], label=f'A={A_values[i]}')
    plt.xlabel('T')
    plt.ylabel('Tiempo de entrenamiento (s)')
    plt.legend()
    plt.show()

    # Graficar tasa de acierto para cada valor de A
    plt.figure(figsize=(10, 5))
    for i in range(len(A_values)):
        plt.plot(T_values, accuracies[:, i], label=f'A={A_values[i]}')
    plt.xlabel('T')
    plt.ylabel('Tasa de acierto (%)')
    plt.legend()
    plt.show()"""

######################################
#              TAREA 1D              #
######################################
class Adaboost_sin:
    def __init__(self, T, A):
        self.T = T
        self.A = A
        self.classifiers = [] # Lista para almacenar los clasificadores y sus pesos

    # Entreno el modelo Adaboost en el conjunto de datos X y Y
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.ones(n_samples) / n_samples  # Inicializo los pesos de las muestras

        for _ in range(self.T):
            # Inicializo variables para encontrar el mejor clasificador
            best_classifier = None
            best_error = float('inf')  
            best_predictions = None

            for _ in range(self.A):
                adaboost = DecisionStump(n_features)
                predictions = adaboost.predict(X)
                error = np.sum(self.weights * (predictions != Y))

                if error < best_error:
                    best_classifier = adaboost
                    best_error = error
                    best_predictions = predictions

            alpha = 0.5 * np.log((1.0 - best_error) / (best_error + 1e-10))
            self.weights *= np.exp(-alpha * Y * best_predictions)
            self.weights /= np.sum(self.weights)
            self.classifiers.append((best_classifier, alpha))

    def predict(self, X):
        # Realizo predicciones sobre el conjunto de datos X utilizando todos los clasificadores
        suma = sum(alpha * adaboost.predict(X) for adaboost, alpha in self.classifiers)
        return suma # Devuelvo las predicciones finales

class MulticlassClassifier:
    def __init__(self, T, A):
        self.classifiers = [Adaboost_sin(T, A) for l in range(10)]

    def fit(self, X, Y):
        for i in range(10):
            binary_y = np.where(Y == i, 1, -1)
            self.classifiers[i].fit(X, binary_y)

    def predict(self, X):
        scores = np.array([adaboost.predict(X) for adaboost in self.classifiers])
        predictions = np.argmax(scores, axis=0)
        return predictions

def tarea1d(T, A):
    # Cargo el conjunto de datos MNIST para entrenar y probar el modelo Adaboost.
    X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()

    # Creo y entreno un clasificador Adaboost para clasificación multiclase.
    adaboost = MulticlassClassifier(T, A)

    start_time = time.time() # Inicio el contador
    adaboost.fit(X_train, Y_train)

    train_predictions = adaboost.predict(X_train)
    end_time = time.time() # Finalizo el contador

    # Calculo y muestro la precisión del clasificador en los datos de entrenamiento.
    train_accuracy = accuracy_score(Y_train, train_predictions)*100

    # Printeo los clasificadores
    for i, adaboost in enumerate(adaboost.classifiers, 1):
        print(f'Clasificador {i}:')
        for j, (clasi, alpha) in enumerate(adaboost.classifiers, 1):
            if clasi.polarity == 1:
                print(f'\tAñadido clasificador {j}: {clasi.feature_idx}, {clasi.threshold:4f}, +1, {alpha:6f}')
            else:
                print(f'\tAñadido clasificador {j}: {clasi.feature_idx}, {clasi.threshold:4f}, {clasi.polarity}, {alpha:6f}')

    # Printeo las tasas
    print(f"Tasas acierto (train) y tiempo: {train_accuracy:.2f}%, {end_time-start_time:.3f} s.")

######################################
#            Load MNIST     N9       #
######################################
def load_MNIST2_for_adaboost():
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de YannLecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    # Formatear imággenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    #X_train = X_train.astype("float32") / 255.0
    #X_test = X_test.astype("float32") / 255.0
    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train = np.where(Y_train== 9,1,-1)
    Y_test = np.where(Y_test== 9,1,-1)

    return X_train, Y_train, X_test, Y_test

######################################
#            Load MNIST              #
######################################
def load_MNIST_for_adaboost():
    # Cargar los datos de entrenamiento y test tal y como nos los sirve keras (MNIST de YannLecun)
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
    # Formatear imággenes a vectores de floats y normalizar
    X_train = X_train.reshape((X_train.shape[0], 28*28)).astype("float32") / 255.0
    X_test = X_test.reshape((X_test.shape[0], 28*28)).astype("float32") / 255.0
    #X_train = X_train.astype("float32") / 255.0
    #X_test = X_test.astype("float32") / 255.0
    # Formatear las clases a enteros con signo para aceptar clase -1
    Y_train = Y_train.astype("int8")
    Y_test = Y_test.astype("int8")

    return X_train, Y_train, X_test, Y_test

def main():
    #X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    # llamo a la tarea 1B
    #X_train, Y_train, X_test, Y_test = load_MNIST2_for_adaboost()
    #show_result(X_train, Y_train, X_test, Y_test)

    #llamo a la tarea 1C
    #grafic_1c()

    #llamo a la tarea 1D
    #tarea1d(20,20)

    #llamo a la tarea 2A
    #X_train, Y_train, X_test, Y_test = load_MNIST_for_adaboost()
    #tarea2a(X_train, Y_train, X_test, Y_test)

    #llamo a la tarea 2B
    #grafic_2b()

    #llamo a la tarea 2D
    tarea2d()
    
    #llamo a la tarea 2E
    #tarea2e()

if __name__ == "__main__":
    main()