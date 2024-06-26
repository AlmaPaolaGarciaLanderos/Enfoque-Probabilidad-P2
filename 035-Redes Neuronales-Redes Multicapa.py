#Alma Paola Garcia Landeros
#21110038
#6E1
#Inteligencia Artificial
#Centro de Enseñanza Tecnica Industrial           

import tensorflow as tf
from tensorflow.keras import layers, models

# Definir la arquitectura de la red neuronal
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Capa de entrada: aplanar la imagen 28x28
    layers.Dense(128, activation='relu'),  # Capa oculta 1: 128 neuronas con función de activación ReLU
    layers.Dense(64, activation='relu'),   # Capa oculta 2: 64 neuronas con función de activación ReLU
    layers.Dense(10, activation='softmax') # Capa de salida: 10 neuronas con función de activación Softmax
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo con datos de ejemplo (por ejemplo, el conjunto de datos MNIST)
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Ajustar el modelo
model.fit(x_train, y_train, epochs=5)

# Evaluar el modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Precisión del modelo en el conjunto de prueba:", test_acc)
