from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
print(np.__version__)

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0 #make images smaller, 255 = blackest | 254 = blacker -> */255 = 1 = blackest | 0.99 = blacker
test_images = test_images/255.0

# print(train_images[7])

#plt.imshow(train_images[9]) #show image
#plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

prediction = model.predict(test_images)

for i in range(5):   #loop 5 keer door een image + laat zien wat het neural network denk dat het is VS. wat het echt is
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()

#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print("Tested Acc", test_acc)

##model.evaluate(test_images, test_labels)
