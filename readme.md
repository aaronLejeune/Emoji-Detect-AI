# Proof of concept :thought_balloon:
###### :boy: Aaron Lejeune

### Samenvatting.
Om een Neural Network te maken (AI) heb je natuurlijk een programmeertaal nodig. Je kan de Tenserflow Library in verschillende talen gebruiken zoals: JS, Swift, Python,... en die deployen op verschillende platformen zoals smarthphones, Raspberry Pi's, Web services, API's,...

Je kan 1001 dingen (laten) bouwen met een AI. Van software die kleren kunnen categoriseren, tot een auto die zichzelf kan laten parkeren. Een neuraal netwerk maken vereist heel wat denkwerk. Je kan niet zomaar '*effkes nen AI maken die Snake kan uitspelen*'. Toch is het minder moeilijk dan dat je op het eerste zicht zou denken, en dat ik laat ik graag even zien met deze simpele AI.

### Gebruikte Technology:
1. Pyhton
2. Numpy (lib.)
     - voor multi-dimensionale arrays
3. Matplotli
     - wiskundige lib voor grafieken en statistieken
4. Tenserflow  
     - machine learning module
5. Keras       
     - machine learning module
     
### Gebruikte Methodes:
##### Start
1. Importeren van de verschillende Libs
'''
from __future__ import absolute_import, division, print_function, unicode_literals

*TensorFlow and tf.keras*
import tensorflow as tf
from tensorflow import keras

*Helper libraries*
import numpy as np
import matplotlib.pyplot as plt
'''

2. Valideren of de import gelukt is:
'''
print(tf.__version__)
print(np.__version__)
'''

3. Importeren + laden test datasets (keras fashion_mnist)
'''
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
'''

##### Label Data
'''
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''
