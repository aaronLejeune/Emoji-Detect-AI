# Proof of concept :thought_balloon:
###### :boy: Aaron Lejeune

[slide deck](https://docs.google.com/presentation/d/1WrbTrKgZD-McyKtD8BvaVnMMyVbe82zXPe5iC77b3Ag/edit?usp=sharing)

## Table of contents
1. Samenvatting
2. 'Testing' fase
3. Uitvoering
4. Conclusie

#  :page_with_curl:  Samenvatting.
Om een Neural Network te maken (AI) heb je natuurlijk een programmeertaal nodig. Je kan de Tenserflow Library in verschillende talen gebruiken zoals: JS, Swift, Python,... en die deployen op verschillende platformen zoals smarthphones, Raspberry Pi's, Web services, API's,...

Je kan 1001 dingen (laten) bouwen met een AI. Van software die kleren kunnen categoriseren, tot een auto die zichzelf kan laten parkeren. Een neuraal netwerk maken vereist heel wat denkwerk. Je kan niet zomaar '*effkes nen AI maken die Snake kan uitspelen*'. Toch is het minder moeilijk dan dat je op het eerste zicht zou denken, en dat ik laat ik graag even zien met deze simpele AI.

# :recycle: Testfase (sorteren van kledingstukken)
### ['Link guide'](https://www.tensorflow.org/tutorials/keras/classification)

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
### Imports
1. __Importeren van de verschillende Libs__

Een AI maken van scratch is bijna onmoglijk. Gelukkig hebben we de bekende AI library van Tenserflow (gemaakt door Google) die het mogelijk maakt om 'vrij eenvoudig' een AI te maken. Daarnaast importeren we ook Keras die ons eigenlijk allemaal datasets geeft om mee te werken. Een dataset is eigenlijk een database van je onderwerp. Wil je een AI maken die kledingstukken kan detecteren? Dan gebruik je een dataset met daarin allemaal afbeeldingen van kleren met hun bijhorende naam.

```
from __future__ import absolute_import, division, print_function, unicode_literals

*TensorFlow and tf.keras*
import tensorflow as tf
from tensorflow import keras

*Helper libraries*
import numpy as np
import matplotlib.pyplot as plt
```

2. __Valideren of de import gelukt is:__

Hier printen we gewoon de versie uit van de libs die we gebruiken. Dit is een simpele manier om te valideren dat de installatie geslaagd is.

```
print(tf.__version__)
print(np.__version__)
```

3. __Importeren + laden test datasets (keras fashion_mnist)__

Zoals eerder al besproken zorgt Keras voor datasets. ['Op de website van Keras'](https://keras.io/examples/mnist_dataset_api/) kan je kiezen tussen allemaal verschillende datasets en tutorials van hoe je API het best kan gebruiken. In het stukje code hieronder importeren we dus 1 bepaalde dataset en delen we die dataset op in TRAIN en TEST data.

```
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
```

TRAIN data is meestal ongeveer 3/4de van je hele dataset en zoals het woord letterlijk zegt, ge je je AI trainen om uiteindelijk een test te gaan doen. De TEST data wordt dan uiteindelijk gebruikt om de effectieve test af te leggen (met de resterende 1/4de van de dataset). Je kan het een beetje bekijken als leren voor een examen. Je maakt allemaal oefeningen (TRAIN data) om je voor te bereiden op het examen (TEST data). Hoe beter je oefeningen (TRAIN data) zijn, hoe waarschijnlijker het is dat het examen (TEST data) ook goed zal verlopen.

Het is belangrijk om TRAIN en TEST data uit een te haen. Stel je voor dat je leert voor een examen. De docent heeft perongeluk de echte examenvragen tussen de oefening gestoken. Een AI mag niet weten wat de examenvragen kunnen zijn, anders zou het natuurlijk veel hogere resulataten halen dan het eigenlijk kan.

### Label Data

Omdat de data in de dataset uit nummers bestaan, gaan we elke nummer een naam geven. Hieronder gaan we eigenlijk categoriseren.
```
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```

### Layers opmaken

Hieronder maken we de verschillende 'trainingschemas' op. Je kan kiezen op welke manier je AI getrained kan worden adv. verschillende curves zoals relu, softmax, ...
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

## :bicyclist: Uitvoering

De code hieronder is de effectieve 'uitvoering' van de AI. Je kan zien dat we in de compile functie nog de laatste parameters meegeven. Dat is heel belangrijk! Slecht afgestelde parameters kunnen ervoor zorgen dat de accuracy heel hard naar beneden gaat. Als de accuracy bijvoorbeeld 80% is, wilt dit zeggen dat de AI er 80% zeker van is dat dit bijvoorbeeld een schoen is.

```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
In de ```fit()``` functie wordt meegegeven wat er getrained moet worden en ```epochs``` staat voor het aantal keer dat hij door die training gaat lopen. In dit geval dus 5 keer.
```
model.fit(train_images, train_labels, epochs=5)

prediction = model.predict(test_images)
```

## :pencil: evaluatie (optioneel)

Dit laatste deel hoeft er eigenlijk niet bij. Hier loopen we door de gegokte en 'echte' resultaten om te zien hoe de ai het heeft gedaan.

```
for i in range(5):   #loop 5 keer door een image + laat zien wat het neural network denk dat het is VS. wat het echt is
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction " + class_names[np.argmax(prediction[i])])
    plt.show()
```

# :bomb: Toepassing met custom dataset (smileydetect) 

Je zou kunnen denken dat we er bijna zijn. Als ik nu de afbeeldingen en classnames kan veranderen in mijn eigen data (smileys), dan zijn we er geraakt!

Helaas was dit niet het geval :/

#### Groot probleem nr.1 :gun:
 Toen ik op de [site van tenserflow een guide vond](https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough) en wou kijken hoe ik dit kon realiseren viel ik in een zwart gat. Ik zag termen staan waarvan ik dacht dat die nog niet konden bestaan! Na de eerste paar blokjes text te hebben gelezen kwam ik tot de conclusie dat die véél te hoog gegrepen was. Hier kwam niet alleen superveel wiskunde bij kijken, dit was precies een heel nieuwe taal opzig! 

#### Groot probleem nr.2 :gun:
 Ik wou graag een custum dataset gebruiken. Met in het groot de titel op hun website: Custom Training: walkthrough gaan ze er eigenlijk vanuit dat je alles custom kan doen buiten de dataset. Ze importeren gewoon een dataset van Google maar er staat nergens uitgelegd hoe je die dataset zelf kan maken met je eigen data.

 Dit is natuurlijk nog geen reden om op te geven. Ik googelde dan maar: 'make custom dataset' maar het had geen zin. Nergens vond ik een duidelijke guide of totorial die me kon uitleggen hoe ik een eigen .CSV file kon maken.

 > De moed zakte wat in de schoenen maar ik had een goede tutorial gevonden van hoe je via je webcam emoties kon aflezen. Helaas moest je €500 betalen voor deze dataset te kunnen gebruiken en ik ben ocharme toch zo'n arme student. :cry:


# :checkered_flag: Conclusie
AI is een hele mooie technolgie. Ik ben er zeker van dat toepassingen met AI in de toekomst meer en meer gaan toenemen. Guides en tutorials over AI zijn overal te vinden. De website van Tenserflow is zelfs voorzien met professionele video's waar experts uitleggen hoe deze technologie werkt.

Ondanks al die bronnen van informatie vind ik het bijna onmogelijk om een eigen toepassing met AI te maken. We kunnen allemaal een tutorial letterlijk volgen of git's clonen maar als je echt wil dieper duiken in de customization van AI, val je (vind ik) in een groot zwart gat. Je stoot op problemen die precies niemand anders heeft en zoek je naar een simpele functie, is het soms uren zoeken in sober uitgelegde docs. 

Komt dit omdat de technologie nog zo nieuw is, of zou het eerder liggen aan de stijle 'learn curve' van AI? Ik zou hier helaas nog geen antwoord op kunnen geven.

[handige link voor het begrijpen  - Python Neural Networks for Beginners](https://www.youtube.com/watch?v=6g4O5UOH304&t=4236s)

[Backup plan- Realtime Emotion Analysis Using Keras](https://www.youtube.com/watch?v=DtBu1u5aBsc&t=473s)
