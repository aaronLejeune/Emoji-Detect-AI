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
#### Imports
1. Importeren van de verschillende Libs

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

2. Valideren of de import gelukt is:

Hier printen we gewoon de versie uit van de libs die we gebruiken. Dit is een simpele manier om te valideren dat de installatie geslaagd is.

```
print(tf.__version__)
print(np.__version__)
```

3. Importeren + laden test datasets (keras fashion_mnist)

Zoals eerder al besproken zorgt Keras voor datasets. ['Op de website van Keras'](https://keras.io/examples/mnist_dataset_api/) kan je kiezen tussen allemaal verschillende datasets en tutorials van hoe je API het best kan gebruiken. In het stukje code hieronder importeren we dus 1 bepaalde dataset en delen we die dataset op in TRAIN en TEST data.

```
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()
```

TRAIN data is meestal ongeveer 3/4de van je hele dataset en zoals het woord letterlijk zegt, ge je je AI trainen om uiteindelijk een test te gaan doen. De TEST data wordt dan uiteindelijk gebruikt om de effectieve test af te leggen (met de resterende 1/4de van de dataset). Je kan het een beetje bekijken als leren voor een examen. Je maakt allemaal oefeningen (TRAIN data) om je voor te bereiden op het examen (TEST data). Hoe beter je oefeningen (TRAIN data) zijn, hoe waarschijnlijker het is dat het examen (TEST data) ook goed zal verlopen.

Het is belangrijk om TRAIN en TEST data uit een te haen. Stel je voor dat je leert voor een examen. De docent heeft perongeluk de echte examenvragen tussen de oefening gestoken. Een AI mag niet weten wat de examenvragen kunnen zijn, anders zou het natuurlijk veel hogere resulataten halen dan het eigenlijk kan.

#### Label Data
```
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
```


# Conclusie
AI is een hele mooie technolgie. Ik ben er zeker van dat toepassingen met AI in de toekomst meer en meer gaan toenemen. Guides en tutorials over AI zijn overal te vinden. De website van Tenserflow is zelfs voorzien met professionele video's waar experts uitleggen hoe deze technologie werkt.

Ondanks al die bronnen van informatie vind ik het bijna onmogelijk om een eigen toepassing met AI te maken. We kunnen allemaal een tutorial letterlijk volgen of git's clonen maar als je echt wil dieper duiken in de customization van AI, val je (vind ik) in een groot zwart gat. Je stoot op problemen die precies niemand anders heeft en zoek je naar een simpele functie, is het soms uren zoeken in sober uitgelegde docs. 

Komt dit omdat de technologie nog zo nieuw is, of zou het eerder liggen aan de stijle 'learn curve' van AI? Ik zou hier helaas nog geen antwoord op kunnen geven.
