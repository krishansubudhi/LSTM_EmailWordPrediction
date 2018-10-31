# LSTM_EmailWordPrediction
LSTM model to predict the next word while composing emails

## Dependencies
  - Keras
  - Numpy
  - Tensorflow

# Steps to run the model
## 1. Export sent emails from outlook
![Alt text](images/Outlook_export.png?raw=true "Export outlook sent mails step 1")
![Alt text](images/Outlook_file_save.PNG?raw=true "Export outlook sent mails step 2")

## 2. Change these fields to appropriate values as per your name and signature.

```python
for line in open('krishan_sent_outlook.CSV','r'):
    if line.startswith('From: Krishan'):
        include=True
        continue
    elif line.startswith('From:') or line.startswith('Thanks and Regards'):
```


## 3. Run the model.
The model is a single layer lstm model with a word embedding layer preceeding it.
```python
from keras.models import Sequential
from keras import layers
from keras import losses
from keras import metrics

def getmodel():
    model = Sequential()
    model.add(layers.Embedding(1000, 32))
    model.add(layers.LSTM(32))
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam',loss=losses.categorical_crossentropy,metrics=[metrics.categorical_accuracy])

    model.summary()
    return model
```

I chose **adam** instead of **rmsprop** this time. 'adam' worked exceptionally well. Not only the training was faster, but the accuracy was improving rapidly which was not happening with rmsprop. The accuracy with rmsprop was 45% at 100 epoch and with adam it was 75%.

```python
model = getmodel()
history = model.fit(x_train_partial,
          y_train_partial,
          epochs=100,
          batch_size=32,
          validation_split=0.2)
```
## 4. Evaluation
The data is still underfitting. More epochs with a some dropout or regularization will help achieve accuracy >80%.
Current accuracy in test set was 75%
```
model.evaluate(x_test,y_test)
[1.7531049308776856, 0.753]
```

## 5. Testing
This function does all the necesary conversions required to test our own texts.
```python
def predict(input_line, count ):
    input_line_clean = input_line.strip().lower()
    predicted_words = []
    input_x= tokenizer.texts_to_sequences([input_line_clean])[0]
    for _ in range(0,count):
        model_input_x = sequence.pad_sequences([input_x],maxlen=maxlen)
        output_y = model.predict(model_input_x)
        input_x.append(np.argmax(output_y))
        predicted_words.append((wordfromonehot(output_y), np.max(output_y)))
    print ("input = {0}, predicted = {1}".format(input_line,predicted_words))
```
## 6. Sample predictions
Example prediction of next words and their confidence level.
```
1. predict("Can you please",10)
input = Can you please, predicted = [('check', 0.23085266), ('the', 0.35568327), ('quality', 0.8381603), ('and', 0.9865416), ('let', 0.96286523), ('me', 0.9975465), ('know', 0.99354774), ('your', 0.7261728), ('feedback', 0.34738642), ('and', 0.851428)]

2. input = but we have no solution, predicted_next = [('to', 0.8416771), ('that', 0.4860347), ('right', 0.8375561), ('now', 0.9249746)]
3. input = on 6th april let s, predicted_next = [('catch', 0.9530264), ('up', 0.997543), ('after', 0.95520276), ('that', 0.42910975)]

```
