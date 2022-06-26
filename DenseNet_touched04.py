from keras.models import Model, load_model
from keras.layers import Conv1D, Dense, Flatten, Activation, Add, Concatenate, Input
from keras.layers import BatchNormalization, Dropout
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, ZeroPadding1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical,plot_model
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import io
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical



X = io.loadmat('../matlab_data/mitFinalPsNB.mat')
X = X['mitFinalPsNB']
df = pd.DataFrame(data=X)

Y = np.array(df[428].values).astype(np.int8)
X = np.array(df[list(range(428))].values)[..., np.newaxis]

oneHot = LabelEncoder()
oneHot.fit(Y)
Y = oneHot.transform(Y)

X = X.reshape(-1, 428, 1)
Y = to_categorical(Y,5)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)

# hyperparameter
growth_k = 32
nb_block = 2
init_learning_rate = 1e-4
epsilon = 1e-8
dropout_rate = 0.2

# momentum optimizer
nesterov_momentum = 0.9
weight_decay = 1e-4

#Label, batch_size
calss_num = 5
batch_size = 50

total_epochs = 100

# bottleneck
def bottleneck_layer(inputs, growth_k=32):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv1D(4*growth_k, 1)(x)
    x = Dropout(dropout_rate)(x)
        
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(growth_k, 3, padding='same')(x)
    x = Dropout(dropout_rate)(x)
        
    return x

# transition
def transition_layer(inputs, growth_k=32):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = Conv1D(growth_k, 1)(x)
    
    in_channel = x.shape[-1]
    
    x = Conv1D(in_channel*0.5,1)(x)
    x = Dropout(dropout_rate)(x)
    x = AveragePooling1D(2,2)(x)
    
    return x

# dense_block
def dense_block(inputs, nb_layers):
    layers_concat=[]
    layers_concat.append(inputs)
    
    x = bottleneck_layer(inputs)
    
    layers_concat.append(x)
    
    for i in range(nb_layers-1):
        x = Concatenate(axis=2)(layers_concat)
        x = bottleneck_layer(x)
        layers_concat.append(x)
    
    x = Concatenate(axis=2)(layers_concat)
    
    return x

# classification_layer
def classify_layer(inputs):
    x = BatchNormalization()(inputs)
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    output_layer = Dense(1000, activation='softmax')(x)
    
    return output_layer

# input_layer 정의
input_layer = Input(shape=(428,1))

# DenseNet 완성!
def Dense_net():
    growth_k = 32
    x = Conv1D(2*growth_k, 7, 2)(input_layer)
    x = MaxPooling1D(3,2)(x)
    
    for i in range(nb_block):
        x = dense_block(x, 6)
        x = transition_layer(x)
        x = dense_block(x, 12)
        x = transition_layer(x)  
        x = dense_block(x, 64)
        x = transition_layer(x)
        
    x = dense_block(x, 32)(x)
    output_layer = classify_layer(x)
    
    return output_layer

denseNet = Model(input_layer)
denseNet.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.001)

denseNet.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
denseNet.summary()

callbacks = [ModelCheckpoint(filepath = "./checkpoint",
                             monitor = 'val_accuracy',
                             save_best_only = True,
                             save_weights_only = False,
                             mode = 'max',
                             verbose = 1)]

history = denseNet.fit(X_train, Y_train,
                    batch_size = 64,
                    epochs = 100,
                    verbose = 1,
                    callbacks = callbacks,
                    validation_data = (X_val, Y_val))



Y_pred = denseNet.predict(X_val)
Y_true = np.argmax(Y_val, axis=1)
Y_pred = np.argmax(Y_pred, axis=1)


plt.figure(figsize=(15,5), facecolor = 'white')
plt.subplot(121)
plt.title("Accuracy")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accurcay')
plt.legend(['train', 'test'])
plt.subplot(122)
plt.title("Loss")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'test'])
plt.show()

score = denseNet.evaluate(X_val, Y_val, verbose = 0)

print('Test loss:', round(score[0], 3))
print('Test accuracy:', round(score[1], 3))
print(classification_report(Y_true, Y_pred))

x_lab = ['N','Q','S','V','F']
y_lab = ['N','Q','S','V','F']

conf_matrix = confusion_matrix(Y_true, Y_pred)
conf_matrix_f = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(14,5), facecolor='white')
plt.subplot(121)
plt.title("Confusion Matrix: num")
aa = sns.heatmap(conf_matrix,
                 xticklabels = x_lab,
                 yticklabels = y_lab,
                 annot=True,
                 fmt=".0f",
                 cmap=plt.cm.binary,
                facecolor='white')

plt.yticks(rotation='horizontal')
plt.ylabel('True')
plt.xlabel('Predict')
plt.subplot(122)
plt.title("Confusion Matrix: ratio")

aa = sns.heatmap(conf_matrix_f,
                 xticklabels = x_lab,
                 yticklabels = y_lab,
                 annot=True,
                 fmt=".3f",
                 cmap=plt.cm.binary,
                facecolor='white')
plt.yticks(rotation='horizontal')
plt.ylabel('True')
plt.xlabel('Predict')
plt.tight_layout()