import tensorflow as tf
from keras.utils import np_utils
from matplotlib import pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

print('training images: {}'.format(x_train.shape))
print('training images: {}'.format(x_test.shape))

print(x_train[0].shape)

print(y_train)

for i in range(132,139):
     plt.subplots(figsize=(2,2))
     img = x_train[i]
     plt.imshow(img)

print(x_train[0].ndim)

print(x_train[0])

x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
X_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test=x_test/255
print(x_train)

n_classes = 10
print("shape before one-hot encoding: ", y_train.shape)
y_train = np_utils.to_categorical(y_train, n_classes)
y_test = np_utils.to_categorical(y_test, n_classes)
print("shape before one-hot encoding:", y_train.shape)



from keras.models import sequential
from keras.layers import Dense, Droupout, Conv2D, Maxpool2D, Flatten
model = sequential()
model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(Maxpool2D(pool_size=(2,2)))
model.add(Droupout(0.25))
model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(Maxpool2D(pool_size=(2,2)))
model.add(Droupout(0.25))
model.Flatten()

model.add(Dense(500, activation='relu'))
model.add(Droupout(0.4))
model.add(Dense(250, activation='relu'))
model.add(Droupout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))


classes = range(0,10)
names = ['airplane,'
         'automobile',
         'bird',
         'cat',
         'deer',
         'dog',
         'frog',
         'horse',
         'ship',
         'truck']
class_labels = dict(zip(classes, names))
batch = x_test[100:109]
labels = np.argmax(y_test[100:109],axis=-1)
predictions = model.predict(batch, verbose = 1)

print(predictions)


for image in predictions:
    print(np.sum(image))


class_result = np.argmax(predictions,axis=-1)
print(class_result)


fig, axs = plt.subplots(3, 3, figsize = (19,6))
fig.subplots_adjust(hspace = 1)
axs = axs.flatten()

for i, img in enumerate(batch):
    for key, value in class_labels.items():
        if class_result[i] == key:
            title = 'Prediction: {}\\nActual: {}'.format(class_labels[key], class_labels[labels[i]])
            axs[i].set_title(title)
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)

    axs[i].imshow(img)

plt.show()








