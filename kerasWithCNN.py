from __future__ import print_function
import numpy as np
import math
import keras
from keras.datasets import mnist
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# Define hyperparameter values
batch_size = 32 # in each iteration, we consider 32 training examples at once
epochs = 15  # we iterate 10 times over the entire training set
kernel_size = 5  # we will use 3x3 kernels throughout
pool_size = 2  # we will use 2x2 pooling throughout
conv_depth_1 = 32  # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64  # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25  # dropout after pooling with probability 0.25
drop_prob_2 = 0.5  # dropout in the FC layer with probability 0.5
hidden_size = 128  # the FC layer will have this number of neurons 128
height, width, depth = 28, 28, 1  # MNIST images are 28x28 and greyscale
num_classes = 10  # there are 10 different classes in the training set
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model_run2.h5'


# Read the data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # fetch mnist data
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train_m = keras.utils.to_categorical(y_train, num_classes)  # One-hot encode the labels
y_test_m = keras.utils.to_categorical(y_test, num_classes)  # One-hot encode the labels

#copying data for plotting
test=x_test.copy()

# Insert single channel dimension (mnist is grey-scale and not RGB)
if keras.backend.image_data_format() == 'channels_first':
    x_train = np.expand_dims(x_train, 1)
    x_test = np.expand_dims(x_test, 1)
else:
    x_train = np.expand_dims(x_train, 3)
    x_test = np.expand_dims(x_test, 3)

# Convert to floating point and rescale to [0 1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train / 255
x_test = x_test / 255

# Define a new Model
inp = Input(shape=(height, width, depth))
# Conv [32] -> Conv [32] -> Pool (with dropout on the pooling layer)
conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out)  # To define a model, just specify its input and output layers
model.compile(loss='categorical_crossentropy',  # using the cross-entropy loss function
              optimizer=keras.optimizers.Adadelta(),  # using the Adam optimiser "adam"
              metrics=['accuracy'])
model.fit(x_train, y_train_m,                # Train the model using the training set
          batch_size=batch_size, epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_m))


# Run the trained model on test data, outputting a probability distribution from each image
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)  # Get the index of most probable class
y_true = y_test_m
y_true = np.argmax(y_true, axis=1)

correct_indices = np.nonzero(y_pred == y_true)[0]
incorrect_indices = np.nonzero(y_pred != y_true)[0]

print(len(correct_indices), "no . of correct samples")
print(len(incorrect_indices), "no . of incorrect samples")
print(str((len(correct_indices)/float(len(y_test)))*100)+'%', "Correct Percentage")

# plot incorrect images
plt.figure()
num_iteration = math.floor(len(incorrect_indices)/25) + 1
for k in range(int(num_iteration)):
    for i, incorrect in enumerate(incorrect_indices[k*25:(k+1)*25]):

        # plt.subplot(sub_row + 1, sub_column + 1, i+1)
        plt.subplot(5, 5, i + 1)
        plt.imshow(test[incorrect], cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(y_pred[incorrect], y_test[incorrect]), fontsize=6).set_position([.5, 0.96])
        # plt.title("Incorrect images")
        plt.axis("off")
    plt.show()


# Plot the confusion matrix
def plot_confusion_matrix(y_true,y_pred):
    cm_array = confusion_matrix(y_true,y_pred)
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array, interpolation='nearest', cmap=plt.cm.jet)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=10)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10).set_position([.5, 0.5])
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size
    print(classification_report(y_true, y_pred))


plot_confusion_matrix(y_true, y_pred)
plt.show()

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

