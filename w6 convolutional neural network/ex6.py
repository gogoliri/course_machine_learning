# Author: Khoa Pham Dinh
# ID: 050359620
# Email: khoa.phamdinh@tuni.fi
import numpy as np
import tensorflow as tf
from skimage.io import imread_collection  # scikit-image
from sklearn.model_selection import train_test_split
import keras
import timeit

# Load data from class 1
# creating a collection with the available images
col_dir_1 = "GTSRB_subset_2/class1/*.jpg"
x_1 = np.array(imread_collection(col_dir_1))
y_1 = np.zeros(x_1.shape[0])

# Load data from class 2
# creating a collection with the available images
col_dir_2 = "GTSRB_subset_2/class2/*.jpg"
x_2 = np.array(imread_collection(col_dir_2))
y_2 = np.ones(x_2.shape[0])

# Combine data from 2 classes
x = np.concatenate((x_1, x_2), axis=0)
y = np.concatenate((y_1, y_2), axis=0)

# Split train and test data and shuffle
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=0)

# This dataset is imbalance
# Create weighting classes
from sklearn.utils import class_weight

class_weight = class_weight.compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y),
    y=y
)
class_weight_dict = dict(enumerate(class_weight))

# Start training
batch_size = 32  # Smaller batch size like 1 give better result, but 32 is the requirements of the exercise
epochs = 20

start = timeit.timeit()
# Choosing gpu or cpu
with tf.device('/gpu:0'):
    # Model's architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1. / 255, input_shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(10, kernel_size=(3, 3), strides=(2, 2), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(10, kernel_size=(3, 3), strides=(2, 2), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    # Compile model
    # Using stochastic gradient descent
    # loss: binary cross entropy
    model.compile(optimizer=tf.optimizers.SGD(0.01), loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train,
                        validation_data=(x_test, y_test),
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        class_weight=class_weight_dict)

end = timeit.timeit()

predict = model.evaluate(x_test, y_test)
print(f"Test loss: {predict[0]} - Test accuracy: {predict[1]}")

# Since the dataset and the model's architecture are small
# The different of GPU and CPU training time is insignificant
time = (end - start)  # Training time for 10 epochs of GPU
print(time)
