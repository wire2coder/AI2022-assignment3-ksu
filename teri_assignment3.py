import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Download and preprocess the CIFAR-10 dataset
# This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories.
# We normalize training data, since we can make sure that the various features have similar value ranges so that gradient descents can converge faster.

# Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images_norm, test_images_norm = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

asdf=[]
plt.figure(figsize=(10,10))

for i in range(25): # mod25, {0,1,2...24}
    plt.subplot(5,5,i+1) # 5 row, 5 columns, and 'something'
    plt.xticks([]) # no 'ticks'
    plt.yticks([])
    plt.grid(False) # don't show background 'grid lines'
    
    plt.imshow(train_images_norm[i])

    # The CIFAR labels happen to be arrays, 
    # which is why you need the extra index
    # asdf.append(train_labels[i][0]) # >> [6, 9, 9, 4, 1, 1, 2, 7, 8, 3, 4, 7 ...]
    name1 = train_labels[i][0]
    plt.xlabel(class_names[name1])
    
plt.show()

# build your own VGG16 with tensorflow.keras
model1 = models.Sequential()

# 'filtering' stuff
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model1.add(layers.Conv2D(32, (3, 3), activation='relu'))
model1.add(layers.MaxPooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model1.add(layers.MaxPooling2D((2, 2)))
# model1.add(layers.Convolution2D(32, 3, 3, activation='relu'))
model1.add(layers.MaxPooling2D((2,2)))

# 'classifying' stuff
model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10)) # the last layer have to 'match' the 'class list size' (10)

model1.summary()

# 'Train' (CREATE) the CNN 'model' (the equation)
# Here, we use SGD optimizer and cross entropy loss function.

model1.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history1 = model1.fit(train_images_norm, train_labels, 
           epochs=10, 
           validation_data=(test_images_norm, test_labels) )

what1 = history1.history['loss'][0]
print(what1)


x_data1 = history1.history['accuracy']
x_data2 = history1.history['val_accuracy']
x_data3 = history1.history['loss']

plt.plot( x_data1, label='training_data accuracy')
plt.plot( x_data2, label='validation_data accuracy')

plt.xlabel('Epoc / Iteration')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.ylim(0.0, 0.9)

plt.plot( x_data3, label='Loss* value (training data)')
plt.xlabel('Epoc / Iteration')
plt.ylabel('Value of *Loss*')

# 'looking' at what is INSIDE 'history1'
# history1.

# 'Evaluate' the 'model' with the data in 'group test'
test_loss_value, test_accuracy_value = model1.evaluate(test_images_norm, test_labels)
print(f"Accuracy value for 'Test Data': {test_accuracy_value}\n'Loss value' for 'Test Data': {test_loss_value}")