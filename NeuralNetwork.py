#This script describes a Neural Network with one hidden layer and 8 nodes in this hidden layer

import csv
import tensorflow as tf

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Read data in from file
with open("banknotes.csv") as f:
    reader = csv.reader(f)
    next(reader)

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[:4]],
            "label": 1 if row[4] == "0" else 0
        })

# Separate data into training and testing groups
evidence = [row["evidence"] for row in data]
labels = [row["label"] for row in data]
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4) #this function only split the data into training and testing data

# Create a neural network
model = tf.keras.models.Sequential() #sequential neural network

# Add a hidden layer with 8 units, with ReLU activation
model.add(tf.keras.layers.Dense(8, input_shape=(4,), activation="relu")) #this layer has 8 nodes(units). we have 4 units in the input. The activacion function is relu

# Add output layer with 1 unit, with sigmoid activation
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))# we create only one unit (rainy or not rainy). Activation function is "sigmoid activation function"

# Train neural network
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)#parameters of the model: how you want to optimizate = "adam", give a loss function = "binary_crossentropy", how to evaluate the model, we care about accuracy in this case. 
model.fit(X_training, y_training, epochs=50) #model is made of layers and .fit train the model.

# Evaluate how well model performs
#model.evaluate(X_testing, y_testing, verbose="auto") #We test our model
predictions = model.predict(X_testing)


fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")


#Relation between two conditions for a rainy day.
data1= [i[1] for i in X_training]
data2= [i[2] for i in X_training]
data3= [i[3] for i in X_training]
#color = [i[0] for i in y_training]
predict1= [i[1] for i in X_testing]
predict2= [i[2] for i in X_testing]
predict3= [i[3] for i in X_testing]
#print(X_training)
#print(y_training)
#print(y_testing)

## Creating plot
ax.scatter3D(data1, data2, data3,  c=y_training,cmap="bwr",marker=1)
ax.scatter3D(predict1, predict2, predict3,  c=predictions, cmap="PiYG",marker=2)
plt.title(f"{type(model).__name__}")
#ax.set_xlabel('variance', fontweight ='bold') #0
ax.set_xlabel('skewness', fontweight ='bold')#1
ax.set_ylabel('curtosis', fontweight ='bold')#2
ax.set_zlabel('entropy', fontweight ='bold')#3
plt.show()
