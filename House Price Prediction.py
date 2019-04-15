#!/usr/bin/env python
# coding: utf-8

# # Load dataset, split into train and test data and check the train data shape

# In[2]:


from keras.datasets import boston_housing
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

x_train.shape


# # Start building the model graph, add input and output layers

# In[3]:


basic_model = Sequential()

basic_model.add(Dense(units=13, input_dim = 13, kernel_initializer = 'normal'))

basic_model.add(Dense(units = 1, kernel_initializer = 'normal', activation='linear'))


# # Choose loss function and optimizer, compile model

# In[4]:


basic_model.compile(optimizer=SGD(lr = 0.000001), loss='mean_absolute_error', metrics=['accuracy', "mse"])

basic_model.summary()


# # Improve the model by adding one more hidden layer

# In[5]:


advanced_model = Sequential()

advanced_model.add(Dense(units=32, input_dim = 13, kernel_initializer = 'truncated_normal'))

advanced_model.add(Dense(units = 16, kernel_initializer = 'truncated_normal'))

advanced_model.add(Dense(units = 1, kernel_initializer = 'truncated_normal'))

advanced_model.compile(optimizer="adam", loss='mean_absolute_error', metrics=['accuracy'])

advanced_model.summary()


# # Check training data

# In[6]:


print("features of the first house in the dataset:", x_train[0])
print("first house in the dataset's price in thousands:",y_train[0])


# # Train the model

# In[7]:


advanced_model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.1)


# # Model evaluation function

# In[8]:


def test_model(model, metric="accuracy"):
    evaluation = model.evaluate(x_test, y_test, verbose=0)
    
    print("-------------------------------------")
    print("Loss over the test dataset: %.2f" % (evaluation[0]))
    print("-------------------------------------")
    if metric == "accuracy":
        print("Accuarcy: %.2f" % (evaluation[1]))
    elif metric == "mae":
        print("Mean absolute error: %.2f" % (evaluation[1]))


# # Test the basic model

# In[10]:


test_model(basic_model)


# # Test the advanced model

# In[11]:


test_model(advanced_model)


# # Define function to check actual house predictions

# In[12]:


def check_predictions(model):
    train_houses=[x_train[0:1], x_train[10:11], x_train[200:201]]
    train_actual_prices = [y_train[0:1], y_train[10], y_train[200]]
    
    print("\n")
    print("Training set points:")
    
    for house, price in zip(train_houses, train_actual_prices):
        prediction = model.predict(house)
        print(f"Predicted price: {prediction}, Actual price: {price}")
    
    test_houses=[x_test[1:2], x_test[50:51], x_test[100:101]]
    test_actual_prices = [y_test[1], y_test[50], y_test[100]]
    
    print("\n")
    print("Testing set points:")
    
    for house, price in zip(test_houses, test_actual_prices):
        prediction = model.predict(house)
        print(f"Predicted price: {prediction}, Actual price: {price}")


# # Demonstrate slicing

# In[13]:


print("sliced: ", x_train[0:1])
print("not sliced: ", x_train[0])


# # Check concrete predictions for our advanced model

# In[14]:


check_predictions(advanced_model)


# # Improve the model

# In[15]:


improved_model = Sequential()
# add neuron units and add nonlinear activation function
improved_model.add(Dense(units=64, input_dim = 13, kernel_initializer = 'truncated_normal', activation="relu")) 

improved_model.add(Dense(units = 32, kernel_initializer = 'truncated_normal', activation="relu"))

improved_model.add(Dense(units = 1, kernel_initializer = 'truncated_normal'))

# mean squared error penalizes larger difference, works better
improved_model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mae']) 
# changes batch size = 32 which works better,both loss and validation loss has decreased at epochs=150
improved_model.fit(x_train, y_train, epochs=150, batch_size=32, verbose=1, validation_split=0.2)

check_predictions(improved_model)


# In[16]:


test_model(improved_model, metric = "mae")


# In[ ]:




