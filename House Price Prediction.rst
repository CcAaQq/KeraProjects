
Set random seeds and import modules
===================================

.. code:: ipython3

    from numpy.random import seed
    seed(1337)
    from tensorflow import set_random_seed
    set_random_seed(1337)
    import keras
    from keras.models import Sequential
    from keras.layers import Activation, Dense
    from keras.optimizers import SGD


.. parsed-literal::

    Using TensorFlow backend.
    

Load dataset, split into train and test data and check the train data shape
===========================================================================

.. code:: ipython3

    from keras.datasets import boston_housing
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    
    x_train.shape




.. parsed-literal::

    (404, 13)



Start building the model graph, add input and output layers
===========================================================

.. code:: ipython3

    basic_model = Sequential()
    
    basic_model.add(Dense(units = 13, input_dim = 13, kernel_initializer = 'normal'))
    
    basic_model.add(Dense(units = 1, kernel_initializer = 'normal', activation = 'linear'))


.. parsed-literal::

    WARNING:tensorflow:From D:\Anaconda\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    

Choose loss function and optimizer, compile model
=================================================

.. code:: ipython3

    basic_model.compile(optimizer = SGD(lr = 0.000001), loss = 'mean_absolute_error', metrics = ['accuracy', "mse"])
    
    basic_model.summary()


.. parsed-literal::

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 13)                182       
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 14        
    =================================================================
    Total params: 196
    Trainable params: 196
    Non-trainable params: 0
    _________________________________________________________________
    

Improve the model by adding one more hidden layer
=================================================

.. code:: ipython3

    advanced_model = Sequential()
    
    advanced_model.add(Dense(units = 32, input_dim = 13, kernel_initializer = 'truncated_normal'))
    
    advanced_model.add(Dense(units = 16, kernel_initializer = 'truncated_normal'))
    
    advanced_model.add(Dense(units = 1, kernel_initializer = 'truncated_normal'))
    
    advanced_model.compile(optimizer = "adam", loss = 'mean_absolute_error', metrics = ['accuracy'])
    
    advanced_model.summary()


.. parsed-literal::

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_3 (Dense)              (None, 32)                448       
    _________________________________________________________________
    dense_4 (Dense)              (None, 16)                528       
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 17        
    =================================================================
    Total params: 993
    Trainable params: 993
    Non-trainable params: 0
    _________________________________________________________________
    

Check training data
===================

.. code:: ipython3

    print("features of the first house in the dataset:", x_train[0])
    print("first house in the dataset's price in thousands:", y_train[0])


.. parsed-literal::

    features of the first house in the dataset: [  1.23247   0.        8.14      0.        0.538     6.142    91.7
       3.9769    4.      307.       21.      396.9      18.72   ]
    first house in the dataset's price in thousands: 15.2
    

Train the model
===============

.. code:: ipython3

    advanced_model.fit(x_train, y_train, epochs = 100, batch_size = 64, verbose = 1, validation_split = 0.1)


.. parsed-literal::

    WARNING:tensorflow:From D:\Anaconda\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 363 samples, validate on 41 samples
    Epoch 1/100
    363/363 [==============================] - 1s 3ms/step - loss: 19.4760 - acc: 0.0000e+00 - val_loss: 16.3330 - val_acc: 0.0000e+00
    Epoch 2/100
    363/363 [==============================] - 0s 59us/step - loss: 15.4819 - acc: 0.0000e+00 - val_loss: 11.7637 - val_acc: 0.0000e+00
    Epoch 3/100
    363/363 [==============================] - 0s 52us/step - loss: 11.0846 - acc: 0.0000e+00 - val_loss: 8.4541 - val_acc: 0.0244
    Epoch 4/100
    363/363 [==============================] - 0s 113us/step - loss: 9.0958 - acc: 0.0055 - val_loss: 7.9600 - val_acc: 0.0244
    Epoch 5/100
    363/363 [==============================] - 0s 66us/step - loss: 9.4670 - acc: 0.0000e+00 - val_loss: 7.3666 - val_acc: 0.0244
    Epoch 6/100
    363/363 [==============================] - 0s 52us/step - loss: 8.2713 - acc: 0.0055 - val_loss: 6.9127 - val_acc: 0.0000e+00
    Epoch 7/100
    363/363 [==============================] - 0s 80us/step - loss: 7.7265 - acc: 0.0083 - val_loss: 6.5761 - val_acc: 0.0000e+00
    Epoch 8/100
    363/363 [==============================] - 0s 55us/step - loss: 7.1934 - acc: 0.0000e+00 - val_loss: 5.8293 - val_acc: 0.0000e+00
    Epoch 9/100
    363/363 [==============================] - 0s 49us/step - loss: 6.7685 - acc: 0.0110 - val_loss: 5.2902 - val_acc: 0.0244
    Epoch 10/100
    363/363 [==============================] - 0s 47us/step - loss: 6.5028 - acc: 0.0083 - val_loss: 5.0997 - val_acc: 0.0000e+00
    Epoch 11/100
    363/363 [==============================] - 0s 38us/step - loss: 6.3201 - acc: 0.0083 - val_loss: 5.0552 - val_acc: 0.0000e+00
    Epoch 12/100
    363/363 [==============================] - 0s 41us/step - loss: 6.2178 - acc: 0.0110 - val_loss: 4.8739 - val_acc: 0.0000e+00
    Epoch 13/100
    363/363 [==============================] - 0s 44us/step - loss: 6.1344 - acc: 0.0110 - val_loss: 4.8573 - val_acc: 0.0244
    Epoch 14/100
    363/363 [==============================] - 0s 41us/step - loss: 6.1014 - acc: 0.0055 - val_loss: 4.7606 - val_acc: 0.0244
    Epoch 15/100
    363/363 [==============================] - 0s 41us/step - loss: 6.0878 - acc: 0.0165 - val_loss: 4.9397 - val_acc: 0.0244
    Epoch 16/100
    363/363 [==============================] - 0s 47us/step - loss: 6.0590 - acc: 0.0165 - val_loss: 4.7149 - val_acc: 0.0244
    Epoch 17/100
    363/363 [==============================] - 0s 41us/step - loss: 6.0327 - acc: 0.0083 - val_loss: 4.8296 - val_acc: 0.0000e+00
    Epoch 18/100
    363/363 [==============================] - 0s 148us/step - loss: 6.0277 - acc: 0.0138 - val_loss: 4.8611 - val_acc: 0.0000e+00
    Epoch 19/100
    363/363 [==============================] - 0s 41us/step - loss: 5.9722 - acc: 0.0083 - val_loss: 4.6558 - val_acc: 0.0244
    Epoch 20/100
    363/363 [==============================] - 0s 110us/step - loss: 5.9582 - acc: 0.0083 - val_loss: 4.8309 - val_acc: 0.0000e+00
    Epoch 21/100
    363/363 [==============================] - 0s 58us/step - loss: 5.9401 - acc: 0.0165 - val_loss: 4.7072 - val_acc: 0.0000e+00
    Epoch 22/100
    363/363 [==============================] - 0s 41us/step - loss: 5.9330 - acc: 0.0110 - val_loss: 4.7840 - val_acc: 0.0000e+00
    Epoch 23/100
    363/363 [==============================] - 0s 49us/step - loss: 5.9041 - acc: 0.0138 - val_loss: 4.6305 - val_acc: 0.0000e+00
    Epoch 24/100
    363/363 [==============================] - 0s 52us/step - loss: 5.8945 - acc: 0.0165 - val_loss: 4.7297 - val_acc: 0.0000e+00
    Epoch 25/100
    363/363 [==============================] - 0s 63us/step - loss: 5.8801 - acc: 0.0138 - val_loss: 4.6038 - val_acc: 0.0000e+00
    Epoch 26/100
    363/363 [==============================] - 0s 55us/step - loss: 5.8606 - acc: 0.0110 - val_loss: 4.7531 - val_acc: 0.0000e+00
    Epoch 27/100
    363/363 [==============================] - 0s 96us/step - loss: 5.8358 - acc: 0.0193 - val_loss: 4.6258 - val_acc: 0.0000e+00
    Epoch 28/100
    363/363 [==============================] - 0s 74us/step - loss: 5.8204 - acc: 0.0220 - val_loss: 4.7563 - val_acc: 0.0000e+00
    Epoch 29/100
    363/363 [==============================] - 0s 63us/step - loss: 5.8194 - acc: 0.0220 - val_loss: 4.6116 - val_acc: 0.0000e+00
    Epoch 30/100
    363/363 [==============================] - 0s 49us/step - loss: 5.8155 - acc: 0.0165 - val_loss: 4.6499 - val_acc: 0.0000e+00
    Epoch 31/100
    363/363 [==============================] - 0s 49us/step - loss: 5.7766 - acc: 0.0193 - val_loss: 4.6057 - val_acc: 0.0000e+00
    Epoch 32/100
    363/363 [==============================] - 0s 38us/step - loss: 5.7788 - acc: 0.0165 - val_loss: 4.6376 - val_acc: 0.0000e+00
    Epoch 33/100
    363/363 [==============================] - 0s 41us/step - loss: 5.7702 - acc: 0.0165 - val_loss: 4.5785 - val_acc: 0.0000e+00
    Epoch 34/100
    363/363 [==============================] - 0s 41us/step - loss: 5.7474 - acc: 0.0193 - val_loss: 4.5631 - val_acc: 0.0000e+00
    Epoch 35/100
    363/363 [==============================] - 0s 49us/step - loss: 5.7322 - acc: 0.0193 - val_loss: 4.6368 - val_acc: 0.0000e+00
    Epoch 36/100
    363/363 [==============================] - 0s 71us/step - loss: 5.7388 - acc: 0.0165 - val_loss: 4.5904 - val_acc: 0.0000e+00
    Epoch 37/100
    363/363 [==============================] - 0s 63us/step - loss: 5.7114 - acc: 0.0165 - val_loss: 4.6517 - val_acc: 0.0000e+00
    Epoch 38/100
    363/363 [==============================] - 0s 49us/step - loss: 5.6890 - acc: 0.0193 - val_loss: 4.5873 - val_acc: 0.0244
    Epoch 39/100
    363/363 [==============================] - 0s 52us/step - loss: 5.6746 - acc: 0.0275 - val_loss: 4.6733 - val_acc: 0.0000e+00
    Epoch 40/100
    363/363 [==============================] - 0s 52us/step - loss: 5.6616 - acc: 0.0193 - val_loss: 4.6441 - val_acc: 0.0000e+00
    Epoch 41/100
    363/363 [==============================] - 0s 49us/step - loss: 5.6382 - acc: 0.0193 - val_loss: 4.5894 - val_acc: 0.0244
    Epoch 42/100
    363/363 [==============================] - 0s 47us/step - loss: 5.6295 - acc: 0.0165 - val_loss: 4.5525 - val_acc: 0.0244
    Epoch 43/100
    363/363 [==============================] - 0s 77us/step - loss: 5.6534 - acc: 0.0083 - val_loss: 4.6188 - val_acc: 0.0244
    Epoch 44/100
    363/363 [==============================] - 0s 47us/step - loss: 5.6632 - acc: 0.0165 - val_loss: 4.4967 - val_acc: 0.0244
    Epoch 45/100
    363/363 [==============================] - 0s 44us/step - loss: 5.6806 - acc: 0.0055 - val_loss: 4.6704 - val_acc: 0.0244
    Epoch 46/100
    363/363 [==============================] - 0s 49us/step - loss: 5.5846 - acc: 0.0165 - val_loss: 4.4033 - val_acc: 0.0244
    Epoch 47/100
    363/363 [==============================] - 0s 49us/step - loss: 5.5573 - acc: 0.0165 - val_loss: 4.6370 - val_acc: 0.0244
    Epoch 48/100
    363/363 [==============================] - 0s 66us/step - loss: 5.5746 - acc: 0.0248 - val_loss: 4.4808 - val_acc: 0.0244
    Epoch 49/100
    363/363 [==============================] - 0s 44us/step - loss: 5.5205 - acc: 0.0083 - val_loss: 4.6498 - val_acc: 0.0244
    Epoch 50/100
    363/363 [==============================] - 0s 49us/step - loss: 5.5128 - acc: 0.0138 - val_loss: 4.3211 - val_acc: 0.0244
    Epoch 51/100
    363/363 [==============================] - 0s 41us/step - loss: 5.5203 - acc: 0.0248 - val_loss: 4.5832 - val_acc: 0.0244
    Epoch 52/100
    363/363 [==============================] - 0s 44us/step - loss: 5.5147 - acc: 0.0165 - val_loss: 4.3736 - val_acc: 0.0000e+00
    Epoch 53/100
    363/363 [==============================] - 0s 69us/step - loss: 5.4574 - acc: 0.0220 - val_loss: 4.8472 - val_acc: 0.0244
    Epoch 54/100
    363/363 [==============================] - 0s 47us/step - loss: 5.4821 - acc: 0.0055 - val_loss: 4.2722 - val_acc: 0.0488
    Epoch 55/100
    363/363 [==============================] - 0s 47us/step - loss: 5.4741 - acc: 0.0165 - val_loss: 4.6613 - val_acc: 0.0488
    Epoch 56/100
    363/363 [==============================] - 0s 44us/step - loss: 5.5240 - acc: 0.0055 - val_loss: 4.3199 - val_acc: 0.0000e+00
    Epoch 57/100
    363/363 [==============================] - 0s 41us/step - loss: 5.4498 - acc: 0.0193 - val_loss: 4.4448 - val_acc: 0.0000e+00
    Epoch 58/100
    363/363 [==============================] - 0s 41us/step - loss: 5.4316 - acc: 0.0138 - val_loss: 4.4583 - val_acc: 0.0000e+00
    Epoch 59/100
    363/363 [==============================] - 0s 44us/step - loss: 5.3661 - acc: 0.0083 - val_loss: 4.4638 - val_acc: 0.0000e+00
    Epoch 60/100
    363/363 [==============================] - 0s 47us/step - loss: 5.3526 - acc: 0.0165 - val_loss: 4.6575 - val_acc: 0.0488
    Epoch 61/100
    363/363 [==============================] - 0s 49us/step - loss: 5.3441 - acc: 0.0110 - val_loss: 4.4425 - val_acc: 0.0000e+00
    Epoch 62/100
    363/363 [==============================] - 0s 69us/step - loss: 5.3081 - acc: 0.0138 - val_loss: 4.4418 - val_acc: 0.0000e+00
    Epoch 63/100
    363/363 [==============================] - 0s 49us/step - loss: 5.2856 - acc: 0.0138 - val_loss: 4.5548 - val_acc: 0.0000e+00
    Epoch 64/100
    363/363 [==============================] - 0s 47us/step - loss: 5.2972 - acc: 0.0165 - val_loss: 4.7408 - val_acc: 0.0244
    Epoch 65/100
    363/363 [==============================] - 0s 44us/step - loss: 5.2679 - acc: 0.0110 - val_loss: 4.4558 - val_acc: 0.0000e+00
    Epoch 66/100
    363/363 [==============================] - 0s 49us/step - loss: 5.2141 - acc: 0.0193 - val_loss: 4.5887 - val_acc: 0.0000e+00
    Epoch 67/100
    363/363 [==============================] - 0s 44us/step - loss: 5.1917 - acc: 0.0110 - val_loss: 4.4285 - val_acc: 0.0000e+00
    Epoch 68/100
    363/363 [==============================] - 0s 38us/step - loss: 5.1757 - acc: 0.0193 - val_loss: 4.4440 - val_acc: 0.0000e+00
    Epoch 69/100
    363/363 [==============================] - 0s 44us/step - loss: 5.1746 - acc: 0.0110 - val_loss: 4.6843 - val_acc: 0.0000e+00
    Epoch 70/100
    363/363 [==============================] - 0s 63us/step - loss: 5.2259 - acc: 0.0083 - val_loss: 4.2294 - val_acc: 0.0488
    Epoch 71/100
    363/363 [==============================] - 0s 47us/step - loss: 5.1430 - acc: 0.0165 - val_loss: 4.7810 - val_acc: 0.0244
    Epoch 72/100
    363/363 [==============================] - 0s 41us/step - loss: 5.2711 - acc: 0.0138 - val_loss: 4.3496 - val_acc: 0.0000e+00
    Epoch 73/100
    363/363 [==============================] - 0s 88us/step - loss: 5.2302 - acc: 0.0028 - val_loss: 4.3123 - val_acc: 0.0000e+00
    Epoch 74/100
    363/363 [==============================] - 0s 71us/step - loss: 5.1935 - acc: 0.0055 - val_loss: 4.6482 - val_acc: 0.0000e+00
    Epoch 75/100
    363/363 [==============================] - 0s 60us/step - loss: 5.2484 - acc: 0.0083 - val_loss: 4.2633 - val_acc: 0.0488
    Epoch 76/100
    363/363 [==============================] - 0s 66us/step - loss: 5.0663 - acc: 0.0193 - val_loss: 4.7944 - val_acc: 0.0244
    Epoch 77/100
    363/363 [==============================] - 0s 49us/step - loss: 5.0433 - acc: 0.0055 - val_loss: 4.3894 - val_acc: 0.0000e+00
    Epoch 78/100
    363/363 [==============================] - 0s 85us/step - loss: 5.0170 - acc: 0.0165 - val_loss: 4.5003 - val_acc: 0.0000e+00
    Epoch 79/100
    363/363 [==============================] - 0s 49us/step - loss: 4.9952 - acc: 0.0110 - val_loss: 4.3675 - val_acc: 0.0000e+00
    Epoch 80/100
    363/363 [==============================] - 0s 47us/step - loss: 5.0244 - acc: 0.0083 - val_loss: 4.7774 - val_acc: 0.0244
    Epoch 81/100
    363/363 [==============================] - 0s 91us/step - loss: 5.1797 - acc: 0.0083 - val_loss: 4.1789 - val_acc: 0.0488
    Epoch 82/100
    363/363 [==============================] - 0s 52us/step - loss: 5.0588 - acc: 0.0138 - val_loss: 4.6822 - val_acc: 0.0000e+00
    Epoch 83/100
    363/363 [==============================] - 0s 49us/step - loss: 5.1104 - acc: 0.0138 - val_loss: 4.5257 - val_acc: 0.0000e+00
    Epoch 84/100
    363/363 [==============================] - 0s 140us/step - loss: 4.9861 - acc: 0.0193 - val_loss: 4.3343 - val_acc: 0.0000e+00
    Epoch 85/100
    363/363 [==============================] - 0s 63us/step - loss: 4.9474 - acc: 0.0110 - val_loss: 4.5225 - val_acc: 0.0000e+00
    Epoch 86/100
    363/363 [==============================] - 0s 47us/step - loss: 4.9094 - acc: 0.0138 - val_loss: 4.5082 - val_acc: 0.0000e+00
    Epoch 87/100
    363/363 [==============================] - 0s 99us/step - loss: 4.8657 - acc: 0.0110 - val_loss: 4.4635 - val_acc: 0.0000e+00
    Epoch 88/100
    363/363 [==============================] - 0s 49us/step - loss: 4.8746 - acc: 0.0165 - val_loss: 4.4516 - val_acc: 0.0000e+00
    Epoch 89/100
    363/363 [==============================] - 0s 49us/step - loss: 4.8652 - acc: 0.0110 - val_loss: 4.6033 - val_acc: 0.0000e+00
    Epoch 90/100
    363/363 [==============================] - 0s 44us/step - loss: 4.8338 - acc: 0.0165 - val_loss: 4.5355 - val_acc: 0.0000e+00
    Epoch 91/100
    363/363 [==============================] - 0s 44us/step - loss: 4.8301 - acc: 0.0138 - val_loss: 4.2821 - val_acc: 0.0488
    Epoch 92/100
    363/363 [==============================] - 0s 44us/step - loss: 4.8001 - acc: 0.0138 - val_loss: 4.6296 - val_acc: 0.0000e+00
    Epoch 93/100
    363/363 [==============================] - 0s 47us/step - loss: 4.7565 - acc: 0.0193 - val_loss: 4.3032 - val_acc: 0.0244
    Epoch 94/100
    363/363 [==============================] - 0s 52us/step - loss: 4.7773 - acc: 0.0055 - val_loss: 4.6034 - val_acc: 0.0244
    Epoch 95/100
    363/363 [==============================] - 0s 49us/step - loss: 4.7643 - acc: 0.0138 - val_loss: 4.4051 - val_acc: 0.0000e+00
    Epoch 96/100
    363/363 [==============================] - 0s 41us/step - loss: 4.7349 - acc: 0.0110 - val_loss: 4.6976 - val_acc: 0.0244
    Epoch 97/100
    363/363 [==============================] - 0s 55us/step - loss: 4.8761 - acc: 0.0083 - val_loss: 4.1767 - val_acc: 0.0000e+00
    Epoch 98/100
    363/363 [==============================] - 0s 58us/step - loss: 5.0155 - acc: 0.0138 - val_loss: 4.7457 - val_acc: 0.0244
    Epoch 99/100
    363/363 [==============================] - 0s 49us/step - loss: 4.7588 - acc: 0.0138 - val_loss: 4.5146 - val_acc: 0.0000e+00
    Epoch 100/100
    363/363 [==============================] - 0s 52us/step - loss: 4.7109 - acc: 0.0055 - val_loss: 4.3469 - val_acc: 0.0488
    



.. parsed-literal::

    <keras.callbacks.History at 0x2f04d1ed0b8>



Model evaluation function
=========================

.. code:: ipython3

    def test_model(model, metric="accuracy"):
        evaluation = model.evaluate(x_test, y_test, verbose=0)
    
        print("-------------------------------------")
        print("Loss over the test dataset: %.2f" % (evaluation[0]))
        print("-------------------------------------")
        if metric == "accuracy":
            print("Accuarcy: %.2f" % (evaluation[1]))
        elif metric == "mae":
            print("Mean absolute error: %.2f" % (evaluation[1]))

Test the basic model
====================

.. code:: ipython3

    test_model(basic_model)


.. parsed-literal::

    -------------------------------------
    Loss over the test dataset: 29.40
    -------------------------------------
    Accuarcy: 0.00
    

Test the advanced model
=======================

.. code:: ipython3

    test_model(advanced_model)


.. parsed-literal::

    -------------------------------------
    Loss over the test dataset: 4.86
    -------------------------------------
    Accuarcy: 0.03
    

Define function to check actual house predictions
=================================================

.. code:: ipython3

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

Demonstrate slicing
===================

.. code:: ipython3

    print("sliced: ", x_train[0:1])
    print("not sliced: ", x_train[0])


.. parsed-literal::

    sliced:  [[  1.23247   0.        8.14      0.        0.538     6.142    91.7
        3.9769    4.      307.       21.      396.9      18.72   ]]
    not sliced:  [  1.23247   0.        8.14      0.        0.538     6.142    91.7
       3.9769    4.      307.       21.      396.9      18.72   ]
    

Check concrete predictions for our advanced model
=================================================

.. code:: ipython3

    check_predictions(advanced_model)


.. parsed-literal::

    
    
    Training set points:
    Predicted price: [[19.079166]], Actual price: [15.2]
    Predicted price: [[17.994045]], Actual price: 12.1
    Predicted price: [[27.042465]], Actual price: 23.9
    
    
    Testing set points:
    Predicted price: [[19.801466]], Actual price: 18.8
    Predicted price: [[32.983814]], Actual price: 35.4
    Predicted price: [[24.607197]], Actual price: 26.7
    

Improve the model
=================

.. code:: ipython3

    improved_model = Sequential()
    # add neuron units and add nonlinear activation function
    improved_model.add(Dense(units = 64, input_dim = 13, kernel_initializer = 'truncated_normal', activation = "relu"))
    
    improved_model.add(Dense(units = 32, kernel_initializer = 'truncated_normal', activation = "relu"))
    
    improved_model.add(Dense(units = 1, kernel_initializer = 'truncated_normal'))
    
    # mean squared error penalizes larger difference, works better
    improved_model.compile(optimizer = "adam", loss = 'mean_squared_error', metrics = ['mae'])
    # changes batch size = 32 which works better,both loss and validation loss has decreased at epochs=150
    improved_model.fit(x_train, y_train, epochs = 150, batch_size = 32, verbose = 1, validation_split = 0.2)
    
    check_predictions(improved_model)


.. parsed-literal::

    Train on 323 samples, validate on 81 samples
    Epoch 1/150
    323/323 [==============================] - 1s 3ms/step - loss: 441.2293 - mean_absolute_error: 18.4925 - val_loss: 312.3085 - val_mean_absolute_error: 14.6657
    Epoch 2/150
    323/323 [==============================] - 0s 123us/step - loss: 180.0133 - mean_absolute_error: 10.5206 - val_loss: 127.5609 - val_mean_absolute_error: 8.8612
    Epoch 3/150
    323/323 [==============================] - 0s 170us/step - loss: 132.8036 - mean_absolute_error: 8.7872 - val_loss: 112.4659 - val_mean_absolute_error: 8.1237
    Epoch 4/150
    323/323 [==============================] - 0s 130us/step - loss: 106.3767 - mean_absolute_error: 7.4756 - val_loss: 116.6352 - val_mean_absolute_error: 7.8220
    Epoch 5/150
    323/323 [==============================] - 0s 111us/step - loss: 91.1831 - mean_absolute_error: 6.7197 - val_loss: 86.2633 - val_mean_absolute_error: 6.9441
    Epoch 6/150
    323/323 [==============================] - 0s 133us/step - loss: 79.1563 - mean_absolute_error: 6.5656 - val_loss: 78.6052 - val_mean_absolute_error: 6.1694
    Epoch 7/150
    323/323 [==============================] - 0s 124us/step - loss: 69.1063 - mean_absolute_error: 5.6202 - val_loss: 78.3706 - val_mean_absolute_error: 5.8507
    Epoch 8/150
    323/323 [==============================] - 0s 83us/step - loss: 64.7206 - mean_absolute_error: 5.6190 - val_loss: 70.2571 - val_mean_absolute_error: 5.7456
    Epoch 9/150
    323/323 [==============================] - 0s 102us/step - loss: 61.1135 - mean_absolute_error: 5.3872 - val_loss: 73.7842 - val_mean_absolute_error: 5.5514
    Epoch 10/150
    323/323 [==============================] - 0s 145us/step - loss: 60.9069 - mean_absolute_error: 5.3647 - val_loss: 68.1519 - val_mean_absolute_error: 5.6286
    Epoch 11/150
    323/323 [==============================] - ETA: 0s - loss: 48.9758 - mean_absolute_error: 5.55 - 0s 142us/step - loss: 58.8660 - mean_absolute_error: 5.3615 - val_loss: 70.6623 - val_mean_absolute_error: 5.4833
    Epoch 12/150
    323/323 [==============================] - 0s 96us/step - loss: 58.3411 - mean_absolute_error: 5.1941 - val_loss: 67.5164 - val_mean_absolute_error: 5.5408
    Epoch 13/150
    323/323 [==============================] - 0s 93us/step - loss: 57.2751 - mean_absolute_error: 5.3322 - val_loss: 69.2805 - val_mean_absolute_error: 5.4331
    Epoch 14/150
    323/323 [==============================] - 0s 148us/step - loss: 56.4446 - mean_absolute_error: 5.0754 - val_loss: 66.6182 - val_mean_absolute_error: 5.5064
    Epoch 15/150
    323/323 [==============================] - 0s 195us/step - loss: 56.9475 - mean_absolute_error: 5.5217 - val_loss: 67.5023 - val_mean_absolute_error: 5.3835
    Epoch 16/150
    323/323 [==============================] - 0s 86us/step - loss: 55.6942 - mean_absolute_error: 5.0036 - val_loss: 66.0344 - val_mean_absolute_error: 5.3915
    Epoch 17/150
    323/323 [==============================] - 0s 96us/step - loss: 54.1180 - mean_absolute_error: 5.1182 - val_loss: 66.5261 - val_mean_absolute_error: 5.3302
    Epoch 18/150
    323/323 [==============================] - 0s 80us/step - loss: 53.6155 - mean_absolute_error: 5.0151 - val_loss: 68.2123 - val_mean_absolute_error: 5.2675
    Epoch 19/150
    323/323 [==============================] - 0s 99us/step - loss: 53.4091 - mean_absolute_error: 4.8091 - val_loss: 64.1694 - val_mean_absolute_error: 5.3649
    Epoch 20/150
    323/323 [==============================] - 0s 151us/step - loss: 52.5406 - mean_absolute_error: 5.0221 - val_loss: 65.9217 - val_mean_absolute_error: 5.2431
    Epoch 21/150
    323/323 [==============================] - 0s 127us/step - loss: 52.2549 - mean_absolute_error: 4.9040 - val_loss: 67.9358 - val_mean_absolute_error: 5.2018
    Epoch 22/150
    323/323 [==============================] - 0s 90us/step - loss: 53.8884 - mean_absolute_error: 4.6979 - val_loss: 61.8989 - val_mean_absolute_error: 5.2811
    Epoch 23/150
    323/323 [==============================] - 0s 108us/step - loss: 51.6003 - mean_absolute_error: 4.9563 - val_loss: 63.0338 - val_mean_absolute_error: 5.1501
    Epoch 24/150
    323/323 [==============================] - 0s 124us/step - loss: 50.4431 - mean_absolute_error: 4.9436 - val_loss: 65.0851 - val_mean_absolute_error: 5.1089
    Epoch 25/150
    323/323 [==============================] - 0s 83us/step - loss: 50.0588 - mean_absolute_error: 4.6187 - val_loss: 60.2674 - val_mean_absolute_error: 5.1475
    Epoch 26/150
    323/323 [==============================] - 0s 74us/step - loss: 48.8548 - mean_absolute_error: 4.7664 - val_loss: 61.2168 - val_mean_absolute_error: 5.0609
    Epoch 27/150
    323/323 [==============================] - 0s 170us/step - loss: 49.4343 - mean_absolute_error: 4.6116 - val_loss: 57.9659 - val_mean_absolute_error: 5.2841
    Epoch 28/150
    323/323 [==============================] - 0s 114us/step - loss: 48.5890 - mean_absolute_error: 5.0150 - val_loss: 66.1015 - val_mean_absolute_error: 5.0633
    Epoch 29/150
    323/323 [==============================] - 0s 93us/step - loss: 50.3099 - mean_absolute_error: 4.6754 - val_loss: 56.8893 - val_mean_absolute_error: 5.1069
    Epoch 30/150
    323/323 [==============================] - 0s 111us/step - loss: 46.9988 - mean_absolute_error: 4.6414 - val_loss: 56.7745 - val_mean_absolute_error: 4.9651
    Epoch 31/150
    323/323 [==============================] - 0s 108us/step - loss: 46.4723 - mean_absolute_error: 4.5852 - val_loss: 56.4074 - val_mean_absolute_error: 4.9237
    Epoch 32/150
    323/323 [==============================] - 0s 105us/step - loss: 45.7336 - mean_absolute_error: 4.4720 - val_loss: 55.3370 - val_mean_absolute_error: 5.0391
    Epoch 33/150
    323/323 [==============================] - 0s 83us/step - loss: 44.9991 - mean_absolute_error: 4.6046 - val_loss: 54.6293 - val_mean_absolute_error: 4.8692
    Epoch 34/150
    323/323 [==============================] - 0s 83us/step - loss: 43.7522 - mean_absolute_error: 4.4672 - val_loss: 56.7307 - val_mean_absolute_error: 4.8244
    Epoch 35/150
    323/323 [==============================] - 0s 133us/step - loss: 43.9845 - mean_absolute_error: 4.3511 - val_loss: 51.8416 - val_mean_absolute_error: 4.9549
    Epoch 36/150
    323/323 [==============================] - 0s 77us/step - loss: 45.8603 - mean_absolute_error: 4.8676 - val_loss: 56.7073 - val_mean_absolute_error: 4.8208
    Epoch 37/150
    323/323 [==============================] - 0s 99us/step - loss: 41.6780 - mean_absolute_error: 4.4411 - val_loss: 51.5690 - val_mean_absolute_error: 4.8282
    Epoch 38/150
    323/323 [==============================] - 0s 80us/step - loss: 41.1173 - mean_absolute_error: 4.3558 - val_loss: 50.9333 - val_mean_absolute_error: 4.7807
    Epoch 39/150
    323/323 [==============================] - 0s 130us/step - loss: 41.3286 - mean_absolute_error: 4.6580 - val_loss: 59.3128 - val_mean_absolute_error: 4.9578
    Epoch 40/150
    323/323 [==============================] - 0s 124us/step - loss: 42.7752 - mean_absolute_error: 4.3898 - val_loss: 49.8825 - val_mean_absolute_error: 5.1964
    Epoch 41/150
    323/323 [==============================] - 0s 111us/step - loss: 41.4585 - mean_absolute_error: 4.4578 - val_loss: 51.0796 - val_mean_absolute_error: 4.6750
    Epoch 42/150
    323/323 [==============================] - 0s 173us/step - loss: 39.3859 - mean_absolute_error: 4.2963 - val_loss: 47.4860 - val_mean_absolute_error: 4.6543
    Epoch 43/150
    323/323 [==============================] - 0s 86us/step - loss: 38.5087 - mean_absolute_error: 4.1458 - val_loss: 46.4232 - val_mean_absolute_error: 4.7336
    Epoch 44/150
    323/323 [==============================] - 0s 83us/step - loss: 38.2484 - mean_absolute_error: 4.3773 - val_loss: 48.1360 - val_mean_absolute_error: 4.5515
    Epoch 45/150
    323/323 [==============================] - 0s 71us/step - loss: 37.0735 - mean_absolute_error: 4.2339 - val_loss: 46.2012 - val_mean_absolute_error: 4.5357
    Epoch 46/150
    323/323 [==============================] - 0s 93us/step - loss: 36.8482 - mean_absolute_error: 4.2344 - val_loss: 49.7975 - val_mean_absolute_error: 4.5537
    Epoch 47/150
    323/323 [==============================] - 0s 83us/step - loss: 37.7333 - mean_absolute_error: 4.2188 - val_loss: 46.0541 - val_mean_absolute_error: 4.4607
    Epoch 48/150
    323/323 [==============================] - 0s 86us/step - loss: 35.6940 - mean_absolute_error: 4.1461 - val_loss: 47.3360 - val_mean_absolute_error: 4.4531
    Epoch 49/150
    323/323 [==============================] - 0s 83us/step - loss: 36.8875 - mean_absolute_error: 4.1100 - val_loss: 42.5192 - val_mean_absolute_error: 4.4994
    Epoch 50/150
    323/323 [==============================] - 0s 108us/step - loss: 34.4098 - mean_absolute_error: 4.0596 - val_loss: 46.5403 - val_mean_absolute_error: 4.4247
    Epoch 51/150
    323/323 [==============================] - 0s 136us/step - loss: 34.5377 - mean_absolute_error: 4.2126 - val_loss: 45.1822 - val_mean_absolute_error: 4.3815
    Epoch 52/150
    323/323 [==============================] - 0s 96us/step - loss: 34.6114 - mean_absolute_error: 4.0607 - val_loss: 40.1372 - val_mean_absolute_error: 4.4437
    Epoch 53/150
    323/323 [==============================] - 0s 117us/step - loss: 33.0691 - mean_absolute_error: 3.9185 - val_loss: 39.6172 - val_mean_absolute_error: 4.6036
    Epoch 54/150
    323/323 [==============================] - 0s 90us/step - loss: 34.4710 - mean_absolute_error: 4.3236 - val_loss: 43.5103 - val_mean_absolute_error: 4.3226
    Epoch 55/150
    323/323 [==============================] - 0s 105us/step - loss: 32.4232 - mean_absolute_error: 4.0532 - val_loss: 40.8916 - val_mean_absolute_error: 4.2706
    Epoch 56/150
    323/323 [==============================] - 0s 157us/step - loss: 34.1398 - mean_absolute_error: 4.1298 - val_loss: 38.2386 - val_mean_absolute_error: 4.3280
    Epoch 57/150
    323/323 [==============================] - 0s 148us/step - loss: 31.8875 - mean_absolute_error: 3.8508 - val_loss: 37.6155 - val_mean_absolute_error: 4.4515
    Epoch 58/150
    323/323 [==============================] - 0s 114us/step - loss: 31.1036 - mean_absolute_error: 4.0828 - val_loss: 44.1133 - val_mean_absolute_error: 4.3349
    Epoch 59/150
    323/323 [==============================] - 0s 96us/step - loss: 31.7447 - mean_absolute_error: 3.8310 - val_loss: 36.2162 - val_mean_absolute_error: 4.3343
    Epoch 60/150
    323/323 [==============================] - 0s 114us/step - loss: 32.5302 - mean_absolute_error: 4.1414 - val_loss: 36.7171 - val_mean_absolute_error: 4.0838
    Epoch 61/150
    323/323 [==============================] - 0s 90us/step - loss: 29.6228 - mean_absolute_error: 3.7965 - val_loss: 35.2041 - val_mean_absolute_error: 4.0948
    Epoch 62/150
    323/323 [==============================] - 0s 96us/step - loss: 29.2416 - mean_absolute_error: 3.7900 - val_loss: 33.8856 - val_mean_absolute_error: 4.1579
    Epoch 63/150
    323/323 [==============================] - 0s 102us/step - loss: 29.1467 - mean_absolute_error: 3.8250 - val_loss: 33.5733 - val_mean_absolute_error: 4.1064
    Epoch 64/150
    323/323 [==============================] - 0s 145us/step - loss: 28.7287 - mean_absolute_error: 3.8176 - val_loss: 32.9463 - val_mean_absolute_error: 4.1101
    Epoch 65/150
    323/323 [==============================] - 0s 139us/step - loss: 29.0928 - mean_absolute_error: 3.7439 - val_loss: 36.0013 - val_mean_absolute_error: 4.7383
    Epoch 66/150
    323/323 [==============================] - 0s 142us/step - loss: 32.0252 - mean_absolute_error: 4.1661 - val_loss: 32.0806 - val_mean_absolute_error: 4.0156
    Epoch 67/150
    323/323 [==============================] - 0s 139us/step - loss: 30.6671 - mean_absolute_error: 4.1989 - val_loss: 32.2960 - val_mean_absolute_error: 3.8858
    Epoch 68/150
    323/323 [==============================] - 0s 120us/step - loss: 28.8286 - mean_absolute_error: 4.0116 - val_loss: 42.5468 - val_mean_absolute_error: 4.3448
    Epoch 69/150
    323/323 [==============================] - 0s 86us/step - loss: 29.7401 - mean_absolute_error: 3.9260 - val_loss: 31.0713 - val_mean_absolute_error: 3.8738
    Epoch 70/150
    323/323 [==============================] - 0s 151us/step - loss: 26.8224 - mean_absolute_error: 3.6821 - val_loss: 31.2173 - val_mean_absolute_error: 3.7880
    Epoch 71/150
    323/323 [==============================] - 0s 108us/step - loss: 26.5629 - mean_absolute_error: 3.7493 - val_loss: 31.4047 - val_mean_absolute_error: 3.7897
    Epoch 72/150
    323/323 [==============================] - 0s 96us/step - loss: 25.9398 - mean_absolute_error: 3.7862 - val_loss: 30.8300 - val_mean_absolute_error: 3.7527
    Epoch 73/150
    323/323 [==============================] - 0s 124us/step - loss: 26.7045 - mean_absolute_error: 3.8248 - val_loss: 29.9571 - val_mean_absolute_error: 3.7495
    Epoch 74/150
    323/323 [==============================] - 0s 108us/step - loss: 26.4636 - mean_absolute_error: 3.6559 - val_loss: 29.2668 - val_mean_absolute_error: 3.7258
    Epoch 75/150
    323/323 [==============================] - 0s 124us/step - loss: 25.2071 - mean_absolute_error: 3.6394 - val_loss: 28.4011 - val_mean_absolute_error: 3.8290
    Epoch 76/150
    323/323 [==============================] - 0s 99us/step - loss: 27.4804 - mean_absolute_error: 3.8675 - val_loss: 28.4731 - val_mean_absolute_error: 3.7212
    Epoch 77/150
    323/323 [==============================] - 0s 111us/step - loss: 30.2229 - mean_absolute_error: 4.0626 - val_loss: 35.4745 - val_mean_absolute_error: 3.9912
    Epoch 78/150
    323/323 [==============================] - 0s 139us/step - loss: 26.4034 - mean_absolute_error: 3.9012 - val_loss: 32.8916 - val_mean_absolute_error: 3.8501
    Epoch 79/150
    323/323 [==============================] - 0s 170us/step - loss: 25.6751 - mean_absolute_error: 3.7336 - val_loss: 27.2353 - val_mean_absolute_error: 3.7029
    Epoch 80/150
    323/323 [==============================] - 0s 133us/step - loss: 25.9962 - mean_absolute_error: 3.6868 - val_loss: 27.3694 - val_mean_absolute_error: 3.6871
    Epoch 81/150
    323/323 [==============================] - 0s 111us/step - loss: 25.4086 - mean_absolute_error: 3.5846 - val_loss: 28.6596 - val_mean_absolute_error: 4.0419
    Epoch 82/150
    323/323 [==============================] - 0s 93us/step - loss: 23.2533 - mean_absolute_error: 3.5878 - val_loss: 31.3840 - val_mean_absolute_error: 3.7778
    Epoch 83/150
    323/323 [==============================] - 0s 80us/step - loss: 24.5994 - mean_absolute_error: 3.6334 - val_loss: 26.3073 - val_mean_absolute_error: 3.5860
    Epoch 84/150
    323/323 [==============================] - 0s 111us/step - loss: 23.6109 - mean_absolute_error: 3.6126 - val_loss: 29.0921 - val_mean_absolute_error: 3.6604
    Epoch 85/150
    323/323 [==============================] - 0s 74us/step - loss: 23.4182 - mean_absolute_error: 3.5376 - val_loss: 25.8445 - val_mean_absolute_error: 3.7310
    Epoch 86/150
    323/323 [==============================] - 0s 114us/step - loss: 23.8492 - mean_absolute_error: 3.5693 - val_loss: 28.0465 - val_mean_absolute_error: 3.5868
    Epoch 87/150
    323/323 [==============================] - 0s 142us/step - loss: 23.3514 - mean_absolute_error: 3.6682 - val_loss: 29.6603 - val_mean_absolute_error: 3.7050
    Epoch 88/150
    323/323 [==============================] - 0s 120us/step - loss: 24.4779 - mean_absolute_error: 3.6224 - val_loss: 24.6481 - val_mean_absolute_error: 3.5349
    Epoch 89/150
    323/323 [==============================] - 0s 123us/step - loss: 22.3639 - mean_absolute_error: 3.5078 - val_loss: 27.8328 - val_mean_absolute_error: 3.6176
    Epoch 90/150
    323/323 [==============================] - 0s 148us/step - loss: 21.8766 - mean_absolute_error: 3.4539 - val_loss: 24.0316 - val_mean_absolute_error: 3.5419
    Epoch 91/150
    323/323 [==============================] - 0s 133us/step - loss: 22.1447 - mean_absolute_error: 3.5500 - val_loss: 25.5041 - val_mean_absolute_error: 3.4615
    Epoch 92/150
    323/323 [==============================] - 0s 124us/step - loss: 22.1175 - mean_absolute_error: 3.3839 - val_loss: 29.5606 - val_mean_absolute_error: 4.3192
    Epoch 93/150
    323/323 [==============================] - ETA: 0s - loss: 19.9985 - mean_absolute_error: 3.63 - 0s 111us/step - loss: 30.8597 - mean_absolute_error: 4.3460 - val_loss: 30.2479 - val_mean_absolute_error: 3.8889
    Epoch 94/150
    323/323 [==============================] - 0s 154us/step - loss: 22.5009 - mean_absolute_error: 3.6067 - val_loss: 23.7829 - val_mean_absolute_error: 3.4745
    Epoch 95/150
    323/323 [==============================] - 0s 93us/step - loss: 23.8507 - mean_absolute_error: 3.6231 - val_loss: 24.7789 - val_mean_absolute_error: 3.4440
    Epoch 96/150
    323/323 [==============================] - 0s 222us/step - loss: 22.5332 - mean_absolute_error: 3.5998 - val_loss: 28.9281 - val_mean_absolute_error: 3.7662
    Epoch 97/150
    323/323 [==============================] - 0s 145us/step - loss: 23.2586 - mean_absolute_error: 3.7074 - val_loss: 27.0760 - val_mean_absolute_error: 3.6083
    Epoch 98/150
    323/323 [==============================] - 0s 114us/step - loss: 22.4820 - mean_absolute_error: 3.4462 - val_loss: 24.7896 - val_mean_absolute_error: 3.7696
    Epoch 99/150
    323/323 [==============================] - 0s 102us/step - loss: 22.5926 - mean_absolute_error: 3.4620 - val_loss: 22.6149 - val_mean_absolute_error: 3.4264
    Epoch 100/150
    323/323 [==============================] - 0s 83us/step - loss: 21.6733 - mean_absolute_error: 3.5086 - val_loss: 23.1227 - val_mean_absolute_error: 3.3168
    Epoch 101/150
    323/323 [==============================] - 0s 114us/step - loss: 20.8163 - mean_absolute_error: 3.4153 - val_loss: 24.0983 - val_mean_absolute_error: 3.3616
    Epoch 102/150
    323/323 [==============================] - 0s 195us/step - loss: 24.4763 - mean_absolute_error: 3.7005 - val_loss: 21.6684 - val_mean_absolute_error: 3.3808
    Epoch 103/150
    323/323 [==============================] - 0s 102us/step - loss: 23.7615 - mean_absolute_error: 3.6549 - val_loss: 23.9016 - val_mean_absolute_error: 3.7813
    Epoch 104/150
    323/323 [==============================] - 0s 124us/step - loss: 21.1402 - mean_absolute_error: 3.5091 - val_loss: 22.1483 - val_mean_absolute_error: 3.4780
    Epoch 105/150
    323/323 [==============================] - 0s 93us/step - loss: 21.4392 - mean_absolute_error: 3.4337 - val_loss: 21.9492 - val_mean_absolute_error: 3.2355
    Epoch 106/150
    323/323 [==============================] - 0s 111us/step - loss: 20.0684 - mean_absolute_error: 3.3428 - val_loss: 21.2497 - val_mean_absolute_error: 3.2560
    Epoch 107/150
    323/323 [==============================] - 0s 148us/step - loss: 19.4042 - mean_absolute_error: 3.2667 - val_loss: 21.0290 - val_mean_absolute_error: 3.2830
    Epoch 108/150
    323/323 [==============================] - 0s 105us/step - loss: 19.7759 - mean_absolute_error: 3.2984 - val_loss: 20.6814 - val_mean_absolute_error: 3.2267
    Epoch 109/150
    323/323 [==============================] - 0s 133us/step - loss: 19.4693 - mean_absolute_error: 3.3001 - val_loss: 21.1351 - val_mean_absolute_error: 3.1729
    Epoch 110/150
    323/323 [==============================] - 0s 96us/step - loss: 19.6198 - mean_absolute_error: 3.2819 - val_loss: 20.4186 - val_mean_absolute_error: 3.2278
    Epoch 111/150
    323/323 [==============================] - 0s 111us/step - loss: 19.8047 - mean_absolute_error: 3.2685 - val_loss: 20.8218 - val_mean_absolute_error: 3.2883
    Epoch 112/150
    323/323 [==============================] - 0s 105us/step - loss: 19.2465 - mean_absolute_error: 3.2460 - val_loss: 20.0454 - val_mean_absolute_error: 3.2355
    Epoch 113/150
    323/323 [==============================] - 0s 108us/step - loss: 18.9483 - mean_absolute_error: 3.1695 - val_loss: 20.4522 - val_mean_absolute_error: 3.1146
    Epoch 114/150
    323/323 [==============================] - 0s 90us/step - loss: 19.1327 - mean_absolute_error: 3.2480 - val_loss: 21.5344 - val_mean_absolute_error: 3.2443
    Epoch 115/150
    323/323 [==============================] - 0s 111us/step - loss: 19.0350 - mean_absolute_error: 3.2379 - val_loss: 20.4012 - val_mean_absolute_error: 3.3402
    Epoch 116/150
    323/323 [==============================] - 0s 83us/step - loss: 20.7621 - mean_absolute_error: 3.4122 - val_loss: 20.8412 - val_mean_absolute_error: 3.4790
    Epoch 117/150
    323/323 [==============================] - 0s 145us/step - loss: 21.7454 - mean_absolute_error: 3.5276 - val_loss: 19.7233 - val_mean_absolute_error: 3.1975
    Epoch 118/150
    323/323 [==============================] - 0s 127us/step - loss: 24.5618 - mean_absolute_error: 3.7458 - val_loss: 26.0701 - val_mean_absolute_error: 3.6625
    Epoch 119/150
    323/323 [==============================] - 0s 83us/step - loss: 22.1554 - mean_absolute_error: 3.6280 - val_loss: 27.6640 - val_mean_absolute_error: 3.7163
    Epoch 120/150
    323/323 [==============================] - 0s 105us/step - loss: 21.0445 - mean_absolute_error: 3.3613 - val_loss: 20.3878 - val_mean_absolute_error: 3.1323
    Epoch 121/150
    323/323 [==============================] - 0s 96us/step - loss: 19.0708 - mean_absolute_error: 3.1861 - val_loss: 19.6435 - val_mean_absolute_error: 3.0827
    Epoch 122/150
    323/323 [==============================] - 0s 148us/step - loss: 18.2059 - mean_absolute_error: 3.1705 - val_loss: 18.5899 - val_mean_absolute_error: 3.1262
    Epoch 123/150
    323/323 [==============================] - 0s 90us/step - loss: 17.8249 - mean_absolute_error: 3.0679 - val_loss: 18.7766 - val_mean_absolute_error: 3.1289
    Epoch 124/150
    323/323 [==============================] - 0s 133us/step - loss: 17.8737 - mean_absolute_error: 3.1528 - val_loss: 20.7438 - val_mean_absolute_error: 3.2077
    Epoch 125/150
    323/323 [==============================] - 0s 96us/step - loss: 19.7929 - mean_absolute_error: 3.2609 - val_loss: 17.8531 - val_mean_absolute_error: 3.1282
    Epoch 126/150
    323/323 [==============================] - 0s 114us/step - loss: 18.7959 - mean_absolute_error: 3.1083 - val_loss: 21.9922 - val_mean_absolute_error: 3.6529
    Epoch 127/150
    323/323 [==============================] - 0s 142us/step - loss: 19.0798 - mean_absolute_error: 3.2989 - val_loss: 22.3788 - val_mean_absolute_error: 3.3188
    Epoch 128/150
    323/323 [==============================] - 0s 102us/step - loss: 17.7891 - mean_absolute_error: 3.1723 - val_loss: 19.9667 - val_mean_absolute_error: 3.1153
    Epoch 129/150
    323/323 [==============================] - 0s 133us/step - loss: 18.4022 - mean_absolute_error: 3.1595 - val_loss: 20.8778 - val_mean_absolute_error: 3.2033
    Epoch 130/150
    323/323 [==============================] - 0s 204us/step - loss: 21.3323 - mean_absolute_error: 3.3144 - val_loss: 19.0583 - val_mean_absolute_error: 3.0794
    Epoch 131/150
    323/323 [==============================] - 0s 99us/step - loss: 18.2053 - mean_absolute_error: 3.1228 - val_loss: 17.3866 - val_mean_absolute_error: 3.1033
    Epoch 132/150
    323/323 [==============================] - 0s 77us/step - loss: 18.5518 - mean_absolute_error: 3.1173 - val_loss: 18.7195 - val_mean_absolute_error: 3.2658
    Epoch 133/150
    323/323 [==============================] - 0s 133us/step - loss: 17.1561 - mean_absolute_error: 3.0492 - val_loss: 20.3995 - val_mean_absolute_error: 3.1630
    Epoch 134/150
    323/323 [==============================] - 0s 74us/step - loss: 17.5334 - mean_absolute_error: 3.0589 - val_loss: 18.0004 - val_mean_absolute_error: 2.9850
    Epoch 135/150
    323/323 [==============================] - 0s 120us/step - loss: 16.8346 - mean_absolute_error: 2.9608 - val_loss: 19.1832 - val_mean_absolute_error: 3.0897
    Epoch 136/150
    323/323 [==============================] - 0s 96us/step - loss: 21.1168 - mean_absolute_error: 3.3528 - val_loss: 17.8902 - val_mean_absolute_error: 3.0453
    Epoch 137/150
    323/323 [==============================] - 0s 111us/step - loss: 16.9605 - mean_absolute_error: 2.9751 - val_loss: 18.2520 - val_mean_absolute_error: 3.1990
    Epoch 138/150
    323/323 [==============================] - 0s 102us/step - loss: 17.0023 - mean_absolute_error: 3.0683 - val_loss: 17.4571 - val_mean_absolute_error: 3.0096
    Epoch 139/150
    323/323 [==============================] - 0s 154us/step - loss: 16.5949 - mean_absolute_error: 3.0306 - val_loss: 19.5907 - val_mean_absolute_error: 3.1396
    Epoch 140/150
    323/323 [==============================] - 0s 80us/step - loss: 18.3641 - mean_absolute_error: 3.1110 - val_loss: 17.0739 - val_mean_absolute_error: 2.9346
    Epoch 141/150
    323/323 [==============================] - 0s 120us/step - loss: 17.4066 - mean_absolute_error: 3.0030 - val_loss: 17.5873 - val_mean_absolute_error: 2.9967
    Epoch 142/150
    323/323 [==============================] - 0s 102us/step - loss: 22.4235 - mean_absolute_error: 3.4047 - val_loss: 23.6989 - val_mean_absolute_error: 4.0217
    Epoch 143/150
    323/323 [==============================] - 0s 108us/step - loss: 18.2031 - mean_absolute_error: 3.1396 - val_loss: 19.5129 - val_mean_absolute_error: 3.1169
    Epoch 144/150
    323/323 [==============================] - 0s 90us/step - loss: 16.6163 - mean_absolute_error: 3.0969 - val_loss: 16.9010 - val_mean_absolute_error: 2.9391
    Epoch 145/150
    323/323 [==============================] - 0s 90us/step - loss: 16.6274 - mean_absolute_error: 2.9923 - val_loss: 16.3245 - val_mean_absolute_error: 2.9505
    Epoch 146/150
    323/323 [==============================] - 0s 105us/step - loss: 17.8811 - mean_absolute_error: 3.0470 - val_loss: 16.4729 - val_mean_absolute_error: 2.9733
    Epoch 147/150
    323/323 [==============================] - 0s 93us/step - loss: 19.0473 - mean_absolute_error: 3.2155 - val_loss: 20.5750 - val_mean_absolute_error: 3.6021
    Epoch 148/150
    323/323 [==============================] - 0s 93us/step - loss: 18.7719 - mean_absolute_error: 3.1663 - val_loss: 19.0754 - val_mean_absolute_error: 3.0535
    Epoch 149/150
    323/323 [==============================] - 0s 90us/step - loss: 19.7815 - mean_absolute_error: 3.2228 - val_loss: 17.9585 - val_mean_absolute_error: 3.2255
    Epoch 150/150
    323/323 [==============================] - 0s 83us/step - loss: 17.2428 - mean_absolute_error: 2.9718 - val_loss: 19.1928 - val_mean_absolute_error: 3.3960
    
    
    Training set points:
    Predicted price: [[19.105759]], Actual price: [15.2]
    Predicted price: [[15.409338]], Actual price: 12.1
    Predicted price: [[27.75078]], Actual price: 23.9
    
    
    Testing set points:
    Predicted price: [[20.693811]], Actual price: 18.8
    Predicted price: [[38.433586]], Actual price: 35.4
    Predicted price: [[29.925093]], Actual price: 26.7
    

.. code:: ipython3

    test_model(improved_model, metric = "mae")


.. parsed-literal::

    -------------------------------------
    Loss over the test dataset: 33.98
    -------------------------------------
    Mean absolute error: 4.23
    
