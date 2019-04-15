
Load dataset, split into train and test data and check the train data shape
===========================================================================

.. code:: ipython3

    from keras.datasets import boston_housing
    (x_train, y_train), (x_test, y_test) = boston_housing.load_data()
    
    x_train.shape


.. parsed-literal::

    Downloading data from https://s3.amazonaws.com/keras-datasets/boston_housing.npz
    57344/57026 [==============================] - 0s 1us/step
    



.. parsed-literal::

    (404, 13)



Start building the model graph, add input and output layers
===========================================================

.. code:: ipython3

    basic_model = Sequential()
    
    basic_model.add(Dense(units=13, input_dim = 13, kernel_initializer = 'normal'))
    
    basic_model.add(Dense(units = 1, kernel_initializer = 'normal', activation='linear'))


.. parsed-literal::

    WARNING:tensorflow:From D:\Anaconda\lib\site-packages\tensorflow\python\framework\op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    

Choose loss function and optimizer, compile model
=================================================

.. code:: ipython3

    basic_model.compile(optimizer=SGD(lr = 0.000001), loss='mean_absolute_error', metrics=['accuracy', "mse"])
    
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
    
    advanced_model.add(Dense(units=32, input_dim = 13, kernel_initializer = 'truncated_normal'))
    
    advanced_model.add(Dense(units = 16, kernel_initializer = 'truncated_normal'))
    
    advanced_model.add(Dense(units = 1, kernel_initializer = 'truncated_normal'))
    
    advanced_model.compile(optimizer="adam", loss='mean_absolute_error', metrics=['accuracy'])
    
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
    print("first house in the dataset's price in thousands:",y_train[0])


.. parsed-literal::

    features of the first house in the dataset: [  1.23247   0.        8.14      0.        0.538     6.142    91.7
       3.9769    4.      307.       21.      396.9      18.72   ]
    first house in the dataset's price in thousands: 15.2
    

Train the model
===============

.. code:: ipython3

    advanced_model.fit(x_train, y_train, epochs=100, batch_size=64, verbose=1, validation_split=0.1)


.. parsed-literal::

    WARNING:tensorflow:From D:\Anaconda\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Train on 363 samples, validate on 41 samples
    Epoch 1/100
    363/363 [==============================] - 1s 2ms/step - loss: 19.3377 - acc: 0.0000e+00 - val_loss: 16.1430 - val_acc: 0.0244
    Epoch 2/100
    363/363 [==============================] - 0s 54us/step - loss: 15.2794 - acc: 0.0000e+00 - val_loss: 11.5674 - val_acc: 0.0000e+00
    Epoch 3/100
    363/363 [==============================] - 0s 52us/step - loss: 10.9137 - acc: 0.0000e+00 - val_loss: 8.3565 - val_acc: 0.0244
    Epoch 4/100
    363/363 [==============================] - 0s 52us/step - loss: 9.0630 - acc: 0.0055 - val_loss: 7.9569 - val_acc: 0.0244
    Epoch 5/100
    363/363 [==============================] - 0s 58us/step - loss: 9.4486 - acc: 0.0000e+00 - val_loss: 7.3125 - val_acc: 0.0244
    Epoch 6/100
    363/363 [==============================] - 0s 74us/step - loss: 8.2134 - acc: 0.0083 - val_loss: 6.9047 - val_acc: 0.0000e+00
    Epoch 7/100
    363/363 [==============================] - 0s 58us/step - loss: 7.7215 - acc: 0.0055 - val_loss: 6.5544 - val_acc: 0.0000e+00
    Epoch 8/100
    363/363 [==============================] - 0s 63us/step - loss: 7.1712 - acc: 0.0000e+00 - val_loss: 5.7953 - val_acc: 0.0000e+00
    Epoch 9/100
    363/363 [==============================] - 0s 99us/step - loss: 6.7517 - acc: 0.0138 - val_loss: 5.2706 - val_acc: 0.0244
    Epoch 10/100
    363/363 [==============================] - 0s 47us/step - loss: 6.4897 - acc: 0.0083 - val_loss: 5.0814 - val_acc: 0.0000e+00
    Epoch 11/100
    363/363 [==============================] - 0s 110us/step - loss: 6.3113 - acc: 0.0083 - val_loss: 5.0397 - val_acc: 0.0000e+00
    Epoch 12/100
    363/363 [==============================] - 0s 91us/step - loss: 6.2136 - acc: 0.0138 - val_loss: 4.8852 - val_acc: 0.0000e+00
    Epoch 13/100
    363/363 [==============================] - 0s 63us/step - loss: 6.1310 - acc: 0.0083 - val_loss: 4.8340 - val_acc: 0.0244
    Epoch 14/100
    363/363 [==============================] - 0s 52us/step - loss: 6.0924 - acc: 0.0055 - val_loss: 4.8082 - val_acc: 0.0244
    Epoch 15/100
    363/363 [==============================] - 0s 66us/step - loss: 6.0763 - acc: 0.0110 - val_loss: 4.8308 - val_acc: 0.0244
    Epoch 16/100
    363/363 [==============================] - 0s 58us/step - loss: 6.0520 - acc: 0.0083 - val_loss: 4.7649 - val_acc: 0.0244
    Epoch 17/100
    363/363 [==============================] - 0s 71us/step - loss: 6.0225 - acc: 0.0110 - val_loss: 4.8193 - val_acc: 0.0000e+00
    Epoch 18/100
    363/363 [==============================] - 0s 71us/step - loss: 6.0137 - acc: 0.0083 - val_loss: 4.7984 - val_acc: 0.0000e+00
    Epoch 19/100
    363/363 [==============================] - 0s 47us/step - loss: 5.9655 - acc: 0.0165 - val_loss: 4.6831 - val_acc: 0.0244
    Epoch 20/100
    363/363 [==============================] - 0s 52us/step - loss: 5.9495 - acc: 0.0110 - val_loss: 4.8068 - val_acc: 0.0000e+00
    Epoch 21/100
    363/363 [==============================] - ETA: 0s - loss: 7.6398 - acc: 0.015 - 0s 49us/step - loss: 5.9347 - acc: 0.0165 - val_loss: 4.7321 - val_acc: 0.0000e+00
    Epoch 22/100
    363/363 [==============================] - 0s 47us/step - loss: 5.9356 - acc: 0.0138 - val_loss: 4.7608 - val_acc: 0.0000e+00
    Epoch 23/100
    363/363 [==============================] - 0s 58us/step - loss: 5.9148 - acc: 0.0165 - val_loss: 4.6238 - val_acc: 0.0000e+00
    Epoch 24/100
    363/363 [==============================] - 0s 69us/step - loss: 5.9201 - acc: 0.0110 - val_loss: 4.7594 - val_acc: 0.0000e+00
    Epoch 25/100
    363/363 [==============================] - 0s 52us/step - loss: 5.9007 - acc: 0.0110 - val_loss: 4.5282 - val_acc: 0.0244
    Epoch 26/100
    363/363 [==============================] - 0s 80us/step - loss: 5.8710 - acc: 0.0138 - val_loss: 4.8214 - val_acc: 0.0000e+00
    Epoch 27/100
    363/363 [==============================] - 0s 60us/step - loss: 5.8383 - acc: 0.0220 - val_loss: 4.5801 - val_acc: 0.0000e+00
    Epoch 28/100
    363/363 [==============================] - 0s 80us/step - loss: 5.8239 - acc: 0.0220 - val_loss: 4.7789 - val_acc: 0.0000e+00
    Epoch 29/100
    363/363 [==============================] - 0s 58us/step - loss: 5.8231 - acc: 0.0110 - val_loss: 4.6579 - val_acc: 0.0000e+00
    Epoch 30/100
    363/363 [==============================] - 0s 60us/step - loss: 5.8157 - acc: 0.0138 - val_loss: 4.6242 - val_acc: 0.0000e+00
    Epoch 31/100
    363/363 [==============================] - 0s 52us/step - loss: 5.7825 - acc: 0.0193 - val_loss: 4.6243 - val_acc: 0.0000e+00
    Epoch 32/100
    363/363 [==============================] - 0s 58us/step - loss: 5.7746 - acc: 0.0220 - val_loss: 4.6317 - val_acc: 0.0000e+00
    Epoch 33/100
    363/363 [==============================] - 0s 49us/step - loss: 5.7655 - acc: 0.0165 - val_loss: 4.5714 - val_acc: 0.0000e+00
    Epoch 34/100
    363/363 [==============================] - 0s 88us/step - loss: 5.7499 - acc: 0.0193 - val_loss: 4.5590 - val_acc: 0.0000e+00
    Epoch 35/100
    363/363 [==============================] - 0s 60us/step - loss: 5.7354 - acc: 0.0193 - val_loss: 4.6802 - val_acc: 0.0000e+00
    Epoch 36/100
    363/363 [==============================] - 0s 55us/step - loss: 5.7372 - acc: 0.0165 - val_loss: 4.6049 - val_acc: 0.0000e+00
    Epoch 37/100
    363/363 [==============================] - 0s 38us/step - loss: 5.7148 - acc: 0.0165 - val_loss: 4.6634 - val_acc: 0.0000e+00
    Epoch 38/100
    363/363 [==============================] - 0s 44us/step - loss: 5.6977 - acc: 0.0138 - val_loss: 4.5502 - val_acc: 0.0244
    Epoch 39/100
    363/363 [==============================] - 0s 55us/step - loss: 5.6713 - acc: 0.0193 - val_loss: 4.6889 - val_acc: 0.0000e+00
    Epoch 40/100
    363/363 [==============================] - 0s 49us/step - loss: 5.6743 - acc: 0.0248 - val_loss: 4.6446 - val_acc: 0.0000e+00
    Epoch 41/100
    363/363 [==============================] - 0s 63us/step - loss: 5.6475 - acc: 0.0165 - val_loss: 4.5411 - val_acc: 0.0244
    Epoch 42/100
    363/363 [==============================] - 0s 49us/step - loss: 5.6345 - acc: 0.0248 - val_loss: 4.5691 - val_acc: 0.0244
    Epoch 43/100
    363/363 [==============================] - 0s 38us/step - loss: 5.6582 - acc: 0.0110 - val_loss: 4.6198 - val_acc: 0.0000e+00
    Epoch 44/100
    363/363 [==============================] - 0s 52us/step - loss: 5.6740 - acc: 0.0165 - val_loss: 4.4955 - val_acc: 0.0244
    Epoch 45/100
    363/363 [==============================] - 0s 33us/step - loss: 5.6975 - acc: 0.0055 - val_loss: 4.6717 - val_acc: 0.0000e+00
    Epoch 46/100
    363/363 [==============================] - 0s 44us/step - loss: 5.6026 - acc: 0.0138 - val_loss: 4.3592 - val_acc: 0.0244
    Epoch 47/100
    363/363 [==============================] - 0s 71us/step - loss: 5.5589 - acc: 0.0165 - val_loss: 4.7023 - val_acc: 0.0244
    Epoch 48/100
    363/363 [==============================] - 0s 49us/step - loss: 5.5779 - acc: 0.0165 - val_loss: 4.4802 - val_acc: 0.0244
    Epoch 49/100
    363/363 [==============================] - 0s 47us/step - loss: 5.5291 - acc: 0.0110 - val_loss: 4.6373 - val_acc: 0.0244
    Epoch 50/100
    363/363 [==============================] - 0s 52us/step - loss: 5.5285 - acc: 0.0165 - val_loss: 4.3598 - val_acc: 0.0000e+00
    Epoch 51/100
    363/363 [==============================] - 0s 41us/step - loss: 5.5238 - acc: 0.0248 - val_loss: 4.5611 - val_acc: 0.0244
    Epoch 52/100
    363/363 [==============================] - 0s 55us/step - loss: 5.5188 - acc: 0.0193 - val_loss: 4.3919 - val_acc: 0.0000e+00
    Epoch 53/100
    363/363 [==============================] - 0s 36us/step - loss: 5.4642 - acc: 0.0220 - val_loss: 4.8235 - val_acc: 0.0244
    Epoch 54/100
    363/363 [==============================] - 0s 41us/step - loss: 5.4808 - acc: 0.0055 - val_loss: 4.2942 - val_acc: 0.0244
    Epoch 55/100
    363/363 [==============================] - 0s 58us/step - loss: 5.4767 - acc: 0.0193 - val_loss: 4.6189 - val_acc: 0.0244
    Epoch 56/100
    363/363 [==============================] - 0s 38us/step - loss: 5.5464 - acc: 0.0028 - val_loss: 4.3824 - val_acc: 0.0000e+00
    Epoch 57/100
    363/363 [==============================] - 0s 93us/step - loss: 5.4499 - acc: 0.0165 - val_loss: 4.3841 - val_acc: 0.0000e+00
    Epoch 58/100
    363/363 [==============================] - 0s 58us/step - loss: 5.4469 - acc: 0.0165 - val_loss: 4.5006 - val_acc: 0.0244
    Epoch 59/100
    363/363 [==============================] - 0s 60us/step - loss: 5.3735 - acc: 0.0138 - val_loss: 4.3970 - val_acc: 0.0000e+00
    Epoch 60/100
    363/363 [==============================] - 0s 44us/step - loss: 5.3702 - acc: 0.0193 - val_loss: 4.7498 - val_acc: 0.0488
    Epoch 61/100
    363/363 [==============================] - 0s 52us/step - loss: 5.3834 - acc: 0.0110 - val_loss: 4.3253 - val_acc: 0.0244
    Epoch 62/100
    363/363 [==============================] - 0s 44us/step - loss: 5.3307 - acc: 0.0193 - val_loss: 4.6236 - val_acc: 0.0244
    Epoch 63/100
    363/363 [==============================] - 0s 44us/step - loss: 5.3079 - acc: 0.0138 - val_loss: 4.4797 - val_acc: 0.0000e+00
    Epoch 64/100
    363/363 [==============================] - 0s 49us/step - loss: 5.3287 - acc: 0.0138 - val_loss: 4.7198 - val_acc: 0.0244
    Epoch 65/100
    363/363 [==============================] - 0s 49us/step - loss: 5.2753 - acc: 0.0138 - val_loss: 4.4145 - val_acc: 0.0000e+00
    Epoch 66/100
    363/363 [==============================] - 0s 36us/step - loss: 5.2438 - acc: 0.0083 - val_loss: 4.5651 - val_acc: 0.0000e+00
    Epoch 67/100
    363/363 [==============================] - 0s 41us/step - loss: 5.2137 - acc: 0.0110 - val_loss: 4.4375 - val_acc: 0.0000e+00
    Epoch 68/100
    363/363 [==============================] - 0s 60us/step - loss: 5.1948 - acc: 0.0138 - val_loss: 4.4903 - val_acc: 0.0000e+00
    Epoch 69/100
    363/363 [==============================] - 0s 91us/step - loss: 5.1922 - acc: 0.0165 - val_loss: 4.6759 - val_acc: 0.0000e+00
    Epoch 70/100
    363/363 [==============================] - 0s 66us/step - loss: 5.2703 - acc: 0.0083 - val_loss: 4.2311 - val_acc: 0.0488
    Epoch 71/100
    363/363 [==============================] - 0s 52us/step - loss: 5.1639 - acc: 0.0193 - val_loss: 4.9564 - val_acc: 0.0488
    Epoch 72/100
    363/363 [==============================] - ETA: 0s - loss: 5.0708 - acc: 0.015 - 0s 66us/step - loss: 5.3052 - acc: 0.0165 - val_loss: 4.3306 - val_acc: 0.0000e+00
    Epoch 73/100
    363/363 [==============================] - 0s 55us/step - loss: 5.2681 - acc: 0.0055 - val_loss: 4.3267 - val_acc: 0.0000e+00
    Epoch 74/100
    363/363 [==============================] - 0s 52us/step - loss: 5.2264 - acc: 0.0028 - val_loss: 4.5921 - val_acc: 0.0000e+00
    Epoch 75/100
    363/363 [==============================] - 0s 60us/step - loss: 5.2670 - acc: 0.0110 - val_loss: 4.3898 - val_acc: 0.0000e+00
    Epoch 76/100
    363/363 [==============================] - 0s 58us/step - loss: 5.0939 - acc: 0.0138 - val_loss: 4.7597 - val_acc: 0.0244
    Epoch 77/100
    363/363 [==============================] - 0s 96us/step - loss: 5.0861 - acc: 0.0083 - val_loss: 4.3564 - val_acc: 0.0000e+00
    Epoch 78/100
    363/363 [==============================] - 0s 44us/step - loss: 5.0469 - acc: 0.0220 - val_loss: 4.5375 - val_acc: 0.0000e+00
    Epoch 79/100
    363/363 [==============================] - 0s 107us/step - loss: 5.0171 - acc: 0.0138 - val_loss: 4.3147 - val_acc: 0.0244
    Epoch 80/100
    363/363 [==============================] - 0s 58us/step - loss: 5.0690 - acc: 0.0083 - val_loss: 4.8726 - val_acc: 0.0244
    Epoch 81/100
    363/363 [==============================] - 0s 63us/step - loss: 5.2675 - acc: 0.0110 - val_loss: 4.2468 - val_acc: 0.0000e+00
    Epoch 82/100
    363/363 [==============================] - 0s 55us/step - loss: 5.1035 - acc: 0.0165 - val_loss: 4.7976 - val_acc: 0.0244
    Epoch 83/100
    363/363 [==============================] - 0s 58us/step - loss: 5.1877 - acc: 0.0110 - val_loss: 4.4285 - val_acc: 0.0000e+00
    Epoch 84/100
    363/363 [==============================] - 0s 102us/step - loss: 5.0233 - acc: 0.0138 - val_loss: 4.3801 - val_acc: 0.0000e+00
    Epoch 85/100
    363/363 [==============================] - 0s 58us/step - loss: 4.9793 - acc: 0.0165 - val_loss: 4.4907 - val_acc: 0.0000e+00
    Epoch 86/100
    363/363 [==============================] - 0s 47us/step - loss: 4.9149 - acc: 0.0165 - val_loss: 4.5768 - val_acc: 0.0000e+00
    Epoch 87/100
    363/363 [==============================] - 0s 41us/step - loss: 4.8994 - acc: 0.0138 - val_loss: 4.4939 - val_acc: 0.0000e+00
    Epoch 88/100
    363/363 [==============================] - 0s 60us/step - loss: 4.9016 - acc: 0.0165 - val_loss: 4.4344 - val_acc: 0.0000e+00
    Epoch 89/100
    363/363 [==============================] - 0s 36us/step - loss: 4.9075 - acc: 0.0110 - val_loss: 4.6966 - val_acc: 0.0244
    Epoch 90/100
    363/363 [==============================] - 0s 63us/step - loss: 4.8838 - acc: 0.0138 - val_loss: 4.4814 - val_acc: 0.0000e+00
    Epoch 91/100
    363/363 [==============================] - 0s 49us/step - loss: 4.8627 - acc: 0.0110 - val_loss: 4.3427 - val_acc: 0.0244
    Epoch 92/100
    363/363 [==============================] - 0s 30us/step - loss: 4.8235 - acc: 0.0110 - val_loss: 4.6968 - val_acc: 0.0000e+00
    Epoch 93/100
    363/363 [==============================] - 0s 58us/step - loss: 4.8011 - acc: 0.0110 - val_loss: 4.1852 - val_acc: 0.0488
    Epoch 94/100
    363/363 [==============================] - 0s 63us/step - loss: 4.8682 - acc: 0.0055 - val_loss: 4.8373 - val_acc: 0.0244
    Epoch 95/100
    363/363 [==============================] - 0s 60us/step - loss: 4.8072 - acc: 0.0083 - val_loss: 4.2787 - val_acc: 0.0488
    Epoch 96/100
    363/363 [==============================] - 0s 60us/step - loss: 4.7688 - acc: 0.0110 - val_loss: 4.6556 - val_acc: 0.0000e+00
    Epoch 97/100
    363/363 [==============================] - 0s 47us/step - loss: 4.8488 - acc: 0.0055 - val_loss: 4.1959 - val_acc: 0.0000e+00
    Epoch 98/100
    363/363 [==============================] - 0s 49us/step - loss: 4.9742 - acc: 0.0138 - val_loss: 4.8878 - val_acc: 0.0000e+00
    Epoch 99/100
    363/363 [==============================] - 0s 69us/step - loss: 4.7650 - acc: 0.0138 - val_loss: 4.4967 - val_acc: 0.0000e+00
    Epoch 100/100
    363/363 [==============================] - 0s 63us/step - loss: 4.7124 - acc: 0.0028 - val_loss: 4.4369 - val_acc: 0.0000e+00
    



.. parsed-literal::

    <keras.callbacks.History at 0x1f76ec3a860>



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
    Loss over the test dataset: 4.94
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
    Predicted price: [[18.685894]], Actual price: [15.2]
    Predicted price: [[17.7905]], Actual price: 12.1
    Predicted price: [[26.6879]], Actual price: 23.9
    
    
    Testing set points:
    Predicted price: [[19.549236]], Actual price: 18.8
    Predicted price: [[32.82071]], Actual price: 35.4
    Predicted price: [[24.151304]], Actual price: 26.7
    

Improve the model
=================

.. code:: ipython3

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


.. parsed-literal::

    Train on 323 samples, validate on 81 samples
    Epoch 1/150
    323/323 [==============================] - 1s 3ms/step - loss: 441.2293 - mean_absolute_error: 18.4925 - val_loss: 312.3085 - val_mean_absolute_error: 14.6657
    Epoch 2/150
    323/323 [==============================] - 0s 111us/step - loss: 180.0432 - mean_absolute_error: 10.5216 - val_loss: 127.5692 - val_mean_absolute_error: 8.8614
    Epoch 3/150
    323/323 [==============================] - 0s 164us/step - loss: 132.7880 - mean_absolute_error: 8.7863 - val_loss: 112.4879 - val_mean_absolute_error: 8.1239
    Epoch 4/150
    323/323 [==============================] - 0s 114us/step - loss: 106.3998 - mean_absolute_error: 7.4760 - val_loss: 116.6259 - val_mean_absolute_error: 7.8212
    Epoch 5/150
    323/323 [==============================] - 0s 127us/step - loss: 91.1631 - mean_absolute_error: 6.7190 - val_loss: 86.2448 - val_mean_absolute_error: 6.9438
    Epoch 6/150
    323/323 [==============================] - 0s 198us/step - loss: 79.1382 - mean_absolute_error: 6.5657 - val_loss: 78.5940 - val_mean_absolute_error: 6.1668
    Epoch 7/150
    323/323 [==============================] - 0s 176us/step - loss: 69.0903 - mean_absolute_error: 5.6188 - val_loss: 78.3495 - val_mean_absolute_error: 5.8493
    Epoch 8/150
    323/323 [==============================] - 0s 93us/step - loss: 64.7033 - mean_absolute_error: 5.6193 - val_loss: 70.2419 - val_mean_absolute_error: 5.7449
    Epoch 9/150
    323/323 [==============================] - 0s 164us/step - loss: 61.0968 - mean_absolute_error: 5.3857 - val_loss: 73.7760 - val_mean_absolute_error: 5.5508
    Epoch 10/150
    323/323 [==============================] - 0s 167us/step - loss: 60.8919 - mean_absolute_error: 5.3636 - val_loss: 68.1633 - val_mean_absolute_error: 5.6258
    Epoch 11/150
    323/323 [==============================] - 0s 83us/step - loss: 58.8582 - mean_absolute_error: 5.3599 - val_loss: 70.6243 - val_mean_absolute_error: 5.4838
    Epoch 12/150
    323/323 [==============================] - 0s 111us/step - loss: 58.3174 - mean_absolute_error: 5.1947 - val_loss: 67.5091 - val_mean_absolute_error: 5.5406
    Epoch 13/150
    323/323 [==============================] - 0s 93us/step - loss: 57.2564 - mean_absolute_error: 5.3294 - val_loss: 69.3033 - val_mean_absolute_error: 5.4322
    Epoch 14/150
    323/323 [==============================] - 0s 130us/step - loss: 56.4566 - mean_absolute_error: 5.0760 - val_loss: 66.6419 - val_mean_absolute_error: 5.5080
    Epoch 15/150
    323/323 [==============================] - 0s 108us/step - loss: 56.9345 - mean_absolute_error: 5.5206 - val_loss: 67.6111 - val_mean_absolute_error: 5.3826
    Epoch 16/150
    323/323 [==============================] - 0s 108us/step - loss: 55.8253 - mean_absolute_error: 5.0098 - val_loss: 65.8577 - val_mean_absolute_error: 5.3946
    Epoch 17/150
    323/323 [==============================] - 0s 99us/step - loss: 54.2151 - mean_absolute_error: 5.1348 - val_loss: 66.5187 - val_mean_absolute_error: 5.3260
    Epoch 18/150
    323/323 [==============================] - 0s 111us/step - loss: 53.5936 - mean_absolute_error: 4.9905 - val_loss: 68.2964 - val_mean_absolute_error: 5.2664
    Epoch 19/150
    323/323 [==============================] - 0s 90us/step - loss: 53.3867 - mean_absolute_error: 4.8189 - val_loss: 64.1012 - val_mean_absolute_error: 5.3639
    Epoch 20/150
    323/323 [==============================] - 0s 108us/step - loss: 52.5234 - mean_absolute_error: 5.0111 - val_loss: 66.0861 - val_mean_absolute_error: 5.2389
    Epoch 21/150
    323/323 [==============================] - 0s 102us/step - loss: 52.3450 - mean_absolute_error: 4.9158 - val_loss: 67.9363 - val_mean_absolute_error: 5.1957
    Epoch 22/150
    323/323 [==============================] - 0s 111us/step - loss: 53.9175 - mean_absolute_error: 4.7033 - val_loss: 62.0455 - val_mean_absolute_error: 5.2796
    Epoch 23/150
    323/323 [==============================] - 0s 117us/step - loss: 51.7349 - mean_absolute_error: 4.9790 - val_loss: 63.2456 - val_mean_absolute_error: 5.1503
    Epoch 24/150
    323/323 [==============================] - 0s 102us/step - loss: 50.5677 - mean_absolute_error: 4.9496 - val_loss: 64.9350 - val_mean_absolute_error: 5.1131
    Epoch 25/150
    323/323 [==============================] - 0s 80us/step - loss: 50.1371 - mean_absolute_error: 4.6265 - val_loss: 60.3325 - val_mean_absolute_error: 5.1448
    Epoch 26/150
    323/323 [==============================] - 0s 114us/step - loss: 48.9439 - mean_absolute_error: 4.7914 - val_loss: 61.2671 - val_mean_absolute_error: 5.0559
    Epoch 27/150
    323/323 [==============================] - 0s 117us/step - loss: 49.5983 - mean_absolute_error: 4.6138 - val_loss: 57.9667 - val_mean_absolute_error: 5.2939
    Epoch 28/150
    323/323 [==============================] - 0s 90us/step - loss: 48.7990 - mean_absolute_error: 5.0424 - val_loss: 66.4215 - val_mean_absolute_error: 5.0748
    Epoch 29/150
    323/323 [==============================] - 0s 99us/step - loss: 50.4781 - mean_absolute_error: 4.6850 - val_loss: 56.9158 - val_mean_absolute_error: 5.1052
    Epoch 30/150
    323/323 [==============================] - 0s 71us/step - loss: 47.0944 - mean_absolute_error: 4.6501 - val_loss: 56.9048 - val_mean_absolute_error: 4.9568
    Epoch 31/150
    323/323 [==============================] - 0s 86us/step - loss: 46.5349 - mean_absolute_error: 4.5881 - val_loss: 56.4728 - val_mean_absolute_error: 4.9232
    Epoch 32/150
    323/323 [==============================] - 0s 86us/step - loss: 45.8473 - mean_absolute_error: 4.4801 - val_loss: 55.7129 - val_mean_absolute_error: 5.0328
    Epoch 33/150
    323/323 [==============================] - 0s 80us/step - loss: 45.0854 - mean_absolute_error: 4.6145 - val_loss: 54.9119 - val_mean_absolute_error: 4.8668
    Epoch 34/150
    323/323 [==============================] - 0s 96us/step - loss: 43.8529 - mean_absolute_error: 4.4730 - val_loss: 56.9102 - val_mean_absolute_error: 4.8244
    Epoch 35/150
    323/323 [==============================] - 0s 68us/step - loss: 44.0612 - mean_absolute_error: 4.3544 - val_loss: 52.0027 - val_mean_absolute_error: 4.9563
    Epoch 36/150
    323/323 [==============================] - 0s 86us/step - loss: 45.9435 - mean_absolute_error: 4.8693 - val_loss: 56.8685 - val_mean_absolute_error: 4.8228
    Epoch 37/150
    323/323 [==============================] - 0s 96us/step - loss: 41.7431 - mean_absolute_error: 4.4434 - val_loss: 51.6266 - val_mean_absolute_error: 4.8318
    Epoch 38/150
    323/323 [==============================] - 0s 102us/step - loss: 41.1683 - mean_absolute_error: 4.3553 - val_loss: 50.9732 - val_mean_absolute_error: 4.7824
    Epoch 39/150
    323/323 [==============================] - 0s 83us/step - loss: 41.3823 - mean_absolute_error: 4.6613 - val_loss: 59.3209 - val_mean_absolute_error: 4.9515
    Epoch 40/150
    323/323 [==============================] - 0s 108us/step - loss: 42.8296 - mean_absolute_error: 4.3896 - val_loss: 49.9025 - val_mean_absolute_error: 5.1980
    Epoch 41/150
    323/323 [==============================] - 0s 99us/step - loss: 41.5078 - mean_absolute_error: 4.4570 - val_loss: 51.0826 - val_mean_absolute_error: 4.6753
    Epoch 42/150
    323/323 [==============================] - 0s 99us/step - loss: 39.4279 - mean_absolute_error: 4.2967 - val_loss: 47.4825 - val_mean_absolute_error: 4.6564
    Epoch 43/150
    323/323 [==============================] - 0s 65us/step - loss: 38.5490 - mean_absolute_error: 4.1442 - val_loss: 46.4433 - val_mean_absolute_error: 4.7347
    Epoch 44/150
    323/323 [==============================] - 0s 105us/step - loss: 38.2773 - mean_absolute_error: 4.3749 - val_loss: 48.0886 - val_mean_absolute_error: 4.5525
    Epoch 45/150
    323/323 [==============================] - 0s 80us/step - loss: 37.1173 - mean_absolute_error: 4.2342 - val_loss: 46.2272 - val_mean_absolute_error: 4.5333
    Epoch 46/150
    323/323 [==============================] - 0s 93us/step - loss: 36.8999 - mean_absolute_error: 4.2354 - val_loss: 49.7590 - val_mean_absolute_error: 4.5478
    Epoch 47/150
    323/323 [==============================] - 0s 96us/step - loss: 37.7824 - mean_absolute_error: 4.2206 - val_loss: 45.9764 - val_mean_absolute_error: 4.4536
    Epoch 48/150
    323/323 [==============================] - 0s 108us/step - loss: 35.7427 - mean_absolute_error: 4.1498 - val_loss: 47.3047 - val_mean_absolute_error: 4.4463
    Epoch 49/150
    323/323 [==============================] - 0s 139us/step - loss: 36.8990 - mean_absolute_error: 4.1119 - val_loss: 42.4641 - val_mean_absolute_error: 4.4948
    Epoch 50/150
    323/323 [==============================] - 0s 120us/step - loss: 34.4384 - mean_absolute_error: 4.0596 - val_loss: 46.3373 - val_mean_absolute_error: 4.4160
    Epoch 51/150
    323/323 [==============================] - 0s 130us/step - loss: 34.5518 - mean_absolute_error: 4.2110 - val_loss: 45.0328 - val_mean_absolute_error: 4.3691
    Epoch 52/150
    323/323 [==============================] - 0s 108us/step - loss: 34.6153 - mean_absolute_error: 4.0479 - val_loss: 39.9536 - val_mean_absolute_error: 4.4301
    Epoch 53/150
    323/323 [==============================] - 0s 136us/step - loss: 33.0776 - mean_absolute_error: 3.9164 - val_loss: 39.4610 - val_mean_absolute_error: 4.5938
    Epoch 54/150
    323/323 [==============================] - 0s 117us/step - loss: 34.4751 - mean_absolute_error: 4.3230 - val_loss: 43.4420 - val_mean_absolute_error: 4.3232
    Epoch 55/150
    323/323 [==============================] - 0s 114us/step - loss: 32.4040 - mean_absolute_error: 4.0483 - val_loss: 40.7306 - val_mean_absolute_error: 4.2572
    Epoch 56/150
    323/323 [==============================] - 0s 90us/step - loss: 34.3240 - mean_absolute_error: 4.1313 - val_loss: 37.9974 - val_mean_absolute_error: 4.3286
    Epoch 57/150
    323/323 [==============================] - 0s 105us/step - loss: 31.8121 - mean_absolute_error: 3.8441 - val_loss: 37.3102 - val_mean_absolute_error: 4.4199
    Epoch 58/150
    323/323 [==============================] - 0s 99us/step - loss: 31.1380 - mean_absolute_error: 4.0875 - val_loss: 44.5791 - val_mean_absolute_error: 4.3534
    Epoch 59/150
    323/323 [==============================] - 0s 105us/step - loss: 31.8495 - mean_absolute_error: 3.8364 - val_loss: 36.1283 - val_mean_absolute_error: 4.3315
    Epoch 60/150
    323/323 [==============================] - 0s 124us/step - loss: 32.6243 - mean_absolute_error: 4.1407 - val_loss: 36.5502 - val_mean_absolute_error: 4.0770
    Epoch 61/150
    323/323 [==============================] - 0s 108us/step - loss: 29.6975 - mean_absolute_error: 3.7945 - val_loss: 35.0168 - val_mean_absolute_error: 4.0836
    Epoch 62/150
    323/323 [==============================] - 0s 130us/step - loss: 29.3467 - mean_absolute_error: 3.7930 - val_loss: 33.9699 - val_mean_absolute_error: 4.1377
    Epoch 63/150
    323/323 [==============================] - 0s 93us/step - loss: 29.2032 - mean_absolute_error: 3.8223 - val_loss: 33.6350 - val_mean_absolute_error: 4.0950
    Epoch 64/150
    323/323 [==============================] - 0s 90us/step - loss: 28.7959 - mean_absolute_error: 3.8196 - val_loss: 33.1522 - val_mean_absolute_error: 4.0856
    Epoch 65/150
    323/323 [==============================] - 0s 105us/step - loss: 29.0926 - mean_absolute_error: 3.7498 - val_loss: 36.0939 - val_mean_absolute_error: 4.7199
    Epoch 66/150
    323/323 [==============================] - 0s 99us/step - loss: 32.0112 - mean_absolute_error: 4.1600 - val_loss: 32.2544 - val_mean_absolute_error: 4.0161
    Epoch 67/150
    323/323 [==============================] - 0s 130us/step - loss: 30.7143 - mean_absolute_error: 4.1967 - val_loss: 32.4940 - val_mean_absolute_error: 3.8982
    Epoch 68/150
    323/323 [==============================] - 0s 133us/step - loss: 28.9669 - mean_absolute_error: 4.0231 - val_loss: 43.1247 - val_mean_absolute_error: 4.3823
    Epoch 69/150
    323/323 [==============================] - 0s 108us/step - loss: 29.8454 - mean_absolute_error: 3.9362 - val_loss: 31.1816 - val_mean_absolute_error: 3.8808
    Epoch 70/150
    323/323 [==============================] - 0s 105us/step - loss: 26.7867 - mean_absolute_error: 3.6840 - val_loss: 31.2587 - val_mean_absolute_error: 3.7957
    Epoch 71/150
    323/323 [==============================] - 0s 90us/step - loss: 26.5554 - mean_absolute_error: 3.7542 - val_loss: 31.4812 - val_mean_absolute_error: 3.7975
    Epoch 72/150
    323/323 [==============================] - 0s 99us/step - loss: 25.9302 - mean_absolute_error: 3.7868 - val_loss: 30.9061 - val_mean_absolute_error: 3.7564
    Epoch 73/150
    323/323 [==============================] - 0s 127us/step - loss: 26.7305 - mean_absolute_error: 3.8275 - val_loss: 30.1014 - val_mean_absolute_error: 3.7442
    Epoch 74/150
    323/323 [==============================] - 0s 86us/step - loss: 26.5178 - mean_absolute_error: 3.6670 - val_loss: 29.3555 - val_mean_absolute_error: 3.7266
    Epoch 75/150
    323/323 [==============================] - 0s 99us/step - loss: 25.2594 - mean_absolute_error: 3.6477 - val_loss: 28.5060 - val_mean_absolute_error: 3.8260
    Epoch 76/150
    323/323 [==============================] - 0s 127us/step - loss: 27.5908 - mean_absolute_error: 3.8750 - val_loss: 28.6590 - val_mean_absolute_error: 3.7161
    Epoch 77/150
    323/323 [==============================] - 0s 157us/step - loss: 30.3430 - mean_absolute_error: 4.0764 - val_loss: 35.5270 - val_mean_absolute_error: 3.9889
    Epoch 78/150
    323/323 [==============================] - 0s 102us/step - loss: 26.4705 - mean_absolute_error: 3.9108 - val_loss: 33.0160 - val_mean_absolute_error: 3.8497
    Epoch 79/150
    323/323 [==============================] - 0s 133us/step - loss: 25.7526 - mean_absolute_error: 3.7409 - val_loss: 27.3364 - val_mean_absolute_error: 3.7062
    Epoch 80/150
    323/323 [==============================] - 0s 114us/step - loss: 26.0399 - mean_absolute_error: 3.6916 - val_loss: 27.3620 - val_mean_absolute_error: 3.6953
    Epoch 81/150
    323/323 [==============================] - 0s 102us/step - loss: 25.4888 - mean_absolute_error: 3.5880 - val_loss: 28.5985 - val_mean_absolute_error: 4.0327
    Epoch 82/150
    323/323 [==============================] - 0s 114us/step - loss: 23.3840 - mean_absolute_error: 3.5941 - val_loss: 32.2617 - val_mean_absolute_error: 3.8207
    Epoch 83/150
    323/323 [==============================] - 0s 102us/step - loss: 24.9312 - mean_absolute_error: 3.6610 - val_loss: 26.4247 - val_mean_absolute_error: 3.5834
    Epoch 84/150
    323/323 [==============================] - 0s 170us/step - loss: 23.6245 - mean_absolute_error: 3.6072 - val_loss: 28.9296 - val_mean_absolute_error: 3.6497
    Epoch 85/150
    323/323 [==============================] - 0s 74us/step - loss: 23.5063 - mean_absolute_error: 3.5415 - val_loss: 25.9023 - val_mean_absolute_error: 3.7372
    Epoch 86/150
    323/323 [==============================] - 0s 99us/step - loss: 23.9279 - mean_absolute_error: 3.5793 - val_loss: 28.0236 - val_mean_absolute_error: 3.5773
    Epoch 87/150
    323/323 [==============================] - 0s 80us/step - loss: 23.4526 - mean_absolute_error: 3.6789 - val_loss: 29.9783 - val_mean_absolute_error: 3.7274
    Epoch 88/150
    323/323 [==============================] - 0s 114us/step - loss: 24.6120 - mean_absolute_error: 3.6356 - val_loss: 24.6421 - val_mean_absolute_error: 3.5490
    Epoch 89/150
    323/323 [==============================] - 0s 117us/step - loss: 22.3409 - mean_absolute_error: 3.5074 - val_loss: 27.7220 - val_mean_absolute_error: 3.6138
    Epoch 90/150
    323/323 [==============================] - 0s 99us/step - loss: 21.9143 - mean_absolute_error: 3.4541 - val_loss: 24.0221 - val_mean_absolute_error: 3.5527
    Epoch 91/150
    323/323 [==============================] - 0s 117us/step - loss: 22.1789 - mean_absolute_error: 3.5516 - val_loss: 25.6153 - val_mean_absolute_error: 3.4726
    Epoch 92/150
    323/323 [==============================] - 0s 133us/step - loss: 22.1095 - mean_absolute_error: 3.3870 - val_loss: 29.8823 - val_mean_absolute_error: 4.3517
    Epoch 93/150
    323/323 [==============================] - 0s 93us/step - loss: 30.8875 - mean_absolute_error: 4.3396 - val_loss: 30.2240 - val_mean_absolute_error: 3.8859
    Epoch 94/150
    323/323 [==============================] - 0s 179us/step - loss: 22.6463 - mean_absolute_error: 3.6161 - val_loss: 23.7627 - val_mean_absolute_error: 3.5001
    Epoch 95/150
    323/323 [==============================] - 0s 111us/step - loss: 23.7859 - mean_absolute_error: 3.6111 - val_loss: 24.7483 - val_mean_absolute_error: 3.4472
    Epoch 96/150
    323/323 [==============================] - 0s 114us/step - loss: 22.6774 - mean_absolute_error: 3.5992 - val_loss: 28.9577 - val_mean_absolute_error: 3.7688
    Epoch 97/150
    323/323 [==============================] - 0s 108us/step - loss: 23.2652 - mean_absolute_error: 3.7063 - val_loss: 26.5150 - val_mean_absolute_error: 3.5744
    Epoch 98/150
    323/323 [==============================] - 0s 222us/step - loss: 22.5006 - mean_absolute_error: 3.4559 - val_loss: 24.7994 - val_mean_absolute_error: 3.7815
    Epoch 99/150
    323/323 [==============================] - 0s 114us/step - loss: 22.7324 - mean_absolute_error: 3.4734 - val_loss: 22.5494 - val_mean_absolute_error: 3.4167
    Epoch 100/150
    323/323 [==============================] - 0s 111us/step - loss: 21.7418 - mean_absolute_error: 3.5047 - val_loss: 23.0724 - val_mean_absolute_error: 3.3167
    Epoch 101/150
    323/323 [==============================] - 0s 96us/step - loss: 20.8496 - mean_absolute_error: 3.4188 - val_loss: 23.9346 - val_mean_absolute_error: 3.3517
    Epoch 102/150
    323/323 [==============================] - 0s 86us/step - loss: 24.5403 - mean_absolute_error: 3.7138 - val_loss: 21.5352 - val_mean_absolute_error: 3.3895
    Epoch 103/150
    323/323 [==============================] - 0s 83us/step - loss: 23.7886 - mean_absolute_error: 3.6531 - val_loss: 23.9480 - val_mean_absolute_error: 3.7927
    Epoch 104/150
    323/323 [==============================] - 0s 78us/step - loss: 21.1716 - mean_absolute_error: 3.5114 - val_loss: 22.1290 - val_mean_absolute_error: 3.5046
    Epoch 105/150
    323/323 [==============================] - 0s 109us/step - loss: 21.5641 - mean_absolute_error: 3.4504 - val_loss: 21.9887 - val_mean_absolute_error: 3.2420
    Epoch 106/150
    323/323 [==============================] - 0s 80us/step - loss: 20.0331 - mean_absolute_error: 3.3342 - val_loss: 21.3270 - val_mean_absolute_error: 3.2622
    Epoch 107/150
    323/323 [==============================] - 0s 102us/step - loss: 19.4168 - mean_absolute_error: 3.2637 - val_loss: 21.0099 - val_mean_absolute_error: 3.2949
    Epoch 108/150
    323/323 [==============================] - 0s 96us/step - loss: 19.7738 - mean_absolute_error: 3.2953 - val_loss: 20.6542 - val_mean_absolute_error: 3.2306
    Epoch 109/150
    323/323 [==============================] - 0s 90us/step - loss: 19.4382 - mean_absolute_error: 3.2961 - val_loss: 21.1207 - val_mean_absolute_error: 3.1743
    Epoch 110/150
    323/323 [==============================] - 0s 80us/step - loss: 19.6162 - mean_absolute_error: 3.2781 - val_loss: 20.3364 - val_mean_absolute_error: 3.2310
    Epoch 111/150
    323/323 [==============================] - 0s 83us/step - loss: 19.8529 - mean_absolute_error: 3.2592 - val_loss: 20.6175 - val_mean_absolute_error: 3.2727
    Epoch 112/150
    323/323 [==============================] - 0s 90us/step - loss: 19.2153 - mean_absolute_error: 3.2420 - val_loss: 20.0234 - val_mean_absolute_error: 3.2534
    Epoch 113/150
    323/323 [==============================] - 0s 78us/step - loss: 18.9983 - mean_absolute_error: 3.1667 - val_loss: 20.2347 - val_mean_absolute_error: 3.1171
    Epoch 114/150
    323/323 [==============================] - 0s 97us/step - loss: 19.0742 - mean_absolute_error: 3.2454 - val_loss: 21.5637 - val_mean_absolute_error: 3.2611
    Epoch 115/150
    323/323 [==============================] - 0s 97us/step - loss: 19.0644 - mean_absolute_error: 3.2458 - val_loss: 20.4283 - val_mean_absolute_error: 3.3456
    Epoch 116/150
    323/323 [==============================] - 0s 97us/step - loss: 20.8252 - mean_absolute_error: 3.4158 - val_loss: 20.7886 - val_mean_absolute_error: 3.4649
    Epoch 117/150
    323/323 [==============================] - 0s 48us/step - loss: 21.7378 - mean_absolute_error: 3.5217 - val_loss: 19.7543 - val_mean_absolute_error: 3.1787
    Epoch 118/150
    323/323 [==============================] - 0s 48us/step - loss: 24.2011 - mean_absolute_error: 3.7176 - val_loss: 26.0985 - val_mean_absolute_error: 3.6635
    Epoch 119/150
    323/323 [==============================] - 0s 48us/step - loss: 21.9768 - mean_absolute_error: 3.6128 - val_loss: 27.3391 - val_mean_absolute_error: 3.6988
    Epoch 120/150
    323/323 [==============================] - 0s 97us/step - loss: 20.9159 - mean_absolute_error: 3.3486 - val_loss: 20.2537 - val_mean_absolute_error: 3.1251
    Epoch 121/150
    323/323 [==============================] - 0s 48us/step - loss: 18.9807 - mean_absolute_error: 3.1812 - val_loss: 19.6301 - val_mean_absolute_error: 3.0882
    Epoch 122/150
    323/323 [==============================] - 0s 48us/step - loss: 18.1201 - mean_absolute_error: 3.1658 - val_loss: 18.6108 - val_mean_absolute_error: 3.1218
    Epoch 123/150
    323/323 [==============================] - 0s 97us/step - loss: 17.7919 - mean_absolute_error: 3.0726 - val_loss: 18.8387 - val_mean_absolute_error: 3.1368
    Epoch 124/150
    323/323 [==============================] - 0s 48us/step - loss: 17.8716 - mean_absolute_error: 3.1507 - val_loss: 20.8497 - val_mean_absolute_error: 3.2188
    Epoch 125/150
    323/323 [==============================] - 0s 97us/step - loss: 19.8196 - mean_absolute_error: 3.2635 - val_loss: 17.9193 - val_mean_absolute_error: 3.1357
    Epoch 126/150
    323/323 [==============================] - 0s 48us/step - loss: 18.8891 - mean_absolute_error: 3.1164 - val_loss: 22.2605 - val_mean_absolute_error: 3.6796
    Epoch 127/150
    323/323 [==============================] - 0s 48us/step - loss: 19.1283 - mean_absolute_error: 3.3006 - val_loss: 22.3630 - val_mean_absolute_error: 3.3212
    Epoch 128/150
    323/323 [==============================] - 0s 48us/step - loss: 17.7838 - mean_absolute_error: 3.1718 - val_loss: 19.9266 - val_mean_absolute_error: 3.1196
    Epoch 129/150
    323/323 [==============================] - 0s 48us/step - loss: 18.3795 - mean_absolute_error: 3.1554 - val_loss: 21.1316 - val_mean_absolute_error: 3.2318
    Epoch 130/150
    323/323 [==============================] - 0s 48us/step - loss: 21.3982 - mean_absolute_error: 3.3223 - val_loss: 19.2016 - val_mean_absolute_error: 3.0926
    Epoch 131/150
    323/323 [==============================] - 0s 48us/step - loss: 18.2629 - mean_absolute_error: 3.1325 - val_loss: 17.4681 - val_mean_absolute_error: 3.1105
    Epoch 132/150
    323/323 [==============================] - 0s 110us/step - loss: 18.5376 - mean_absolute_error: 3.1147 - val_loss: 18.6901 - val_mean_absolute_error: 3.2497
    Epoch 133/150
    323/323 [==============================] - 0s 32us/step - loss: 17.1632 - mean_absolute_error: 3.0488 - val_loss: 20.1927 - val_mean_absolute_error: 3.1479
    Epoch 134/150
    323/323 [==============================] - 0s 48us/step - loss: 17.4911 - mean_absolute_error: 3.0558 - val_loss: 17.7912 - val_mean_absolute_error: 2.9721
    Epoch 135/150
    323/323 [==============================] - 0s 48us/step - loss: 16.8063 - mean_absolute_error: 2.9615 - val_loss: 19.1894 - val_mean_absolute_error: 3.0984
    Epoch 136/150
    323/323 [==============================] - 0s 98us/step - loss: 21.3573 - mean_absolute_error: 3.3727 - val_loss: 17.7115 - val_mean_absolute_error: 3.0481
    Epoch 137/150
    323/323 [==============================] - 0s 86us/step - loss: 17.0557 - mean_absolute_error: 2.9824 - val_loss: 18.3122 - val_mean_absolute_error: 3.1958
    Epoch 138/150
    323/323 [==============================] - 0s 108us/step - loss: 17.0506 - mean_absolute_error: 3.0785 - val_loss: 17.2924 - val_mean_absolute_error: 3.0061
    Epoch 139/150
    323/323 [==============================] - 0s 114us/step - loss: 16.5535 - mean_absolute_error: 3.0254 - val_loss: 19.5172 - val_mean_absolute_error: 3.1474
    Epoch 140/150
    323/323 [==============================] - 0s 102us/step - loss: 18.3438 - mean_absolute_error: 3.1133 - val_loss: 17.1349 - val_mean_absolute_error: 2.9412
    Epoch 141/150
    323/323 [==============================] - 0s 127us/step - loss: 17.4415 - mean_absolute_error: 3.0075 - val_loss: 17.6426 - val_mean_absolute_error: 3.0064
    Epoch 142/150
    323/323 [==============================] - 0s 139us/step - loss: 22.4355 - mean_absolute_error: 3.4028 - val_loss: 23.6010 - val_mean_absolute_error: 4.0093
    Epoch 143/150
    323/323 [==============================] - 0s 102us/step - loss: 18.1588 - mean_absolute_error: 3.1391 - val_loss: 19.4816 - val_mean_absolute_error: 3.1096
    Epoch 144/150
    323/323 [==============================] - 0s 105us/step - loss: 16.5809 - mean_absolute_error: 3.0968 - val_loss: 16.9270 - val_mean_absolute_error: 2.9478
    Epoch 145/150
    323/323 [==============================] - 0s 96us/step - loss: 16.6051 - mean_absolute_error: 2.9902 - val_loss: 16.3353 - val_mean_absolute_error: 2.9492
    Epoch 146/150
    323/323 [==============================] - 0s 25us/step - loss: 17.8693 - mean_absolute_error: 3.0475 - val_loss: 16.4722 - val_mean_absolute_error: 2.9743
    Epoch 147/150
    323/323 [==============================] - 0s 48us/step - loss: 19.1125 - mean_absolute_error: 3.2168 - val_loss: 20.5508 - val_mean_absolute_error: 3.5925
    Epoch 148/150
    323/323 [==============================] - 0s 60us/step - loss: 18.7916 - mean_absolute_error: 3.1712 - val_loss: 18.9894 - val_mean_absolute_error: 3.0502
    Epoch 149/150
    323/323 [==============================] - 0s 48us/step - loss: 19.8153 - mean_absolute_error: 3.2266 - val_loss: 18.0748 - val_mean_absolute_error: 3.2624
    Epoch 150/150
    323/323 [==============================] - 0s 48us/step - loss: 17.2501 - mean_absolute_error: 2.9782 - val_loss: 18.9983 - val_mean_absolute_error: 3.3637
    
    
    Training set points:
    Predicted price: [[19.00385]], Actual price: [15.2]
    Predicted price: [[15.412333]], Actual price: 12.1
    Predicted price: [[27.759348]], Actual price: 23.9
    
    
    Testing set points:
    Predicted price: [[20.641031]], Actual price: 18.8
    Predicted price: [[38.48296]], Actual price: 35.4
    Predicted price: [[30.027851]], Actual price: 26.7
    

.. code:: ipython3

    test_model(improved_model, metric = "mae")


.. parsed-literal::

    -------------------------------------
    Loss over the test dataset: 33.99
    -------------------------------------
    Mean absolute error: 4.21
    

