import tensorflow as tf
from tensorflow.keras import layers, activations, regularizers, Input
import pandas as pd
import numpy as np

def Block2D(filters, input_tensor):
    
    x = layers.Conv2D(filters=filters, kernel_size=(3,3), padding='valid', kernel_initializer='he_normal')(input_tensor)
    x = activations.relu(x)
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)

    return x


# Network A and B
def network(input_shape):
    
    #Input
    input_img = Input(shape=input_shape, name='Image')

    
    # Convolutional layers
    x = Block2D(filters=16, input_tensor=input_img)

    x = Block2D(filters=32, input_tensor=x)
    
    x = Block2D(filters=64, input_tensor=x)
    
    x = Block2D(filters=128, input_tensor=x)
    
    x = Block2D(filters=18, input_tensor=x)
    
    # Fully-connected layers
    x = layers.Flatten()(x)
    
    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Outputs
    skin = layers.Dense(1, name='Skin')(x)
    fat = layers.Dense(1, name='Fat')(x)
    muscle = layers.Dense(1, name='Muscle')(x)

    model = tf.keras.Model(inputs=input_img, outputs=[skin,fat,muscle], name='network')
    
    return model


# Network C
def network_categorical(input_shape):
    
    #Inputs
    input_img = Input(shape=input_shape, name='Image')
    input_pos = Input(shape=(12), name='Position')
    input_ext = Input(shape=(4), name='Extremity')
    
    # Convolutional layers
    x = Block2D(filters=16, input_tensor=input_img)

    x = Block2D(filters=32, input_tensor=x)
    
    x = Block2D(filters=64, input_tensor=x)
    
    x = Block2D(filters=128, input_tensor=x)
    
    x = Block2D(filters=18, input_tensor=x)
    
    # Fully-connected layers
    x = layers.Flatten()(x)
    
    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    # Pre-outputs
    pre_output_fat_muscle = layers.Dense(1, kernel_initializer='he_normal')(x)
    x1 = layers.Concatenate()([pre_output_fat_muscle, input_pos, input_ext])
    x1 = layers.Dense(16, activation='relu', kernel_initializer='he_normal')(x1)
    x1 = layers.BatchNormalization()(x1)
    
    # Outputs
    skin = layers.Dense(1, name='Skin')(x)
    fat = layers.Dense(1, name='Fat')(x1)
    muscle = layers.Dense(1, name='Muscle')(x1)
    
    model = tf.keras.Model(inputs=[input_img, input_pos, input_ext], outputs=[skin,fat,muscle], name='categorical_network')
    
    return model


# Error function

def one_hot(x, size):
    
    shape = (x.size, size)
    one_hot = np.zeros(shape)
    rows = np.arange(x.size)
    one_hot[rows, x-1] = 1
    
    return one_hot

def error_function(x_img, x_ctg, y, df, models):
    # x_img: images
    # x_ctg: categorical inputs
    # y: true label
    # df: dataframe with all the data
    # models: the four models used in the 'ClevelandClinic_Phase1' script
    # categorical: True or False. This indicates if you will use x_ctg
    
    df_skin = pd.DataFrame()
    df_muscle = pd.DataFrame()
    df_fat = pd.DataFrame()
    
    if models[0].name=='categorical_network':
        out = models[0].predict({'Image': x_img, 'Extremity': one_hot(x_ctg[:,0], size=4),
                                  'Position': one_hot(x_ctg[:,1], size=12)})
    else:
        out = models[0].predict(x_img)
    out = np.array(out, dtype=np.float64)
    out = out.reshape(out.shape[:2]).T
    out = np.abs(out - y)
    df_skin['no_norm'] = out[:, 0]
    df_fat['no_norm'] = out[:, 1]
    df_muscle['no_norm'] = out[:, 2]
    print('finish no_normalization')

    if models[1].name=='categorical_network':
        out = models[1].predict({'Image': x_img, 'Extremity': one_hot(x_ctg[:,0], size=4),
                                  'Position': one_hot(x_ctg[:,1], size=12)})
    else:
        out = models[1].predict(x_img)
    out = np.array(out, dtype=np.float64)
    out = out.reshape(out.shape[:2]).T
    out = pd.DataFrame(out, columns=['Skin', 'Fat', 'Muscle'])
    out = out * df.std() + df.mean()
    out = np.abs(np.array(out) - y)
    df_skin['z_score'] = out[:, 0]
    df_fat['z_score'] = out[:, 1]
    df_muscle['z_score'] = out[:, 2]
    print('finish z_score')

    if models[2].name=='categorical_network':
        out = models[2].predict({'Image': x_img, 'Extremity': one_hot(x_ctg[:,0], size=4),
                                  'Position': one_hot(x_ctg[:,1], size=12)})
    else:
        out = models[2].predict(x_img)
    out = np.array(out, dtype=np.float64)
    out = out.reshape(out.shape[:2]).T
    out = pd.DataFrame(out, columns=['Skin', 'Fat', 'Muscle'])
    out = out * (df.max() - df.min()) + df.min()
    out = np.abs(np.array(out) - y)
    df_skin['min_max'] = out[:, 0]
    df_fat['min_max'] = out[:, 1]
    df_muscle['min_max'] = out[:, 2]
    print('finish min_max')

    if models[3].name=='categorical_network':
        out = models[3].predict({'Image': x_img, 'Extremity': one_hot(x_ctg[:,0], size=4),
                                  'Position': one_hot(x_ctg[:,1], size=12)})
    else:
        out = models[3].predict(x_img)
    out = np.array(out, dtype=np.float64)
    out = out.reshape(out.shape[:2]).T
    out = pd.DataFrame(out, columns=['Skin', 'Fat', 'Muscle'])
    out = out * np.array([10, 100, 100])
    out = np.abs(np.array(out) - y)
    df_skin['decimal_scaling'] = out[:, 0]
    df_fat['decimal_scaling'] = out[:, 1]
    df_muscle['decimal_scaling'] = out[:, 2]
    print('finish decimal_scaling')

    return df_skin, df_muscle, df_fat


def costum_mse(y_true, y_pred):
    # example
    # y = np.array([[i for i in range (30)]]).reshape((10,3))
    # x = np.zeros((10,3))
    # x[:,0] = y[:,0]+0.1
    # x[:,1] = y[:,1]+0.2
    # x[:,2] = y[:,2]+0.3
    # costum_mse(y, x)
    # Result: 1.4361567200000036

    weigths = tf.constant(np.array([[1.474027,  8.394525, 28.019913]], dtype=np.float32))
    
    error = tf.reduce_mean(tf.square(y_true-y_pred), axis=0)
    
    error = tf.multiply(weigths, error)
    
    error = tf.reduce_sum(error)
    
    return error/2


def costum_mae(y_true, y_pred):
    # example
    # y = np.array([[i for i in range (30)]]).reshape((10,3))
    # x = np.zeros((10,3))
    # x[:,0] = y[:,0]+0.1
    # x[:,1] = y[:,1]+0.2
    # x[:,2] = y[:,2]+0.3
    # costum_mae(y, x)
    # Result: 5.116140800000006
    
    weigths = tf.constant(np.array([[1.474027,  8.394525, 28.019913]], dtype=np.float32))
    
    error = tf.reduce_mean(tf.abs(y_true-y_pred), axis=0)
    
    error = tf.multiply(weigths, error)
    
    error = tf.reduce_sum(error)
    
    return error/2


def fit_network(model, batchsize, epoch, callback_properties,
                            x_img_train, x_ctg_train, y_train,
                            x_img_val, x_ctg_val, y_val):
    # model: neural network model
    # bacthsize: the batchsize used in the model
    # epoch : the number of epochs to train your model
    # callback_properties: it is a list that has 3 elements. The first variable is a boolean and indicates if you want to use a callback. If you do not want, do not add a second variable. Else, add a second variable that is also a boolean and indicates wether or not your output data is normalized. The third element is string that represents the file where you save the best model. E.g: name_file.h5. It s important that it finishes with '.h5'
    # x_img: images
    # x_ctg: categorical inputs
    # y: true label
    # The '_train' indicates you have to use the training data, and '_val', the validation data
    
    if callback_properties[0]: # Will you use a callback?
        if callback_properties[1]: # Are the model's outputs normalized?
            callback = tf.keras.callbacks.ModelCheckpoint(callback_properties[2], verbose=2, monitor='val_costum_mae',
                               save_best_only=True, save_weights_only=True)
        else:
            callback = tf.keras.callbacks.ModelCheckpoint(callback_properties[2], verbose=2, monitor='val_mae',
                               save_best_only=True, save_weights_only=True)
    else:
        callback = None
    
    if model.name=='categorical_network':            
        
        history = model.fit(x={'Image': x_img_train, 'Extremity': one_hot(x_ctg_train[:,0], size=4), 
                               'Position': one_hot(x_ctg_train[:,1], size=12)}, 
                            y={'Skin': y_train[:,0], 'Fat': y_train[:,1], 'Muscle': y_train[:,2]}, 
                            validation_data=({'Image': x_img_val, 'Extremity': one_hot(x_ctg_val[:,0], size=4),
                                              'Position': one_hot(x_ctg_val[:,1], size=12)}, 
                                             {'Skin': y_val[:,0], 'Fat': y_val[:,1], 'Muscle': y_val[:,2]}),
                            steps_per_epoch=len(x_img_train)//batchsize,
                            validation_steps=len(x_img_val)//batchsize,
                            epochs=epoch, callbacks=callback, verbose=2)
        
    else:
            
        history = model.fit(x=x_img_train, 
                            y={'Skin': y_train[:,0], 'Fat': y_train[:,1], 'Muscle':y_train[:,2]}, 
                            validation_data=(x_img_val, {'Skin': y_val[:,0], 'Fat': y_val[:,1], 'Muscle': y_val[:,2]}),
                            steps_per_epoch=len(x_img_train)//batchsize,
                            validation_steps=len(x_img_val)//batchsize,
                            epochs=epoch, callbacks=callback, verbose=2)
    
    return history, model
