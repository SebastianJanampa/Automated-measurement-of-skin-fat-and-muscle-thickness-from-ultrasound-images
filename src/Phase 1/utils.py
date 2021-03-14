##################
#######INFO#######
##################
"""
FILE: utils.py
MAIN FILE: 
PYTHON: 3.8.3
AUTHOR: Sebastian Janampa Rojas
EMAIL: sebastian.janampa@utec.edu.pe
CREATE DATE:
"""


################# 
####LIBRARIES####
#################
import numpy as np
np.random.seed(1)# For reproducibility
import pandas as pd
import glob
from medpy.io import load
import os
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, regularizers, Input
tf.random.set_seed(1234)# For reproducibility
from tqdm import tqdm
import matplotlib.pyplot as plt


##########################################
##############IMPORT DATASET##############
##########################################
def import_data(path, subjects):
    """
    This function loads the csv files and images. 
    Moreover, it removes data that do not have a value for muscles.
    
    VARIABLES
        - path:path where the images are located
        - subjects: name of the subjects to import. Type: List
    """
    
    x_img = [] #Original image
    x_ctg = pd.DataFrame() #Categorical data
    y = pd.DataFrame() #csv files
    
    folder = "Ultrasound_minFrame"
    
    tot_use, tot_no_use = 0, 0

    for subject in tqdm(subjects):
#         Load csv files
        values = pd.read_csv(os.path.join(path, subject, 'thickness.csv')) # Thickness
        ctg_data = pd.read_csv(os.path.join(path, subject, 'categorical_values.csv')) # Categorical data (extremity and position)
        
#         Remove NAN data
        use = values['Muscle'].apply(pd.notna) 
        no_use=values['Muscle'].apply(np.isnan)
        values.dropna(subset = ["Muscle"], inplace=True)
        x_ctg = x_ctg.append(ctg_data[use], ignore_index=True)
        y = y.append(values, ignore_index=True)        
        
#         Load images
        images_name = ctg_data['Name'][use]
        directory = os.path.join(path, subject, folder)

        for name in images_name:
            img_dir = glob.glob(os.path.join(directory, name)+'*.IMA')[0]
            image,_ = load(img_dir)
            image = image[201:801, 31:745, 0, 0].T/255
            image = cv2.resize(image, (256, 304))
            x_img.append(image)
        
        tot_use += use.sum()
        tot_no_use +=  no_use.sum()
    print('# of images used: %i' %tot_use)
    print('# of images unused: %i' %tot_no_use)
    
    x_img = np.array(x_img, dtype=np.float32)
    x_img = x_img.reshape(np.append(x_img.shape, 1))# We add a fourth dimension which represents the number of channels
    
#    Converting categorical data to numerical data
    cleanup = {
        'Extremity': {'LL': 1, 'UL': 2, 'LA': 3, 'UA': 4},
        'Position': {'AD': 1, 'AC': 2, 'AP': 3, 'LD': 4, 'LP': 5, 'LC': 6, 'MC': 7, 'MD': 8, 'MP': 9, 'PC': 10, 'PD': 11, 'PP': 12}
    }
    x_ctg.replace(cleanup, inplace=True)
    
    return [x_img, x_ctg.drop(columns=['Name']), y.drop(columns=['Name'])]

def load_data(data_dir, **kwargs):
    """
    This function load different datasets that are specified in kwargs.
    It uses the function import_data().
    
    VARIABLES
        - data_dir: directory of the data. Type: str
        - kwargs: datasets. Type: dict
    
    EXAMPLE
        cur_dir = os.getcwd() # Current directory
        main_dir = '\\'.join(cur_dir.split('\\')[:-2]) # Main directory
        data_dir = os.path.join(main_dir,'NN\\data') # Data directory
        datasets = {'training': subjects_train, 'validation': subjects_val, 'test':subjects_test}
        training, validation, testing = utils.load_data(data_dir, **datasets)
    """
    datasets = []
    for key, values in kwargs.items():
        print('----------IMPORTING %s SET----------'%key.upper())
        datasets.append(import_data(data_dir, values))
    return datasets


##########################################
##############PRE-PROCESSING##############
##########################################
def normalization_tech(dataset, methods = None):
    """
    Normalization of the output data using:
    - z-score normalization
    - linear scaling
    - decimal scaling
    """
    if methods == None:
        raise Exception('Error. Insert a normalization technique name.')
    else:
        train, validation, testing = dataset
        std, mean, mini, maxi = train.std(), train.mean(), train.min(), train.max()
        dec_vals = pd.Series(np.array([10 ,100 ,100]), index=['Skin', 'Fat', 'Muscle'])
        non = pd.Series(np.array([1 ,1 ,1]), index=['Skin', 'Fat', 'Muscle'])
        params = {'std': std, 'mean': mean, 
                  'mini': mini, 'maxi': maxi, 
                  'dec_vals': dec_vals, 
                  'non': non}
        norm_datasets = [] # normalized datasets
        for method in methods:
            if method == 'std':
                std_train = (train - mean)/std
                std_val = (validation - mean)/std
                std_test = (testing - mean)/std
                norm_datasets.append([std_train, std_val, std_test])
            elif method == 'lin':
                lin_train = (train - mini)/(maxi-mini)
                lin_val = (validation - mini)/(maxi-mini)
                lin_test = (testing - mini)/(maxi-mini)
                norm_datasets.append([lin_train, lin_val, lin_test])
            elif method == 'dec':
                dec_train = train/dec_vals
                dec_val = validation/dec_vals
                dec_test = testing/dec_vals
                norm_datasets.append([dec_train, dec_val, dec_test])
            else:
                raise Exception("Technique '%s' is unknown.\nTechnique names available: 'std', 'lin' and 'dec'" %method)
        norm_datasets.append([train, validation, testing])
        methods.append('non')
        if len(norm_datasets) == 1:
            return norm_datasets[0], params
        else:
            dic_norm_datasets = []
            for i in range(len(dataset)):
                dic_norm_dataset={}
                j=0
                for method in methods:
                    dic_norm_dataset[method] = norm_datasets[j][i]
                    j += 1
                dic_norm_datasets.append(dic_norm_dataset)
            return dic_norm_datasets, params
        
def func_remove_outliers(dataset, params):
    z_score = dataset['thickness']['std']
    index = ((abs(z_score)<3).apply(np.sum,axis=1))==3
    # Removing outliers
    dataset['images'] = dataset['images'][index]
    dataset['categories'] = dataset['categories'][index]
    for method, output in dataset['thickness'].items():
        dataset['thickness'][method] = output[index]
    return dataset

def remove_outliers(datasets, params):
    new_datasets = []
    print( '##############################\n###Before removing outliers###\n##############################')
    print('Training: %i samples'%datasets[0]['images'].shape[0])
    print('Validation: %i samples'%datasets[1]['images'].shape[0])
    print('Testing: %i samples'%datasets[2]['images'].shape[0])
    for dataset in datasets:
        dataset = func_remove_outliers(dataset, params)
        new_datasets.append(dataset)
    print( '#############################\n###After removing outliers###\n#############################')
    print('Training: %i samples'%datasets[0]['images'].shape[0])
    print('Validation: %i samples'%datasets[1]['images'].shape[0])
    print('Testing: %i samples'%datasets[2]['images'].shape[0])
    return new_datasets


##########################################
##############DEEP LEARNING###############
##########################################
"""
This part is used for the following sections of 
    - Data Normalization
    - Costum Loss
    - Categorical Data
"""
# Basic convolutional block (CONV-MAXPOOL-RELU)
class Block2D(layers.Layer):
    def __init__(self, out_channels, name=None,kernel_size=3):
        if name is not None:
            name = 'Block2D_'+str(name)
        super().__init__(name=name)
        self.conv = layers.Conv2D(out_channels, kernel_size, padding="same", kernel_initializer='he_normal')
        self.max = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        x = self.max(x)
        x = tf.nn.relu(x)
        return x

# Network A and B
def ModelAB(input_shape, name):
    # For reproducibility
    tf.random.set_seed(1234)
    # Inputs
    input_img = Input(shape=input_shape, name='Image')
    x = input_img
    # Convolutional layers
    layers_channels = [32, 64, 128, 256, 256]
    idx_blocks = [i+1 for i in range(len(layers_channels))]
    for out_channels, idx in list(zip(layers_channels, idx_blocks)):
        x = Block2D(out_channels, name=idx)(x) 
    # Fully-connected layers
    x = layers.Flatten(name='Flatten')(x)
    x = layers.Dense(512, activation='relu', kernel_initializer='he_normal', name='Dense1')(x)
    x = layers.Dropout(0.5, name='Drop1')(x)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal', name='Dense2')(x)
    x = layers.Dropout(0.5, name='Drop2')(x)    
    # Tissues thickness (outputs)
    skin = layers.Dense(1, name='Skin')(x)
    fat = layers.Dense(1, name='Fat')(x)
    muscle = layers.Dense(1, name='Muscle')(x)
    # Create model
    model = tf.keras.Model(inputs=input_img, 
                           outputs=[skin,fat,muscle], 
                           name='sim-'+name)
    return model

# Create models
def create_models(versions, input_shape, dic_weights=None, methods=None, lr=1e-4):
    # Parameters
    optimizer = keras.optimizers.Adam(lr=lr)
    if dic_weights is None:
        dic_weights = {method: None for method in methods}
    # Pre-process
    """"This part is to make two list with the same size"""
    if type(versions) is not list:
        if (type(methods) is str) or  (methods is None):
            versions = [versions]
            methods = [methods]
        else:
            versions = [versions for _ in range(len(methods))]
    else:
        if (type(names) is str) or  (names is None):
            methods = [methods for _ in range(len(versions))]
        else:
            if len(methods)!=len(versions): 
                if (len(versions)>1) and (len(methods)>1):
                    raise Exception("Error. Size of 'versions' (%i) is different to 'methods' (%i)." %(len(versions), len(methods)))
                elif (len(versions)>1) and (len(methods)==1):
                    names = [methods[0] for _ in range(len(versions))]
                else:
                    versions = [versions[0] for _ in range(len(methods))]
    # Function
    models = []
    print('###############################\n########Creating Models########\n###############################')
    for version, method in list(zip(versions, methods)):
        tf.random.set_seed(1234)
        model = version(input_shape=input_shape ,name=method)
        name = model.name.split('-')
        weights = dic_weights[name[1]]
        model.compile(optimizer=optimizer,
                      loss='mse', metrics='mae',
                      loss_weights=weights)
        models.append(model)
        print('%s model was created'%model.name)
        del model
    print('\n')
    return models
    
# Pre-processing
def one_hot(x, size):
    """ 
    One hot enconding
    
    VARIABLES
        x: an array with integers. Type: np.array
        size: the depth of the one hot encoding. It has to be greater or equal to np.amax(x). Type: int
    
    EXAMPLE
        one_hot(np.array([2,3]),3)
    
    RESULT
        array([[0., 1., 0.],
           [0., 0., 1.]])
    """
    shape = (x.size, size)
    one_hot = np.zeros(shape)
    rows = np.arange(x.size)
    one_hot[rows, x-1] = 1
    return one_hot

# Training
def myFit(models, batchsize=64, num_epochs=100, verbose=0, **kwargs):
    # Unpacking datasets
    training = kwargs['training']
    if 'validation' in kwargs.keys():
        validation = kwargs['validation']
    else:
        validation = None
    # Create callback
    class myCallback(keras.callbacks.Callback):
        """Display the metrics of the model every 10 epochs"""
        def __init__(self, num_epochs):
            self.epochs = num_epochs
            
        def on_epoch_end(self, epoch, logs=None):
            if (epoch%2==1) or (epoch==0):
                print("    Epoch %04i/%04i:"%(epoch+1, self.epochs), end=' ')
                for key, value in logs.items():
                    print('%s: %.3f'%(key, value), end=' - ')
                print()
    # Traning the models
    print('################################\n#########Start Training#########\n################################\n')
    results = {}
    for model in models:
        name = model.name.split('-')
        print("#############\n%s model\n#############"%name[1])
        # Defining x and y (input and outputs)
        y_train = training['thickness'][name[1]]
        y_train = {tissue: y_train[tissue] for tissue in y_train.keys()}
        if validation is not None:
            y_val = validation['thickness'][name[1]]
            y_val = {tissue: y_val[tissue] for tissue in y_val.keys()}
        if name[0] == 'sim':
            x_train = training['images']
            if validation is not None:
                x_val = validation['images']
                val_data = (x_val, y_val)
                del x_val, y_val
            else:
                val_data = None
        else:
            x_train = {'Image': training['images'], 
                       'Extremity': one_hot(training['categories']['Extremity'], size=4),
                       'Position': one_hot(training['categories']['Position'], size=12)}
            if validation is not None:
                x_val = {'Image': validation['images'], 
                         'Extremity': one_hot(validation['categories']['Extremity'], size=4),
                         'Position': one_hot(validation['categories']['Position'], size=12)}
                val_data = (x_val, y_val)
                del x_val, y_val
            else:
                val_data = None
        # Using 'fit' method
        tf.random.set_seed(1234) # For reproducibility (do not erase)
        history = model.fit(x=x_train,
                            y=y_train,
                            validation_data=val_data,
                            epochs=num_epochs,
                            batch_size=batchsize,
                            verbose=verbose
                            shuffle=True,
                            callbacks = myCallback(num_epochs))
        # Save the model and the history
        results[name[1]]={'model': model, 'history': history}
        del model, history, x_train, y_train, val_data
    print('\n')
    return results


##########################################
#################RESULTS##################
##########################################
# Results
def show_results(results, parameters, **kwargs):
    print('#################################\n#########Showing Results#########\n#################################\n')
    # Parameters
    dic_weights = {
    'non': parameters['non'],
    'std': parameters['std'],
    'lin': parameters['maxi'] - parameters['mini'],
    'dec': parameters['dec_vals']
    }
    ioe_mae = {'Skin': 0.36, 'Fat':0.78, 'Muscle':0.65}
    ioe_std = {'Skin': 0.38, 'Fat':1.10, 'Muscle':1.11}
    # Getting results
    for dataset_name in kwargs.keys():
        print('#############\n%s\n#############'%dataset_name.upper())
        dataset = kwargs[dataset_name]
        for method in results.keys():
            model = results[method]['model']
            print(model.name.upper())
            if model.name.split('-')[0] == 'sim':
                x = dataset['images']
            else:
                x = {'Image': training['images'], 
                     'Extremity': one_hot(training['categories']['Extremity'], size=4),
                     'Position': one_hot(training['categories']['Position'], size=12)}
            y_pred = np.array(model.predict(x))
            y_pred = y_pred.reshape(y_pred.shape[:2]).T
            mae = (y_pred - dataset['thickness'][method]).abs()*dic_weights[method]
            for tissue in mae.keys():
                print('MAE of %s: %.3f +- %.2f'%(tissue, mae[tissue].mean(), mae[tissue].std()))
            print('-'*40)
        # Print the Inter-observer error
        print('IOE')
        for tissue in ioe_mae.keys():
            print('MAE of IOE: %.3f +- %.2f'%(ioe_mae[tissue], ioe_std[tissue]))
        print('-'*40)
        
# Plot figures
def plot_modelsv1(results, parameters, filename):
    # Parameters
    dic_weights = {
    'non': parameters['non'],
    'std': parameters['std'],
    'lin': parameters['maxi'] - parameters['mini'],
    'dec': parameters['dec_vals']
    }
    colors = {
        'non': '#A6EB73',
        'std': '#F5CB64',
        'lin': '#DB5546',
        'dec': '#BA82F5'
    }
    # Figure
    sufix = '_mae'
    tissues = ['Skin', 'Fat', 'Muscle']
    fig, axs = plt.subplots(3, 2)
    fig.tight_layout(pad=0.5)
    fig.set_figheight(14)
    fig.set_figwidth(12)
    for method in results.keys():
        histories = results[method]['history'].history
        color = colors[method]
        for col, prefix in enumerate(['', 'val_']):
            for row, tissue in enumerate(tissues):
                interest = prefix + tissue + sufix
                hist = histories[interest] * np.array(dic_weights[method][tissue])
                epochs = [i+1 for i in range(len(hist))]
                axs[row, col].plot(epochs, hist, label=method, linewidth=1.5, color=color, linestyle='-')
                if prefix == 'val_':
                    title_name = 'Validation - ' + tissue
                else:
                    title_name = 'Training - ' + tissue
                axs[row,col].set_title(title_name, fontsize=15, fontweight='bold')
                axs[row,col].set_xlabel('Epochs', fontsize=12)
                axs[row,col].set_ylabel('Error', fontsize=12)
                axs[row,col].tick_params(axis='x', labelsize=12) 
                axs[row,col].tick_params(axis='y', labelsize=12) 
                axs[row,col].grid(True)
                axs[row,col].legend(fontsize=10, borderpad=1)
                axs[row,col].set_xlim(1, len(epochs))
    plt.savefig('images\\' + filename)
    
def plot_modelsv2(results, parameters, filename):
    # Parameters
    dic_weights = {
    'non': parameters['non'],
    'std': parameters['std'],
    'lin': parameters['maxi'] - parameters['mini'],
    'dec': parameters['dec_vals']
    }
    colors = {
        'non': '#A6EB73',
        'std': '#F5CB64',
        'lin': '#DB5546',
        'dec': '#BA82F5'
    }
    # Figure
    sufix = '_mae'
    tissues = ['Skin', 'Fat', 'Muscle']
    fig, axs = plt.subplots(3, 1)
    fig.tight_layout(pad=0.5)
    fig.set_figheight(14)
    fig.set_figwidth(6)
    for method in results.keys():
        histories = results[method]['history'].history
        color = colors[method]
        for col, prefix in enumerate(['', 'val_']):
            if prefix == 'val_':
                linsty = '--'
                label = None
            else:
                linsty = '-'
                label = method
            for row, tissue in enumerate(tissues):
                interest = prefix + tissue + sufix
                hist = histories[interest] * np.array(dic_weights[method][tissue])
                epochs = [i+1 for i in range(len(hist))]
                axs[row].plot(epochs, hist, label=label, linewidth=1.5, color=color, linestyle=linsty)
                axs[row].set_title(tissue, fontsize=15, fontweight='bold')
                axs[row].set_xlabel('Epochs', fontsize=12)
                axs[row].set_ylabel('Error', fontsize=12)
                axs[row].tick_params(axis='x', labelsize=12) 
                axs[row].tick_params(axis='y', labelsize=12) 
                axs[row].grid(True)
                axs[row].legend(fontsize=10, borderpad=1)
                axs[row].set_xlim(1, len(epochs))
    plt.savefig('images\\' + filename)