import sklearn 
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys
import h5py
from scipy import signal

#sklearn_version = sklearn.__version__


# -*- coding: utf-8 -*-

import sys
sys.path.insert(1, './script')
import Spectrum_load as Loader 
import Spectrum_filter as Filters
import Spectrum_noise as Noise
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, BatchNormalization, SpatialDropout1D, LeakyReLU, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from sklearn.model_selection import RepeatedKFold
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import random
import warnings
import os
import tensorflow.keras
from csv import writer
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
import pandas as pd
from pandas import read_csv, concat
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from datetime import date
from tensorflow.keras import models 
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

## command parameters
GPU_ID = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID  # The GPU id to use
NOISE = [False, 'KeepDist', 0] # 3 args: True/False, UniDist/KeepDist, Nb Noised Spectra By Original Spectra 
FILTER = True
FILTER_DEPTH = 1

## Loading options
DECIMAL="."
HEADER = None # None/0
DTYPE = np.float32

## PARAMS
SCALERS_RANGE = (0.1, 0.9)
NB_EPOCHS = 1
BATCH_SIZE = 5
K_FOLD_NB = 2
K_FOLD_REPEAT = 1
SEED = 84269713
DROPOUT_RATE= 0.2
REGULARIZATION = (0, 0, 1) # 0 = BatcNorm; 1=DropOut
OPTIMIZER = 'rmsprop' 
NB_EPOCHS_ran=[]




## Random seed
def set_seed(sd):
    """ Initialize random seed in python, numpy and tensorflow """
    random.seed(sd)
    np.random.seed(sd)







def load_data():
    """ Load and format data """
    ## Load csv
    if (NOISE[0]==True):
        x = read_csv(FileName + NOISE[1] + '_Xcal.csv', header=None, sep=r'\,|\;', dtype=DTYPE, engine='python')
        xp = read_csv(FileName + 'Xval.csv', header=None, sep=r'\,|\;', dtype=DTYPE, engine='python')
        y=read_csv(FileName + NOISE[1] + '_Ycal.csv', header=None, sep=r'\,|\;', skip_blank_lines=False, dtype=DTYPE, engine='python')
        yp=read_csv(FileName + 'Yval.csv', header=None, sep=r'\,|\;', skip_blank_lines=False, dtype=DTYPE, engine='python')
        # Remove rows with NaN in xcal and ycal based on ycal
        nnaindyp=np.argwhere(~np.isnan(yp.values))
        xp=xp.iloc[nnaindyp[:,0]]
        yp=yp.iloc[nnaindyp[:,0],0]
        y_original = np.copy(y)
    else:
        x = read_csv(FileName + 'Xcal.csv', header=None, sep=r'\,|\;', dtype=DTYPE, engine='python')
        xp = read_csv(FileName + 'Xval.csv', header=None, sep=r'\,|\;', dtype=DTYPE, engine='python')
        y=read_csv(FileName + 'Ycal.csv', header=None, sep=r'\,|\;', skip_blank_lines=False, dtype=DTYPE, engine='python')
        yp=read_csv(FileName + 'Yval.csv', header=None, sep=r'\,|\;', skip_blank_lines=False, dtype=DTYPE, engine='python')
        # Remove rows with NaN in xcal and ycal based on ycal
        nnaindy=np.argwhere(~np.isnan(y.values))
        nnaindyp=np.argwhere(~np.isnan(yp.values))
        x=x.iloc[nnaindy[:,0]]
        y=y.iloc[nnaindy[:,0],0]
        xp=xp.iloc[nnaindyp[:,0]]
        yp=yp.iloc[nnaindyp[:,0],0]
        y_original = np.copy(y)

    ## Filter Data
    if FILTER == True:
        [x, xp] = Filters.apply_all_filters([x, xp], FILTER_DEPTH)
    else:
        [x, xp] = Filters.apply_solo_filter([x, xp])
    
    ## Normalise data
    [x, xp] = Loader.cpl_scale_2D([x, xp], SCALERS_RANGE)
    x_original_norm = np.copy(x)
    [y], std_scaler, min_scaler = Loader.cpl_scale_1D([y.values.reshape(-1,1)], SCALERS_RANGE)

    ## Shuffle data
    p = np.random.permutation(len(y))
    y = y[p]
    x = x[p]

    return x, xp, y, yp, std_scaler, min_scaler, x_original_norm, y_original
    
    
    

    
    
def create_model(data):
    """ Create previously hyperparametrized convolutional model """
    feature_count = data.shape[1]
    dimension = data.shape[2]
    
    # NIRS HYPERIZED GLOBAL
    model = Sequential()
    model.add(SpatialDropout1D(0.08, input_shape=(feature_count, dimension), seed=SEED))
    model.add(Conv1D (filters=4, kernel_size=21, strides=5, activation='selu'))
    if (REGULARIZATION[0]==0):
        model.add(BatchNormalization())
    else:
        model.add(SpatialDropout1D(DROPOUT_RATE))
    model.add(Conv1D (filters=64, kernel_size=16, strides=3, activation='relu'))
    if (REGULARIZATION[1]==0):
        model.add(BatchNormalization())
    else:
        model.add(SpatialDropout1D(DROPOUT_RATE))
    # model.add(Bidirectional(LSTM(500)))
    model.add(Conv1D (filters=32, kernel_size=5, strides=3, activation='elu'))
    if (REGULARIZATION[2]==0):
        model.add(BatchNormalization())
    else:
        model.add(SpatialDropout1D(DROPOUT_RATE))
    model.add(Flatten())
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    return model
    




#Create New Model

def create_new_model(data):
    """ We previously created  hyperparametrized convolutional model """
    feature_count = data.shape[1]
    dimension = data.shape[2]
    model_0 = load_model('./out/BACON_LUCAS_SOC_Organic_3528_Nocita_2022-03-16_10:53:40_filt1_fold2_model.hdf5')
    #"/home/phenomen/Workspace/TRANSPIR/BACON/out/BACON_LUCAS_SOC_Organic_3528_Nocita_2022-03-16_10:53:40_filt1_fold2_model.hdf5"
    new_model = Sequential()
    #new_model.add(SpatialDropout1D(0.08, input_shape=(feature_count, dimension), seed=SEED))
    #new_model.add(Flatten())
    #new_model.add(Dense(4200*13, activation='selu'))
    #new_model.add(tf.keras.layers.Reshape((4200, 13), input_shape=(4200*13,)))
    for layer in model_0.layers[:]:
        new_model.add(layer)
    return new_model



# Transferer les dernières couches du modèle pré-entrainé

def create_new_model_2(data):
    """ We previously created  hyperparametrized convolutional model """
    feature_count = data.shape[1]
    dimension = data.shape[2]
    model_0 = load_model('./out/BACON_LUCAS_SOC_Cropland_6111_Nocita_filt1_fold2_model.hdf5')
    
    new_model = Sequential()
    new_model.add(SpatialDropout1D(0.08, input_shape=(feature_count, dimension), seed=SEED))
    new_model.add(Conv1D (filters=4, kernel_size=21, strides=5, activation='selu'))
    if (REGULARIZATION[0]==0):
        new_model.add(BatchNormalization())
    else:
        new_model.add(SpatialDropout1D(DROPOUT_RATE))
    new_model.add(Conv1D (filters=64, kernel_size=16, strides=3, activation='relu'))
    if (REGULARIZATION[1]==0):
        new_model.add(BatchNormalization())
    else:
        new_model.add(SpatialDropout1D(DROPOUT_RATE))
    new_model.add(Conv1D (filters=32, kernel_size=5, strides=3, activation='elu'))
    if (REGULARIZATION[2]==0):
        new_model.add(BatchNormalization())
    else:
        new_model.add(SpatialDropout1D(DROPOUT_RATE)) 
    for layer in model_0.layers[6:]:
        new_model.add(layer)
    return new_model
    
    
    
    
    
# Transferer le modèle pré-entrainé sauf les premières couches
    
def create_new_model_3(data):
    """ We previously created  hyperparametrized convolutional model """
    feature_count = data.shape[1]
    dimension = data.shape[2]
    model_0 = load_model('./out/BACON_LUCAS_SOC_Organic_3528_Nocita_2022-03-16_10:53:40_filt1_fold2_model.hdf5')

    new_model = Sequential()
    new_model.add(SpatialDropout1D(0.08, input_shape=(feature_count, dimension), seed=SEED))
    new_model.add(Conv1D (filters=4, kernel_size=21, strides=5, activation='selu')) 
    for layer in model_0.layers[2:]:
        new_model.add(layer)
    return new_model    
    


def create_new_model_4(data):
    """ We previously created  hyperparametrized convolutional model """
    feature_count = data.shape[1]
    dimension = data.shape[2]
    model_0 = load_model('./out/BACON_LUCAS_SOC_Organic_3528_Nocita_filt1_fold0_model.hdf5')

    new_model = Sequential()
    new_model.add(SpatialDropout1D(0.2, input_shape=(feature_count, dimension), seed=SEED))
    new_model.add(Conv1D (filters=4, kernel_size=21, strides=5, activation='selu')) 
    for layer in model_0.layers[2:len(model_0.layers)-2]:
        new_model.add(layer)
    new_model.add(Flatten())
    new_model.add(Dense(16, activation='sigmoid'))
    new_model.add(Dense(1, activation='sigmoid'))
    return new_model  
    




def huber_loss(y_pred, y, delta=1.0):
    
    huber_mse = 0.5*(y-y_pred)**2
    huber_mae = delta * (np.abs(y - y_pred) - 0.5 * delta)
    return np.mean(np.where(np.abs(y - y_pred) <= delta, huber_mse, huber_mae))
    
    

def huber(y_true, y_pred, delta=1.0):

    return np.mean(delta**2*( (1+((y_true-y_pred)/delta)**2)**0.5 -1))
    
    



def run_cv():
    """ Run the training and return prediction """
    ## Get data
    x, xp, y, yp, _, min_scaler, x_original_norm, y_original = load_data()
    
    #x = signal.resample(x, 4200, axis = 1)
    #xp = signal.resample(xp, 4200, axis = 1)
    #x_original_norm = signal.resample(x_original_norm, 4200, axis = 1)
    
    #print(xp.shape)
    #x = tf.keras.layers.UpSampling1D(size=4)(x)
    #xp = tf.keras.layers.UpSampling1D(size=4)(xp)
    #x_original_norm = tf.keras.layers.UpSampling1D(size=4)(x_original_norm)
    
    #print('--------------------------------------')
    #print(x_original_norm.shape)
    ## Run KFold training
    kfold = RepeatedKFold(n_splits=K_FOLD_NB, n_repeats=K_FOLD_REPEAT, random_state=SEED)
    fold = 0

    for train, test in kfold.split(y):
        # config fold
        prefix = './out/BACON_' + adir + '_filt' + str(FILTER_DEPTH)
        #model_filepath = prefix + '_fold' + str(fold) + '_model.hdf5'
        prediction_filepath = prefix + '_fold' + str(fold) + '_prediction.csv'
        global_prediction_filepath = prefix + '_fold' + str(fold) + '_global_prediction'

        ## Config callbacks (adaptative learning rate, early stopping, saving) and optimizer
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, verbose=0, min_delta =0.5e-5, 
                                           mode='min')
        earlyStopping = EarlyStopping(monitor='val_loss', patience=500, verbose=1, mode='min') # TODO test restore_best_weights=True (Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.)
        #mcp_save = ModelCheckpoint(model_filepath, save_best_only=True, monitor='val_loss', mode='min') 

        # fit model
        
        model_filepath = "./out/BACON_LUCAS_SOC_Cropland_6111_Nocita_filt1_fold2_model.hdf5"
        #model_0 = load_model('./out/BACON_LUCAS_SOC_Cropland_6111_Nocita_filt1_fold2_model.hdf5')
        #model = create_model(x)
        model = create_new_model(x)
        model.compile(loss='mean_squared_error', metrics=['mae','mse'], optimizer=OPTIMIZER)
        history = model.fit(x[train], y[train], 
                    epochs=NB_EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    validation_data = (x[test], y[test]),
                    verbose=1, 
                    callbacks=[reduce_lr_loss, earlyStopping])

        # load best and evaluate RMSE
        print("Fold n°" + str(fold) + " done.")
        model.load_weights(model_filepath)

        predict_test = model.predict(x[test])
        predict_test = min_scaler.inverse_transform(predict_test)
        y_test = min_scaler.inverse_transform(y[test])
        RMSE = np.sqrt(np.mean((y_test-predict_test)**2))
        print("*** TEST SET RMSE *** > %.3f" % (RMSE))

        y_predict = model.predict(x_original_norm)
        y_predict = min_scaler.inverse_transform(y_predict)
        RMSE = np.sqrt(np.mean((y_original-y_predict)**2))
        np.savetxt(global_prediction_filepath + ".csv", np.c_[y_original, y_predict], delimiter=";")
        print("*** GLOBAL RMSE *** > %.3f" % (RMSE))

        # predict unknown data
        predict = model.predict(xp)
        predict = min_scaler.inverse_transform(predict)
        np.savetxt(prediction_filepath, predict, delimiter=";")
        
        NB_EPOCHS_ran.append(len(history.history['loss']))
        fold = fold + 1



path='./data/'
dirs=os.listdir(path)
for adir in sorted(dirs): 
    FileName = path + adir + "/"
    print(FileName)
print("****************************************************************************")
#print(Filename)

x, xp, y, yp, _, min_scaler, x_original_norm, y_original = load_data()


print('The shape of x is:')
print(x.shape)  






model = create_new_model_2(x)

#layer1 = model.get_layer('spatial_dropout1d_2') # pas entrainable
layer2 = model.layers[2] #model.get_layer('conv1d_3') # entrainable
print(layer2)

# Freeze the second layer
layer2.trainable = False

# Keep a copy of the weights of layer2 for later reference
initial_layer2_weights_values = layer2.get_weights()

print(model.summary())







model = load_model('./out/BACON_LUCAS_SOC_Cropland_6111_Nocita_filt1_fold2_model.hdf5')
model.compile(loss='mean_squared_error', metrics=['mae','mse'], optimizer=OPTIMIZER)
kfold = RepeatedKFold(n_splits=K_FOLD_NB, n_repeats=K_FOLD_REPEAT, random_state=SEED)
fold = 0

for train, test in kfold.split(y):
    
    path ='./data/'
    dirs = os.listdir(path)
    FileName = path + adir + "/"
    prefix = './out/BACON_' + adir + '_filt' + str(FILTER_DEPTH)
    model_filepath = prefix + '_fold' + str(fold) + '_model.hdf5'
    prediction_filepath = prefix + '_fold' + str(fold) + '_prediction.csv'
    global_prediction_filepath = prefix + '_fold' + str(fold) + '_global_prediction'
    print(prefix)
    ## Config callbacks (adaptative learning rate, early stopping, saving) and optimizer
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, verbose=1, min_delta =0.5e-5, 
                                           mode='min')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='min') # TODO test restore_best_weights=True (Whether to restore         model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are        used.)
    mcp_save = ModelCheckpoint(model_filepath, save_best_only=True, monitor='val_loss', mode='min')
    
    
    history = model.fit(x[train], y[train], 
                    epochs=NB_EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    validation_data = (x[test], y[test]),
                    verbose=1,  
                    callbacks=[reduce_lr_loss, earlyStopping, mcp_save] )



    # load best and evaluate RMSE
    model_filepath2 = "./out/BACON_LUCAS_SOC_Organic_3528_Nocita_2022-03-16_10:53:40_filt1_fold2_model.hdf5"
    print("Fold n°" + str(fold) + " done.")
    model.load_weights(model_filepath2)
    
    predict_test = model.predict(x[test])
    predict_test = min_scaler.inverse_transform(predict_test)
    y_test = min_scaler.inverse_transform(y[test])
    RMSE = np.sqrt(np.mean((y_test-predict_test)**2))
    print("*** TEST SET RMSE *** > %.3f" % (RMSE))

    y_predict = model.predict(x_original_norm)
    y_predict = min_scaler.inverse_transform(y_predict)
    RMSE = np.sqrt(np.mean((y_original-y_predict)**2))
    np.savetxt(global_prediction_filepath + ".csv", np.c_[y_original, y_predict], delimiter=";")
    print("*** GLOBAL RMSE *** > %.3f" % (RMSE))

    # predict unknown data
    predict = model.predict(xp)
    predict = min_scaler.inverse_transform(predict)
    np.savetxt(prediction_filepath, predict, delimiter=";")
        
    NB_EPOCHS_ran.append(len(history.history['loss']))
    fold = fold + 1


   
    
    # Check predictions for calibration 
    ResPredKfoldCal=[]
    for i in range(K_FOLD_NB):
        y_scores_i=read_csv('./out/BACON_' + adir + '_filt' + str(FILTER_DEPTH) + '_fold' + str(i) + '_global_prediction.csv', 
                            sep=r'\,|\;', header=None, dtype=DTYPE, engine='python')
        ResPredKfoldCal.append(y_scores_i) 
        
    df_concat = concat(ResPredKfoldCal)
    by_row_index = df_concat.groupby(df_concat.index)
    ycal = by_row_index.mean()
    
    rmse_cal=round(np.sqrt(np.mean((ycal[0]-ycal[1])**2)),3)
    r2_cal = round(r2_score(ycal[0], ycal[1]),3)
    
    # Check predictions for validation 
    yval_obs = read_csv(FileName + 'Yval.csv', header=HEADER, sep=r'\,|\;', dtype=DTYPE, engine='python')
    ResPredKfoldVal=[]
    for i in range(K_FOLD_NB):
        y_scores_i=read_csv('./out/BACON_' + adir + '_filt' + str(FILTER_DEPTH) + '_fold' + str(i) + '_prediction.csv', 
                            sep=r'\,|\;', header=None, dtype=DTYPE, engine='python')
        ResPredKfoldVal.append(y_scores_i) 
    
    df_concat = concat(ResPredKfoldVal)
    by_row_index = df_concat.groupby(df_concat.index)
    yval_pred = by_row_index.mean()
    resid_val=yval_pred.values-yval_obs.values
    rmse_val=round(np.sqrt(np.mean(resid_val**2)),3)
    r2_val = round(r2_score(yval_obs, yval_pred),3)
    resid_cal=ycal[1]-ycal[0]


    huber_loss_cal = round(huber_loss(ycal[1], ycal[0], delta=1.0),3)
    huber_loss_val = round(huber_loss(yval_obs.values, yval_pred.values, delta=1.0),3)
    
    huber_cal = round(huber(ycal[1], ycal[0], delta=1.0),3)
    huber_val = round(huber(yval_obs.values, yval_pred.values, delta=1.0),3)
    
    print ("##################################################################")

    print ("RMSE  Calibration/Validation\t%0.2F\t%0.2F"%(rmse_cal, rmse_val))
    print ("r2 Calibration/Validation\t%0.4F\t%0.4F"%(r2_cal, r2_val))
    print ("Huber Calibration/Validation\t%0.4F\t%0.4F"%(huber_cal, huber_val))
    










#####################################################_TRANSFER_2 ################################################################################





print('############# SECTION_2 ################')


for layer in model.layers:
    lay = layer.name
    print('The layer', lay, 'has', len(layer.weights), 'weights in total and', len(layer.trainable_weights), 'are trainable')

print('############# apres avoir mis trainable = False ################')

for layer in model.layers:
    laye = layer.name
    layer.trainable = False
    print('The layer', laye, 'has', len(layer.weights), 'weights in total and', len(layer.trainable_weights), 'are trainable')

