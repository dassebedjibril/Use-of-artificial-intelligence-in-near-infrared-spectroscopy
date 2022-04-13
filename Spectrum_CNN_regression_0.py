# -*- coding: utf-8 -*-
"""
"""
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
import random
import warnings
import os
import tensorflow.keras
from csv import writer
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
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
NOISE = [True, 'KeepDist', 0] # 3 args: True/False, UniDist/KeepDist, Nb Noised Spectra By Original Spectra 
FILTER = True
FILTER_DEPTH = 1

## Loading options
DECIMAL="."
HEADER = 0 # None/0
DTYPE = np.float32

## PARAMS
SCALERS_RANGE = (0.1, 0.9)
NB_EPOCHS = 50
BATCH_SIZE = 4
K_FOLD_NB = 3
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
        x = read_csv(FileName + NOISE[1] + '_Xcal.csv', header=HEADER, sep=r'\,|\;', dtype=DTYPE, engine='python')
        xp = read_csv(FileName + 'Xval.csv', header=HEADER, sep=r'\,|\;', dtype=DTYPE, engine='python')
        y=read_csv(FileName + NOISE[1] + '_Ycal.csv', header=HEADER, sep=r'\,|\;', skip_blank_lines=False, dtype=DTYPE, engine='python')
        yp=read_csv(FileName + 'Yval.csv', header=HEADER, sep=r'\,|\;', skip_blank_lines=False, dtype=DTYPE, engine='python')
        # Remove rows with NaN in xcal and ycal based on ycal
        nnaindyp=np.argwhere(~np.isnan(yp.values))
        xp=xp.iloc[nnaindyp[:,0]]
        yp=yp.iloc[nnaindyp[:,0],0]
        y_original = np.copy(y)
    else:
        x = read_csv(FileName + 'Xcal.csv', header=HEADER, sep=r'\,|\;', dtype=DTYPE, engine='python')
        xp = read_csv(FileName + 'Xval.csv', header=HEADER, sep=r'\,|\;', dtype=DTYPE, engine='python')
        y=read_csv(FileName + 'Ycal.csv', header=HEADER, sep=r'\,|\;', skip_blank_lines=False, dtype=DTYPE, engine='python')
        yp=read_csv(FileName + 'Yval.csv', header=HEADER, sep=r'\,|\;', skip_blank_lines=False, dtype=DTYPE, engine='python')
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
    [y], std_scaler, min_scaler = Loader.cpl_scale_1D([y], SCALERS_RANGE)

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
    
    
    
    
    




def run_cv():
    """ Run the training and return prediction """
    ## Get data
    x, xp, y, yp, _, min_scaler, x_original_norm, y_original = load_data()

    ## Run KFold training
    kfold = RepeatedKFold(n_splits=K_FOLD_NB, n_repeats=K_FOLD_REPEAT, random_state=SEED)
    fold = 0

    for train, test in kfold.split(y):
        # config fold
        prefix = './out/BACON_' + adir + '_filt' + str(FILTER_DEPTH)
        model_filepath = prefix + '_fold' + str(fold) + '_model.hdf5'
        prediction_filepath = prefix + '_fold' + str(fold) + '_prediction.csv'
        global_prediction_filepath = prefix + '_fold' + str(fold) + '_global_prediction'

        ## Config callbacks (adaptative learning rate, early stopping, saving) and optimizer
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=200, verbose=1, min_delta =0.5e-5, 
                                           mode='min')
        earlyStopping = EarlyStopping(monitor='val_loss', patience=500, verbose=0, mode='min') # TODO test restore_best_weights=True (Whether to restore model weights from the epoch with the best value of the monitored quantity. If False, the model weights obtained at the last step of training are used.)
        mcp_save = ModelCheckpoint(model_filepath, save_best_only=True, monitor='val_loss', mode='min') 

        # fit model
        model = create_model(x)
        model.compile(loss='mean_squared_error', metrics=['mae','mse'], optimizer=OPTIMIZER)
        history = model.fit(x[train], y[train], 
                    epochs=NB_EPOCHS, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True, 
                    validation_data = (x[test], y[test]),
                    verbose=1, 
                    callbacks=[reduce_lr_loss, earlyStopping, mcp_save])

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

    if (NOISE[0]==True):
        Noise.DataAugment(FileName, NOISE[1], NOISE[2], HEADER)  
    set_seed(SEED)
    run_cv()
    
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


   # Get graph
    yval_obsn=np.min(yval_obs)[0]
    yval_obsx=np.max(yval_obs)[0]
    plt.rcParams["figure.figsize"] = (20,30)
    plt.subplot(2, 2, 1)
    plt.suptitle(adir + ", Filter" + str(FILTER) + str(FILTER_DEPTH) + ", Noise" + str(NOISE))
    plt.scatter(ycal[0], ycal[1], c = 'g', alpha=0.1)
    plt.text(np.min(ycal[0])+0.05*(np.max(ycal[0])-np.min(ycal[0])), 
             np.max(ycal[1]), "RMSE = " + str(rmse_cal) + "\n" + "R² = " + str(r2_cal))
    plt.title('Cross-validation', size=16, c="g")
    plt.plot(np.linspace(0,np.max(ycal[0])), np.linspace(0,np.max(ycal[0])), color="black", ls="--")
    plt.axvline(x=np.max(ycal[0]), c='b')
    plt.axvline(x=np.min(ycal[0]), c='b')
    plt.text(np.max(ycal[0])-(np.max(ycal[0])-np.min(ycal[0]))/2, np.min(ycal[1])-0.05*(np.max(ycal[1])-np.min(ycal[1])), 
             "Calibration range : " + str(round(np.min(ycal[0]), 2)) + " - " + str(round(np.max(ycal[0]), 2)), 
             c="b", size=16, ha="center", va="center")
    plt.xlim(np.min(ycal[0])-0.1*(np.max(ycal[0])-np.min(ycal[0])), np.max(ycal[0])+0.1*(np.max(ycal[0])-np.min(ycal[0])))
    plt.ylim(np.min(ycal[1])-0.1*(np.max(ycal[1])-np.min(ycal[1])), np.max(ycal[1])+0.1*(np.max(ycal[1])-np.min(ycal[1])))
    
    
    
    plt.subplot(2, 2, 2)
    plt.scatter(yval_obs, yval_pred, c = 'red', alpha=0.1)
    plt.text(yval_obsn+0.05*(yval_obsx-yval_obsn), 
             np.max(yval_pred), "RMSE = " + str(rmse_val) + "\n" + "R² = " + str(r2_val))
    plt.title('Validation', size=16, color="red")
    plt.plot(np.linspace(0,np.max(yval_obs)), np.linspace(0,np.max(yval_obs)), color="black", ls="--")
    plt.axvline(x=yval_obsx, c='b')
    plt.axvline(x=yval_obsn, c='b')
    plt.text(yval_obsx-(yval_obsx-yval_obsn)/2, np.min(yval_pred)-0.05*(np.max(yval_pred)-np.min(yval_pred)), 
             "Validation range : " + str(round(yval_obsn, 2)) + " - " + str(round(yval_obsx, 2)), 
             c="b", size=16, ha="center", va="center")
    plt.xlim(yval_obsn-0.1*(yval_obsx-yval_obsn), yval_obsx+0.1*(yval_obsx-yval_obsn))
    plt.ylim(np.min(yval_pred)[0]-0.1*(np.max(yval_pred)[0]-np.min(yval_pred)[0]), np.max(yval_pred)[0]+0.1*(np.max(yval_pred)[0]-np.min(yval_pred)[0]))
    
    
    # Plot residues
    plt.subplot(2, 2, 3)
    plt.scatter(ycal[0], resid_cal, color="g", alpha=0.1)
    plt.xlim(np.min(ycal[0])-0.1*(np.max(ycal[0])-np.min(ycal[0])), np.max(ycal[0])+0.1*(np.max(ycal[0])-np.min(ycal[0])))
    plt.ylim(np.min(resid_cal)-0.1*(np.max(resid_cal)-np.min(resid_cal)), np.max(resid_cal)+0.1*(np.max(resid_cal)-np.min(resid_cal)))
    plt.axvline(x=np.max(ycal[0]), c='b')
    plt.axvline(x=np.min(ycal[0]), c='b')
    plt.axhline(y=0, c='k', ls=":")
    
    plt.subplot(2, 2, 4)
    plt.scatter(yval_obs, resid_val, color="r", alpha=0.1)
    plt.xlim(yval_obsn-0.1*(yval_obsx-yval_obsn), yval_obsx+0.1*(yval_obsx-yval_obsn))
    plt.ylim(np.min(resid_val)-0.1*(np.max(resid_val)-np.min(resid_val)), np.max(resid_val)+0.1*(np.max(resid_val)-np.min(resid_val)))
    plt.axvline(x=yval_obsx, c='b')
    plt.axvline(x=yval_obsn, c='b')
    plt.axhline(y=0, c='k', ls=":")
    
    plt.savefig('./out/Graphs/'+ adir + "_Filter" + str(FILTER) + str(FILTER_DEPTH) + "_Noise" + str(NOISE) +  "_Batch" + str(BATCH_SIZE) + "_Epochs" + str(NB_EPOCHS) + '_PredObs.png', bbox_inches='tight', dpi=80)
    
    
    # Save perf in csv file
    today = date.today()
    model = models.load_model("./out/BACON_" +  adir + '_filt' + str(FILTER_DEPTH) + '_fold' + str(i) + '_model.hdf5')
    LAYERS = model.layers
    
    Splitt = "Cf. Publi"
    Res=[today, adir, len(ycal[0]), len(yval_obs), FILTER, FILTER_DEPTH, NOISE[0], NOISE[1], NOISE[2], SCALERS_RANGE[0], SCALERS_RANGE[1], NB_EPOCHS_ran, BATCH_SIZE, K_FOLD_NB, OPTIMIZER, REGULARIZATION, DROPOUT_RATE, SEED, Splitt, str(tensorflow.keras.__version__), LAYERS, rmse_cal, r2_cal, rmse_val, r2_val]
    
    def append_list_as_row(file_name, list_of_elem):
        # Open file in append mode
        with open(file_name, 'a+', newline='') as write_obj:
            # Create a writer object from csv module
            csv_writer = writer(write_obj, delimiter=";")
            # Add contents of list as last row in the csv file
            csv_writer.writerow(list_of_elem)
    append_list_as_row('./out/BaconRunResults/BaconRunResults.csv', Res)
    plt.close('all')