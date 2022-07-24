import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
import numpy as np
import pandas as pd
import pickle5
import LLCNN_Utils
import time

### Author: Colby Thompson 5/31/2021
### Code designed to train LL particle data on a CNN

###Import and manipulate datasets
sigf = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\pkl\1D_eventstructured_JetOrdered_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-800_CTau-1mm.pkl"
bkgf = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\pkl\1D_eventstructured_JetOrdered_QCD_HT1000to1500.pkl"
signal_file = open(sigf, 'rb')
dfsignal = pickle5.load(signal_file)
dfsignal.replace([np.inf, -np.inf], 0, inplace=True)
dfsignal = dfsignal.dropna(axis=1)
signal_file.close()
background_file = open(bkgf, 'rb')
dfbackground = pickle5.load(background_file)
dfbackground.replace([np.inf, -np.inf], 0, inplace=True)
dfbackground = dfbackground.dropna(axis=1)
background_file.close()

# Reduce size of dfs
#dfsignal = dfsignal.sample(frac=0.1)
#dfbackground = dfbackground.sample(frac=0.1)

#Reduce the dataframe to only the desired variables
desired_variables = ['jet_eta', 'jet_energy', 'jet_phi', 'jet_pt', 'jet_ntracks']
dfsignal = LLCNN_Utils.getvars(dfsignal, desired_variables)
dfbackground = LLCNN_Utils.getvars(dfbackground, desired_variables)

#Change variable range
jetlength = 4
tracklength = 4
dfsignal = LLCNN_Utils.manipulate_var_ranges(dfsignal, range1=1, range2=tracklength, number_list=[1,4])
dfbackground = LLCNN_Utils.manipulate_var_ranges(dfbackground, range1=1, range2=tracklength, number_list=[1,4])

vars = []
for var in dfsignal:
    vars.append(var)
varlen = len(vars)

#Preprocess the data
testing_fraction = 0.3
data_test, data_train, data_val, label_test, label_train, label_val, sig_nEvents, bkg_nEvents = LLCNN_Utils.process(dfsignal, dfbackground, vars, testing_fraction, equal_bkg_sig_samples=True)

#Hyperparameters
jetfilter_size = 2 #Use odd NxN filter sizes, larger filter sizes if you believe a big amount of pixels are necessary for the network to recognize the object
jetnum_filters = 32 #Number of filters
jetmaxpool_size = 1 #2x2 max pooling size

trackfilter_size = 2 #Use odd NxN filter sizes, larger filter sizes if you believe a big amount of pixels are necessary for the network to recognize the object
tracknum_filters = 32 #Number of filters
trackmaxpool_size = 1 #2x2 max pooling size

input_size = int(data_train.shape[1]/len(desired_variables)) #Input image pixel size
batch_size = 1024 #Batch size used during training
steps_per_epoch = int(len(data_train)/batch_size) #Number of iterations in each training epoch, usually training size divided by batch size
validation_steps = int(len(data_val)/batch_size)
epochs = 2 #Number of epochs during training
dropout = 0.35 #Dropout value for training
l2_reg = l2(0.1375) #l2 regularizer used for overfitting prevention

#Start train model timer
start_time = time.time()

#Make the datasets 3D in order for the CNN to work
data_train = np.array(data_train)
data_test = np.array(data_test)
data_val = np.array(data_val)
data_train = np.reshape(data_train, newshape=(data_train.shape[0], data_train.shape[1], 1))
data_test = np.reshape(data_test, newshape=(data_test.shape[0], data_test.shape[1], 1))
data_val = np.reshape(data_val, newshape=(data_val.shape[0], data_val.shape[1], 1))

#Create model
input_shape = (input_size, data_train.shape[2])
merged, input_list = LLCNN_Utils.create_multichannel_model(desired_variables, input_shape, jet_filtersize=jetfilter_size, track_filtersize=trackfilter_size,
                                                           jetnum_filters=jetnum_filters, tracknum_filters=tracknum_filters, jet_maxpool=jetmaxpool_size,
                                                           track_maxpool=trackmaxpool_size, dropout=dropout)

#Interpretation
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(1, activation='sigmoid')(dense1)
model = Model(inputs=input_list, outputs=outputs)
#Compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#Summarize
model.summary()

# Setup callbacks
fit_callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25, min_delta=0.005, restore_best_weights=True)]
#ModelCheckpoint(filepath=modelcheckpoint_filepath, monitor='val_loss', mode='min', verbose=0, save_best_only=True)]

#Split datasets for multichannel purposes
trainsplit = LLCNN_Utils.splitDatasetForChannels(data_train, vars=desired_variables, jetlength=jetlength, tracklength=tracklength)
testsplit = LLCNN_Utils.splitDatasetForChannels(data_test, vars=desired_variables, jetlength=jetlength, tracklength=tracklength)
valsplit = LLCNN_Utils.splitDatasetForChannels(data_val, vars=desired_variables, jetlength=jetlength, tracklength=tracklength)

#Set up and train on data
history = model.fit(trainsplit, label_train, validation_data=(valsplit, label_val),
                    batch_size=batch_size, epochs=epochs, verbose=2, callbacks=fit_callbacks)

#Finish train model timer
model_time = time.time() - start_time

#Evaluate model on testing data
print("--- Train model time %s seconds ---" % model_time)
LLCNN_Utils.significance(model, testsplit, label_test, testing_fraction, sig_nEvents, bkg_nEvents,
                         minBackground=500, logarithmic=False)
LLCNN_Utils.epoch_history(history)
LLCNN_Utils.roc_plot(model, testsplit, label_test)