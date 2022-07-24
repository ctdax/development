from keras.regularizers import l2
import LLFFNN_Utils
import time
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

#Filepaths
signal_filepath = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\pkl\1D_eventstructured_allvars_GluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-800_CTau-1mm.pkl"
background_filepath = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\pkl\1D_eventstructured_allvars_QCD_HT1000to1500.pkl"
modelcheckpoint_filepath = r"C:\Users\Colby\Box Sync\Neu-work\Longlive master\machine learning\
                1DReducedGluinoGluinoToNeutralinoNeutralinoTo2T2B2S_M-800_CTau-1mm\FFNN\checkpoints\bestmodel.hdf5"

#Hyperparameters
desired_variables = ['jet_eta', 'jet_energy', 'jet_phi', 'jet_pt', 'jet_ntracks', 'tk_which_pv', 'tk_phi', 'tk_eta']
testing_fraction = 0.2
batch_size = 1024
dropout = 0.4
epochs = 50
patience = 10
min_delta = 0.01
input_layer_nodes = 5000
hidden_layer1_nodes = 500
l2_reg = l2(0.0125)

dfsignal = pd.read_csv(signal_filepath)
dfbackground = pd.read_csv(background_filepath)

#Reduce the dataframe to only the desired variables
dfsignal = LLFFNN_Utils.getvars(dfsignal, desired_variables)
dfbackground = LLFFNN_Utils.getvars(dfbackground, desired_variables)

#Change variable range
dfsignal = LLFFNN_Utils.manipulate_var_ranges(dfsignal, range1=5, range2=5)
dfbackground = LLFFNN_Utils.manipulate_var_ranges(dfbackground, range1=5, range2=5)

vars = []
for var in dfsignal:
    vars.append(var)
varlen = len(vars)

#Preprocess data
data_test, data_train, data_val, label_test, label_train, label_val, sig_nEvents, bkg_nEvents = LLFFNN_Utils.process(dfsignal, dfbackground, vars, testing_fraction, equal_bkg_sig_samples=True)

#Start train model timer
start_time = time.time()

#Create model
model = Sequential()
model.add(Dense(input_layer_nodes, activation='relu', input_dim=varlen, kernel_regularizer=l2_reg))
model.add(Dropout(dropout))
model.add(BatchNormalization())
model.add(Dense(hidden_layer1_nodes, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Setup callbacks
fit_callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=patience, min_delta=min_delta, restore_best_weights=True),
ModelCheckpoint(filepath=modelcheckpoint_filepath, monitor='val_loss', mode='min', verbose=0, save_best_only=True)]

# Train model
history = model.fit(data_train, label_train, validation_data=(data_val, label_val), epochs=epochs,
                    callbacks=fit_callbacks, verbose=2, batch_size=batch_size)


#Finish train model timer
model_time = time.time() - start_time

#Analyze model
trainscores, testscores = LLFFNN_Utils.accuracy(model, data_train, label_train, data_test, label_test)
print(trainscores, '\n', testscores)
print("--- Train model time %s seconds ---" % model_time)
LLFFNN_Utils.significance(model, data_test, label_test, testing_fraction, sig_nEvents=sig_nEvents,
                          bkg_nEvents=bkg_nEvents, minBackground=500, logarithmic=True)
LLFFNN_Utils.epoch_history(history)
LLFFNN_Utils.roc_plot(model, data_test, label_test)
model.summary()