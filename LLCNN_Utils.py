import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from statistics import mean
from LLcommonFunctions import returnTestSamplesSplitIntoSignalAndBackground, compareManyHistograms, returnBestCutValue
from zipfile import ZipFile
from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization, Input, concatenate
import re
import numpy as np
import time
import os, sys
import math as m
import random


def extractfiles(dihiggs_filepath):
    # Extract dihiggs csv file from zip
    with ZipFile(dihiggs_filepath, 'r') as zip:
        zip.extractall()
        dihiggs_file = zip.namelist()[0]

    return dihiggs_file


def deletefiles(dihiggs_file):
    # Deletes files from current directory
    if os.path.exists(dihiggs_file):
        os.remove(dihiggs_file)
    else:
        print("Cannot delete file, it does not exist")


def getvars(df, varlist):
    # Return a new dataframe with only wanted variables from the old dataframe
    print("Reducing the dataframe to the following variables:\n" + str(varlist))
    newdf = pd.DataFrame()
    cols = list(df.columns)
    for col in cols:
        for var in varlist:
            if var in col:
                newdf[col] = df[col]

    return newdf


def get_trailing_number(s):
    # Get the value of the numbers at the end of a string
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def manipulate_var_ranges(df, range1, range2, number_list=None):
    # Reduce the range of variable columns in the dataframe. Can give a range or a list of numbers
    if number_list == None:
        print("Changing the range of variables to " + str(range1) + "-" + str(range2))
        for var in df:
            if 'jet' in var:
                pass
            else:
                i = get_trailing_number(var)
                if i < range1:
                    del df[var]
                if i > range2:
                    del df[var]
    else:
        print("Changing the variable numbers to " + str(number_list))
        for var in df:
            i = get_trailing_number(var)
            if i in number_list:
                pass
            else:
                del df[var]

    return df


def manipulate_var_ranges_basedOn_jet(df, tracksPerJet, saveToPklFileName=None):
    # Reduce the range of track variables per jet to specified value
    print("Changing the number of track columns per unique jet in an event to " + str(tracksPerJet))
    # Instatiate some placeholder dataframes and dictionaries
    d = {}
    iterations = 0
    itercounter = 0
    timer_start = time.time()

    # For each row in the df change the number of tracks per jet to 'tracksPerJet'. If there are not enough tracks for a
    # particular jet in a given row, delete that row
    for index, row in df.iterrows():
        jet = []
        for var in df:
            if 'tk_which_Jet' in var:
                jet.append(df.at[index, var])
        n1 = sum(item == 1 for item in jet)
        n2 = sum(item == 2 for item in jet)
        n3 = sum(item == 3 for item in jet)
        n4 = sum(item == 4 for item in jet)
        nJetsinRow = min(n1, n2, n3, n4)

        for var in df:
            if 'jet' in var:
                if var in d.keys():
                    pass
                else:
                    d[var] = df[var]
            else:
                if get_trailing_number(var) == 1:
                    l = range(1, tracksPerJet + 1)
                    d = appendToDict_forJetManipulation(d=d, df=df, var=var, index=index, tracksPerJet=tracksPerJet,
                                                        nJetsinRow=nJetsinRow, trailingNumberRange=l)
                elif get_trailing_number(var) == 1 + tracksPerJet:
                    l = range(1 + tracksPerJet, (2 * tracksPerJet) + 1)
                    d = appendToDict_forJetManipulation(d=d, df=df, var=var, index=index, tracksPerJet=tracksPerJet,
                                                        nJetsinRow=nJetsinRow, trailingNumberRange=l)
                elif get_trailing_number(var) == 1 + (2 * tracksPerJet):
                    l = range(1 + (2 * tracksPerJet), (3 * tracksPerJet) + 1)
                    d = appendToDict_forJetManipulation(d=d, df=df, var=var, index=index, tracksPerJet=tracksPerJet,
                                                        nJetsinRow=nJetsinRow, trailingNumberRange=l)
                elif get_trailing_number(var) == 1 + 3 * tracksPerJet:
                    l = range(1 + (3 * tracksPerJet), (4 * tracksPerJet) + 1)
                    d = appendToDict_forJetManipulation(d=d, df=df, var=var, index=index, tracksPerJet=tracksPerJet,
                                                        nJetsinRow=nJetsinRow, trailingNumberRange=l)

        # Print iteration number every 5% iterations
        if itercounter == int(round(df.shape[0] / 20)):
            iterations += itercounter
            end_time = time.time()
            row_ratio = round(iterations / len(df), 2)
            elapsed_time, expected_finish_time = calculate_predicted_time(timer_start, end_time,
                                                                          row_ratio)
            print("Iterated through " + str(iterations) + " of " + str(len(df)) + " rows. Elapsed time = " + str(
                elapsed_time) +
                  " minutes. Expected finish time = " + str(expected_finish_time) + " more minutes.")
            itercounter = 0
        itercounter += 1

    # Pass the dictionary into a new df
    rangedFixedDF = manipulate_var_ranges(df, 1, tracksPerJet * 4)
    dfnew = pd.DataFrame(columns=list(rangedFixedDF.columns))

    for column in dfnew:
        for key in d.keys():
            if key == column:
                try:
                    dfnew[column] = d[key]
                except ValueError:
                    print("An error occured when trying to insert " + column + " into the new dataframe")
                    dfnew[column] = pd.Series(data=d[key])

    dfnew = dfnew.dropna(axis=0)
    if saveToPklFileName != None:
        print("Saving the new dataframe to pickle")
        dfnew.to_pickle(saveToPklFileName)

    return dfnew


def appendToDict_forJetManipulation(d, df, var, index, tracksPerJet, nJetsinRow, trailingNumberRange):
    l = trailingNumberRange
    trailess_name = var.replace(str(get_trailing_number(var)), '')
    if nJetsinRow >= tracksPerJet:
        for i in l:
            NewName = trailess_name + str(i)
            if NewName in d.keys():
                d[NewName].append(df[NewName][index])
            else:
                d[NewName] = []
                d[NewName].append(df[NewName][index])
    else:
        for i in l:
            NewName = trailess_name + str(i)
            if NewName in d.keys():
                d[NewName].append(np.nan)
            else:
                d[NewName] = []
                d[NewName].append(np.nan)

    return d


def calculate_predicted_time(start, end, row_ratio):
    elapsed_time_seconds = end - start
    elapsed_time_minutes = round(elapsed_time_seconds / 60, 2)
    expected_total_time_to_finish = round((1 / row_ratio) * elapsed_time_minutes, 2)
    expected_time_remaining = round(expected_total_time_to_finish - elapsed_time_minutes, 2)

    return elapsed_time_minutes, expected_time_remaining


def process(dfsignal, dfbackground, vars, testing_fraction, equal_bkg_sig_samples=False):
    # Sort by top variables variables
    signaltopvars = dfsignal[vars]
    backgroundtopvars = dfbackground[vars]

    # If equal_bkg_sig_samples is False, does not change the csv, otherwise qcd and dihiggs samples will be equal
    if equal_bkg_sig_samples == False:
        pass
    elif equal_bkg_sig_samples == True:
        bkglen = len(backgroundtopvars)
        siglen = len(signaltopvars)
        if bkglen > siglen:
            backgroundtopvars = backgroundtopvars.sample(n=siglen)
            print("Background samples reduced to equal signal samples. QCD length = " + str(
                len(backgroundtopvars)) + ", dihiggs length = " + str(len(signaltopvars)))
        else:
            signaltopvars = signaltopvars.sample(n=bkglen)
            print("Signal samples reduced to equal background samples. QCD length = " + str(
                len(backgroundtopvars)) + ", dihiggs length = " + str(len(signaltopvars)))
    else:
        print("Error, 'equal_bkg_sig_samples' is a boolean statement, parameter must be either true or false.")
        sys.exit()

    # Insert binary column into each df
    pd.options.mode.chained_assignment = None  # Removes SettingWithCopyWarning which is a false positive
    signaltopvars['Result'] = 1
    backgroundtopvars['Result'] = 0
    # Append bkg df to sig df
    dataset = signaltopvars.append(backgroundtopvars)
    # Normalize data
    dataset[vars] = preprocessing.scale(dataset[vars])
    normal_set_df = pd.DataFrame(dataset, columns=dataset.columns)
    # Seperate dataset from results
    data = normal_set_df.loc[:, normal_set_df.columns != 'Result']
    label = normal_set_df.loc[:, 'Result']
    # Split data into training, validation, and testing sets
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=testing_fraction,
                                                                      shuffle=True)  # 80% train, 20% test
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=testing_fraction,
                                                                    shuffle=True)  # 80% train, 20% validation

    return data_test, data_train, data_val, label_test, label_train, label_val, len(signaltopvars), len(
        backgroundtopvars)


def largest_root(n):
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def round_up_to_odd(f):
    return int(np.ceil(f+1) // 2 * 2 + 1)


def reshape_data(train_data, test_data, validation_data): #Reshape and normalize 2D array data to 3D image data
    originalTrainShape = train_data.shape
    originalTestShape = test_data.shape
    originalValidationShape = validation_data.shape
    n = largest_root(train_data.shape[1])
    diff = train_data.shape[1] - (n**2)
    if diff != 0:
        while diff != 0:
            col_index = random.randint(0, train_data.shape[1]-1)
            col_name = train_data.columns[col_index]
            try:
                train_data.drop(columns=col_name, inplace=True)
                test_data.drop(columns=col_name, inplace=True)
                validation_data.drop(columns=col_name, inplace=True)
                diff -= 1
            except KeyError:
                continue
    train = train_data.values
    test = test_data.values
    val = validation_data.values
    reshape_train = train.reshape(train.shape[0], n, n, 1).astype('float32')
    reshape_test = test.reshape(test.shape[0], n, n, 1).astype('float32')
    reshape_val = val.reshape(val.shape[0], n, n, 1).astype('float32')
    print("Train data shape transformed from " + str(originalTrainShape) + " to " + str(reshape_train.shape))
    print("Test data shape transformed from " + str(originalTestShape) + " to " + str(reshape_test.shape))
    print("Validation data shape transformed from " + str(originalValidationShape) + " to " + str(reshape_val.shape))

    return reshape_train, reshape_test, reshape_val


def CNNTestSamplesSplitIntoSignalAndBackground(data_test_list, label_test): #Splits testing data into signal and background for significance calculations, assumes np.ndarray
    label_test = np.array(label_test)
    signal = []
    background = []
    for channel in data_test_list:
        signal.append(np.empty(shape=(0, channel.shape[1], channel.shape[2])))
        background.append(np.empty(shape=(0, channel.shape[1], channel.shape[2])))

    for t in range(len(data_test_list)):
        for i in range(len(label_test)):
            if label_test[i] == 1:
                signal[t] = np.append(signal[t], [data_test_list[t][i]], axis=0)
            else:
                background[t] = np.append(background[t], [data_test_list[t][i]], axis=0)

    return signal, background


def create_multichannel_model(vars, jetinput_shape, trackinput_shape, jet_filtersize, track_filtersize, jetnum_filters,
                              tracknum_filters, jet_maxpool, track_maxpool, jet_stride, track_stride, dropout): #Create set of channels for multichannel CNN
    channel_list = []
    input_list = []
    for var in vars:
        if 'jet' in var:
            inputs = Input(shape=jetinput_shape)
            conv = Conv1D(filters=jetnum_filters, kernel_size=jet_filtersize, activation='relu', strides=jet_stride, padding='same')(inputs)
            drop = Dropout(dropout)(conv)
            batch = BatchNormalization()(drop)
            if jet_maxpool != None:
                pool = MaxPooling1D(pool_size=jet_maxpool)(batch)
                flat = Flatten()(pool)
            else:
                flat = Flatten()(batch)
        else:
            inputs = Input(shape=trackinput_shape)
            conv = Conv1D(filters=tracknum_filters, kernel_size=track_filtersize, activation='relu', strides=track_stride, padding='same')(inputs)
            drop = Dropout(dropout)(conv)
            batch = BatchNormalization()(drop)
            if track_maxpool != None:
                pool = MaxPooling1D(pool_size=track_maxpool)(batch)
                flat = Flatten()(pool)
            else:
                flat = Flatten()(batch)

        input_list.append(inputs)
        channel_list.append(flat)
    merged = concatenate(channel_list)

    return merged, input_list


def splitDatasetForChannels(data, vars, jetlength, tracklength): #Split the numpy data into respective variable datasets
    columncounter = 0
    columnlist = []
    for var in vars:
        varcolumns = []
        if 'jet' in var:
            jetcounter = 0
            while jetcounter < jetlength:
                varcolumns.append(columncounter)
                columncounter+=1
                jetcounter+=1
        else:
            trackcounter = 0
            while trackcounter < tracklength:
                varcolumns.append(columncounter)
                columncounter += 1
                trackcounter += 1
        columnlist.append(varcolumns)

    splitData = []
    for set in columnlist:
        splitData.append(data[:, set])

    return splitData


def significance(model, data_test_list, label_test, testing_fraction, sig_nEvents, bkg_nEvents, minBackground=500,
                 logarithmic=False):
    signal_data_test, background_data_test = CNNTestSamplesSplitIntoSignalAndBackground(data_test_list, label_test)
    pred_signal = model.predict(signal_data_test)
    pred_background = model.predict(background_data_test)
    # Plot significance histograms
    _nBins = 40
    predictionResults = {'signal_pred': pred_signal, 'background_pred': pred_background}
    compareManyHistograms(predictionResults, ['signal_pred', 'background_pred'], 2, 'Signal Prediction', 'CNN Score',
                          0, 1,
                          _nBins, _normed=True, _testingFraction=testing_fraction, logarithmic=logarithmic)
    # Show significance
    returnBestCutValue('CNN', pred_signal.copy(), pred_background.copy(), _minBackground=minBackground,
                       _testingFraction=testing_fraction, ll_nEventsGen=int(10e4 * (sig_nEvents / 62642)),
                       qcd_nEventsGen=int(660740 * (bkg_nEvents / 138509)))


def accuracy(model, data_train, label_train, data_test, label_test):
    trainscores_raw = []
    testscores_raw = []
    x = 0
    while x <= 4:
        trainscores_raw.append(model.evaluate(data_train, label_train)[1] * 100)
        testscores_raw.append(model.evaluate(data_test, label_test)[1] * 100)
        x += 1
    trainscores = ("Training Accuracy: %.2f%%\n" % (mean(trainscores_raw)))
    testscores = ("Testing Accuracy: %.2f%%\n" % (mean(testscores_raw)))

    return trainscores, testscores


def epoch_history(history):
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('ff-NN Accuracy')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('ff-NN Model Loss')
    plt.ylabel('Loss [A.U.]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def roc_plot(model, data_test, label_test):  # Plot roc curve with auc score
    predictions = model.predict(data_test)
    fpr, tpr, threshold = roc_curve(label_test, predictions)
    roc_auc = auc(fpr, tpr)

    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()