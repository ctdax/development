import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from statistics import mean
from LLcommonFunctions import returnTestSamplesSplitIntoSignalAndBackground, compareManyHistograms, returnBestCutValue, getLumiScaleFactor
import re
import numpy as np
import time
import os, sys
import uproot
import uproot3
from uncertainties import umath, unumpy
import math


def getvars(df, varlist):
    '''
    :param df: Pandas dataframe
    :param varlist: List of desired variables
    :return: Subsets the dataframe to only contain those variables
    '''
    print("Reducing the dataframe to the following variables:\n" + str(varlist) )
    newdf = pd.DataFrame()
    cols = list(df.columns)
    for col in cols:
        for var in varlist:
            if var in col:
                newdf[col] = df[col]

    return newdf


def get_trailing_number(s):
    '''
    :param s: String
    :return: Returns the number at the end of the string
    '''
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def manipulate_var_ranges(df, range1, range2):
    '''
    :param df: Pandas dataframe
    :param range1: Lower bound track range i.e. 1
    :param range2: Upper bound track range i.e. 5
    :return: Subsets the dataframe track columns
    '''
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

    return df


def manipulate_var_ranges_basedOn_jet(df, tracksPerJet, saveToPklFileName=None):
    '''
    :param df: Pandas dataframe containing the data
    :param tracksPerJet: Desired number of tracks per jet
    :param saveToPklFileName: Output filepath if desired
    :return: Subsets the dataframes columns by reducing the total number of track variables
    '''
    print("Changing the number of track columns per unique jet in an event to " + str(tracksPerJet))
    #Instatiate some placeholder dataframes and dictionaries
    d = {}
    iterations = 0
    itercounter = 0
    timer_start = time.time()

    #For each row in the df change the number of tracks per jet to 'tracksPerJet'. If there are not enough tracks for a
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
        nJetsinRow = min(n1,n2,n3,n4)

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
            print("Iterated through " + str(iterations) + " of " + str(len(df)) + " rows. Elapsed time = " + str(elapsed_time) +
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
    elapsed_time_minutes = round(elapsed_time_seconds/60,2)
    expected_total_time_to_finish = round((1/row_ratio)*elapsed_time_minutes,2)
    expected_time_remaining = round(expected_total_time_to_finish - elapsed_time_minutes,2)

    return elapsed_time_minutes, expected_time_remaining


def process(dfsignal, dfbackground, vars, testing_fraction, equal_bkg_sig_samples=False):
    '''
    :param dfsignal: Pandas dataframe containing signal data
    :param dfbackground: Pandas dataframe containing background data
    :param vars: Subset of desired variables passed as a list
    :param testing_fraction: Testing fraction used in machine learning
    :param equal_bkg_sig_samples: If true, will make sure the total number of samples in signal and background are equal
    :return: Returns training, testing, and validation data along with all labels
    '''
    # Sort by top variables
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
            print("Background samples reduced to equal signal samples. Background length = " + str(
                len(backgroundtopvars)) + ", Signal length = " + str(len(signaltopvars)))
        else:
            signaltopvars = signaltopvars.sample(n=bkglen)
            print("Signal samples reduced to equal background samples. Background length = " + str(
                len(backgroundtopvars)) + ", Signal length = " + str(len(signaltopvars)))
    else:
        print("Error, 'equal_bkg_sig_samples' is a boolean statement, parameter must be either true or false.")
        sys.exit()

# Insert binary column into each df
    pd.options.mode.chained_assignment = None  # Removes SettingWithCopyWarning which is a false positive
    signaltopvars['Result'] = 1
    backgroundtopvars['Result'] = 0
# Append bkg df to sig df
    dataset = signaltopvars.append(backgroundtopvars)
    data = dataset.loc[:, dataset.columns != 'Result']
    label = dataset.loc[:, 'Result']
# Normalize data
    if 'Filename' in vars:
        filecol = list(data['Filename'])
        data.drop(axis=1, columns='Filename', inplace=True)
        vars.remove('Filename')
        dataset_scaled = preprocessing.scale(data[vars])
        data = pd.DataFrame(dataset_scaled, columns=data.columns)
        data['Filename'] = filecol
    else:
        dataset_scaled = preprocessing.scale(data)
        data = pd.DataFrame(dataset_scaled, columns=data.columns)

# Split data into training, validation, and testing sets
    data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=testing_fraction, shuffle=True)
    data_train, data_val, label_train, label_val = train_test_split(data_train, label_train, test_size=testing_fraction, shuffle=True)

    return data_test, data_train, data_val, label_test, label_train, label_val, len(signaltopvars), len(backgroundtopvars)


def significance(model, data_test, label_test, testing_fraction, sig_nEvents, bkg_nEvents, minBackground=500, logarithmic=False):
    '''
    :param model: FFNN model
    :param data_test: Testing data
    :param label_test: Testing labels
    :param testing_fraction: Testing fraction used in machine learning
    :param sig_nEvents: Number of events in the signal
    :param bkg_nEvents: Number of events in the background
    :param minBackground: Minimum number of background samples desired in the significance cut
    :param logarithmic: If true, the y scale on the signal prediction plot will be logarithm transformed
    :return: Outputs plots and print statements
    '''
    signal_data_test, signal_labels_test, background_data_test, background_labels_test = \
        returnTestSamplesSplitIntoSignalAndBackground(data_test, label_test)
    pred_signal = model.predict(signal_data_test)
    pred_background = model.predict(background_data_test)

    _nBins = 40
    predictionResults = {'signal_pred': pred_signal, 'background_pred': pred_background}
    compareManyHistograms(predictionResults, ['signal_pred', 'background_pred'], 2, 'Signal Prediction', 'ff-NN Score', 0, 1,
                          _nBins, _normed=True, _testingFraction=testing_fraction, logarithmic=logarithmic)

    returnBestCutValue('ff-NN', pred_signal.copy(), pred_background.copy(), _minBackground=minBackground,
                       _testingFraction=testing_fraction, ll_nEventsGen=int(10e4*(sig_nEvents/62642)), qcd_nEventsGen=int(660740*(bkg_nEvents/138509)))


def accuracy(model, data_train, label_train, data_test, label_test):
    '''
    :param model: FFNN model
    :param data_train: Training data
    :param label_train: Training labels
    :param data_test: Testing data
    :param label_test: Testing labels
    :return: Accuracy for training and testing data
    '''
    trainscores_raw = []
    testscores_raw = []
    x = 0
    while x <= 4:
        trainscores_raw.append(model.evaluate(data_train, label_train)[1]*100)
        testscores_raw.append(model.evaluate(data_test, label_test)[1]*100)
        x+=1
    trainscores = ("Training Accuracy: %.2f%%\n" % (mean(trainscores_raw)))
    testscores = ("Testing Accuracy: %.2f%%\n" % (mean(testscores_raw)))

    return trainscores, testscores


def epoch_history(history):
    '''
    :param history: FFNN history
    :return: Outputs a plot of the accuracy and loss history of the training
    '''
#Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('ff-NN Accuracy')
    plt.ylabel('Accuracy [%]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
#Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('ff-NN Model Loss')
    plt.ylabel('Loss [A.U.]')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


def roc_plot(model, data_test, label_test):
    '''
    :param model: FFNN model
    :param data_test: Testing data
    :param label_test: Testing labels
    :return: Outputs an ROC plot
    '''
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


def fetchGeneratedEvents(root, printOutput=True):
    '''
    :param root: File path of the ROOT file
    :param printOutput: If true prints the number of generated events
    :return: Returns the events generated in the ROOT
    '''
    f = uproot3.open(root)
    nEvents = int(f['mfvWeight/h_sums'].values[f['mfvWeight/h_sums'].xlabels.index('sum_nevents_total')])
    if printOutput:
        r = root.split('\\')[-1]
        rootName = r.split('.')[0]
        print("{} contains {} generated events".format(rootName, nEvents))

    return nEvents

def fetchSampledEvents(root, signalcsv, bkgcsv, printOutput=True):
    '''
    :param root: File path of the ROOT
    :param signalcsv: File path of the signal csv data
    :param bkgcsv: File path of the background csv data
    :param printOutput: If true prints the number of sampled events
    :return: Returns the total number of events used from the ROOT
    '''
    if 'splitSUSY' in root:
        df = pd.read_csv(signalcsv)
    else:
        df = pd.read_csv(bkgcsv)
    r = root.split('\\')[-1]
    rootName = r.split('.')[0]
    dfSubset = df[df['Filename'].str.contains(rootName)]
    eventList = []
    for filename in dfSubset['Filename']:
        f1 = filename.split('Evt ')[-1]
        eventList.append(int(f1.split(' Vtx')[0]))
    nEvents = max(eventList) - min(eventList) + 1

    if printOutput:
        print("{} contains {} sampled events".format(rootName, nEvents))

    return nEvents


def calculateLumiScaleForDirectory(rootdirectory, crossSectionFilepath, signalcsvFilepath, bkgcsvFilepath, testingFraction=0.3, filepath=None, signalFlag='SplitSUSY'):
    '''
    :param rootdirectory: File path of the directory containing all ROOTs
    :param crossSectionFilepath: Filepath to the csv file containing all cross sections for each ROOT
    :param signalcsvFilepath: File path of the signal csv data
    :param bkgcsvFilepath: File path of the background csv data
    :param testingFraction: Testing fraction used during machine learning
    :param filepath: File path to which the output data will be saved to
    :param signalFlag: Flag used in the ROOT filenames to differentiate signal from background
    :return: Returns the pandas data frame containing the output data
    '''
    signalfiles, bkgfiles = files_from_directory_to_list(rootdirectory)
    crossdf = pd.read_csv(crossSectionFilepath)
    files = signalfiles + bkgfiles
    df = pd.DataFrame({'ROOT': [], 'Events Generated': [], 'Events in ROOT': [], 'Events Sampled': [], 'Lumi Scale Factor': []})
    for file in files:
        r = file.split('\\')[-1]
        rootName = r.split('.')[0]
        genEvt = fetchGeneratedEvents(file, printOutput=False)
        samEvt = fetchSampledEvents(file, signalcsvFilepath, bkgcsvFilepath, printOutput=False)
        testingImages = math.trunc(samEvt * testingFraction)

        root = uproot.open(file)
        ttree = root['mfvVertexTreer']['tree_DV']
        rootTotalEvt = len(ttree['vtx_tk_px'].array())
        index = crossdf.index[crossdf['ROOT'] == rootName][0]

        if signalFlag in file:
            isSignal = True
            llevts = genEvt
            qcdevts = 0
            ll_xsec = crossdf.at[index, 'Cross Section']
            qcd_xsec = 0
        else:
            isSignal = False
            llevts = 0
            qcdevts = genEvt
            ll_xsec = 0
            qcd_xsec = crossdf.at[index, 'Cross Section']

        lumi = getLumiScaleFactor(testingImages/rootTotalEvt, isSignal, llevts, qcdevts, ll_xsec, qcd_xsec)
        df.loc[len(df.index)] = [rootName, genEvt, rootTotalEvt, samEvt, lumi]
        if filepath != None:
            df.to_csv(filepath)

    return df


def files_from_directory_to_list(directory, signalflag='splitSUSY'):
    '''
    :param directory: Directory that contains the ROOT files
    :param signalflag: Flag in the filenames that differentiates the signal from the background
    :return: Returns a list of filepaths for the signal and background files
    '''
    signalfiles = []
    bkgfiles = []
    for file in os.listdir(directory):
        if signalflag in file:
            signalfiles.append(file)
        else:
            bkgfiles.append(file)

    for i in range(len(signalfiles)):
        signalfiles[i] = os.path.join(directory, signalfiles[i])
    for i in range(len(bkgfiles)):
        bkgfiles[i] = os.path.join(directory, bkgfiles[i])

    for i in range(len(signalfiles)):
        r = uproot.open(signalfiles[i])
        try:
            r['mfvVertexTreer']['tree_DV']['vtx_tk_px'].array()
        except ValueError:
            del signalfiles[i]
    for i in range(len(bkgfiles)):
        r = uproot.open(bkgfiles[i])
        try:
            r['mfvVertexTreer']['tree_DV']['vtx_tk_px'].array()
        except ValueError:
            del bkgfiles[i]

    return signalfiles, bkgfiles


def fetchBestCutOnVar(sig_df, bkg_df, min_background=50, equal_signal_background=True, printOutput=True):
    '''
    :param sig_df: Single column pandas data frame containing the signal for the desired variable
    :param bkg_df: Single column pandas data frame containing the background for the desired variable
    :param min_background: Minimum number of background required to make a cut
    :param equal_signal_background: If true makes sure that the total number of samples in background and signal are equal
    :param printOutput: If true prints the significance
    :return: Returns the true positive rate, false positive rate, best significance cut, and direction of the cut
    '''
    significance = 0
    nsig = 0
    nbkg = 0
    cut = 0
    direction = ''
    variable = sig_df.name

    if equal_signal_background:
        if len(sig_df) > len(bkg_df):
            sig_df = sig_df.sample(n=len(bkg_df))
        else:
            bkg_df = bkg_df.sample(n=len(sig_df))
    siglabel = [1] * len(sig_df)
    bkglabel = [0] * len(bkg_df)
    labels = siglabel + bkglabel
    dfsig = sig_df.sort_values(ascending=False, ignore_index=True)
    dfbkg = bkg_df.sort_values(ascending=False, ignore_index=True)

    #Greater than cuts
    loc = min_background - 1
    while loc <= len(bkg_df):
        try:
            greaterCut = round(dfbkg.iloc[[loc]].tolist()[0], 4)
        except IndexError:
            break
        tempNsig = len(dfsig[dfsig >= greaterCut])
        tempSignificance = tempNsig/math.sqrt(loc + 1)
        if tempSignificance > significance:
            significance = tempSignificance
            nsig = tempNsig
            nbkg = loc + 1
            cut = greaterCut
            direction = '>='
        loc += 1

    #Less than cuts
    loc = min_background
    while loc <= len(bkg_df):
        try:
            lessCut = round(dfbkg.iloc[[-loc]].tolist()[0], 4)
        except IndexError:
            break
        tempNsig = len(dfsig[dfsig <= lessCut])
        tempSignificance = tempNsig / math.sqrt(loc)
        if tempSignificance > significance:
            significance = tempSignificance
            nsig = tempNsig
            nbkg = loc
            cut = lessCut
            direction = '<='
        loc += 1

    predictions = ([1] * nsig) + ([0] * (len(dfsig) - nsig)) + ([1] * nbkg) + ([0] * (len(dfbkg) - nbkg))
    fpr, tpr, threshold = roc_curve(labels, predictions)

    if printOutput:
        print("nSig = {}, nBkg = {} for {} {} {}".format(nsig, nbkg, variable, direction, cut))

    return tpr, fpr, cut, direction, significance


def fetchBestCutForSimultaneousVars(sig_csv, bkg_csv, variable_list, min_background=50, iterations=3, equal_signal_background=True, printOutput=True):
    '''
    :param sig_csv: CSV of the signal data
    :param bkg_csv: CSV of the background data
    :param variable_list: Desired list of variables
    :param min_background: Minimum number of background required to make a cut
    :param iterations: Total number of iterations over all variables
    :param equal_signal_background: If true makes sure that the total number of samples in background and signal are equal
    :param printOutput: If true prints the significance
    :return: Returns the true positive rate and false positive rate
    '''
    global dfsig
    global dfbkg

    dfsig = getvars(pd.read_csv(sig_csv), variable_list)
    dfbkg = getvars(pd.read_csv(bkg_csv), variable_list)
    if equal_signal_background:
        if len(dfsig) > len(dfbkg):
            dfsig = dfsig.sample(n=len(dfbkg))
        else:
            dfbkg = dfbkg.sample(n=len(dfsig))
    siglabel = [1] * len(dfsig)
    bkglabel = [0] * len(dfbkg)
    nsig_original = len(dfsig)
    nbkg_original = len(dfbkg)
    labels = siglabel + bkglabel

    #Try iterating through each variable, finding the best cut for the given min background. Once every variable has been iterated
    #through, iterate through again x more times to fine tune.

    info = {'Variables Iterated': [], 'Cut List': [], 'Direction List': [], 'Significance List': []}
    iterable_min_bkg = min_background + (10 * iterations * len(variable_list))
    iterable = 0
    while iterable < iterations:
        for var in variable_list:
            if len(info['Variables Iterated']) != 0:
                for i in range(len(info['Variables Iterated'])):
                    if info['Direction List'][i] == '>=':
                        dfsig = dfsig[dfsig[info['Variables Iterated'][i]] >= info['Cut List'][i]]
                        dfbkg = dfbkg[dfbkg[info['Variables Iterated'][i]] >= info['Cut List'][i]]
                    else:
                        dfsig = dfsig[dfsig[info['Variables Iterated'][i]] <= info['Cut List'][i]]
                        dfbkg = dfbkg[dfbkg[info['Variables Iterated'][i]] <= info['Cut List'][i]]

            sigvar = dfsig[var]
            bkgvar = dfbkg[var]
            tpr, fpr, cut, direction, significance = fetchBestCutOnVar(sigvar, bkgvar, iterable_min_bkg, equal_signal_background, printOutput=printOutput)
            iterable_min_bkg -= 10
            if iterable == 0:
                info['Variables Iterated'].append(var)
                info['Cut List'].append(cut)
                info['Direction List'].append(direction)
                info['Significance List'].append(significance)
            else:
                for i in range(len(info['Variables Iterated'])):
                    if info['Variables Iterated'][i] == var:
                        if significance >= info['Significance List'][i]:
                            info['Cut List'][i] = cut
                            info['Direction List'][i] = direction
                            info['Significance List'][i] = significance
        iterable += 1

    nsig = len(dfsig)
    nbkg = len(dfbkg)
    predictions = ([1] * nsig) + ([0] * (nsig_original - nsig)) + ([1] * nbkg) + ([0] * (nbkg_original - nbkg))
    fpr, tpr, threshold = roc_curve(labels, predictions)

    return tpr, fpr


def ROCPlotForN_tpr_fpr(sig_csv, bkg_csv, variable_list, min_background=50, iterations=3, equal_signal_background=True, filepath=None, ml_pr=None):
    '''
    :param sig_csv: CSV of the signal data
    :param bkg_csv: CSV of the background data
    :param variable_list: Desired list of variables
    :param min_background: Minimum number of background required to make a cut
    :param iterations: Total number of iterations over all variables
    :param equal_signal_background: If true makes sure that the total number of samples in background and signal are equal
    :param filepath: If not None, saves the ROC plot to the given filepath
    :param ml_pr: Filepath to the machine learning csv tpr and fpr
    :return: Outputs an ROC plot for all desired parameters
    '''
    d = {'Name': [], 'tpr': [], 'fpr': []}
    if ml_pr != None:
        pr_df = pd.read_csv(ml_pr)
        d['Name'].append('FFNN')
        d['tpr'].append(pr_df['TPR'].tolist())
        d['fpr'].append(pr_df['FPR'].tolist())

    sigdf = pd.read_csv(sig_csv)
    bkgdf = pd.read_csv(bkg_csv)
    for var in variable_list:
        print("Calculating the TPR and FPR for {}".format(var))
        sigvar = sigdf[var]
        bkgvar = bkgdf[var]
        tpr, fpr, cut, direction, significance = fetchBestCutOnVar(sigvar, bkgvar, min_background, equal_signal_background, printOutput=False)
        d['Name'].append(var)
        d['tpr'].append(tpr)
        d['fpr'].append(fpr)

    tpr, fpr = fetchBestCutForSimultaneousVars(sig_csv, bkg_csv, variable_list, min_background, iterations, equal_signal_background, printOutput=False)
    d['Name'].append('Best Cuts')
    d['tpr'].append(tpr)
    d['fpr'].append(fpr)

    colors = plt.cm.jet(np.linspace(0, 1, len(d['Name'])))
    plt.title('ROC Curves for LL SUSY Production')
    for i in range(len(d['Name'])):
        if d['Name'][i] == 'FFNN' or d['Name'][i] == 'Best Cuts' or d['Name'][i] == 'CNN Predictions':
            roc_auc = round(auc(d['fpr'][i], d['tpr'][i]), 2)
            plt.plot(d['fpr'][i], d['tpr'][i], 'b', label='{} AUC = {}'.format(d['Name'][i], roc_auc), color=colors[i])
        else:
            plt.plot(d['fpr'][i], d['tpr'][i], 'b', color=colors[i])
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    if filepath != None:
        plt.savefig(filepath)
    else:
        plt.show()


def significance2(model, data_test, label_test, test_filenames, lumi_data_filepath, signalROOT='all', min_background=50,
                  plotBackground=True, outputFilepath=None):
    '''
    :param model: Machine learning model that will be used to make predictions
    :param data_test: Testing data
    :param label_test: Testing labels
    :param test_filenames: ROOT filenames for each test vertex
    :param lumi_data_filepath: Filepath to the luminosity data for all ROOTs, this can be calculated using the
            calculateLumiScaleForDirectory function above
    :param cross_section_filepath: Filepath to the csv file containing all cross sections for each ROOT
    :param signalROOT: String identifying the signal ROOT to be used during calculations. If set to 'all', it will
            calculate significance for all signals
    :param min_background: Minimum background needed to make a cut during significance calculations
    :param plotBackground: If True, plots the background data in this histogram
    :param outputFilepath: Output filepath for prediction histogram if desired
    :return: Returns the significance for the given predictions and ROOT parameters
    '''
    predictions = model.predict(data_test)
    test_filenames_fixed = []
    for i in test_filenames.tolist():
        i1 = i.split('/')[1]
        test_filenames_fixed.append(i1.split(' Evt')[0])


    df = pd.DataFrame({'ROOT': test_filenames_fixed, 'Prediction': predictions.flatten().round(3), 'Label': label_test.tolist()})
    lumidf = pd.DataFrame({'Prediction': np.linspace(0, 1, 1000).flatten().round(3)})
    for root in list(set(df['ROOT'])):
        subset = df[df['ROOT'] == root]
        values = subset['Prediction'].value_counts(ascending=True)
        lumidf[root] = np.zeros([1, 1000])[0]
        for value, n in values.iteritems():
            index = lumidf.index[lumidf['Prediction'] == round(value, 3)]
            lumidf.at[index, root] = n

    luminosity_data = pd.read_csv(lumi_data_filepath)
    for column in lumidf:
        if column == 'Prediction':
            continue
        lumi_scale_factor_index = luminosity_data.index[luminosity_data['ROOT'] == column]
        lumi_scale_factor = luminosity_data.at[lumi_scale_factor_index[0], 'Lumi Scale Factor']
        lumidf[column] = lumidf[column].apply(lambda x: x*lumi_scale_factor)

    bkg_roots = []
    sig_roots = []
    for root in list(set(df['ROOT'])):
        if 'mfv_splitSUSY' in root:
            sig_roots.append(root)
        else:
            bkg_roots.append(root)
    sigsubset = lumidf[sig_roots]
    lumidf['mfv_splitSUSYCombined'] = sigsubset.sum(axis=1)
    lumidf['mfv_splitSUSYCombined'] = unumpy.uarray(lumidf['mfv_splitSUSYCombined'], np.sqrt(lumidf['mfv_splitSUSYCombined']))
    bkgsubset = lumidf[bkg_roots]
    lumidf['Background Vertex Count'] = bkgsubset.sum(axis=1)
    lumidf['Background Vertex Count'] = unumpy.uarray(lumidf['Background Vertex Count'], np.sqrt(lumidf['Background Vertex Count']))

    if signalROOT != 'all':
        if type(signalROOT) != list:
            sroot = [signalROOT]
        else:
            sroot = signalROOT
    else:
        sroot = []
        for root in lumidf:
            if 'mfv_splitSUSY' in root:
                sroot.append(root)
    for sr in sroot:
        try:
            lumidf[sr] = unumpy.uarray(lumidf[sr], np.sqrt(lumidf[sr]))
        except TypeError:
            pass
        bestSignificance = -1
        bestCut = -1
        bestnSig = -1
        bestnBkg = -1

        df_pre_cut = lumidf[['Prediction', sr, 'Background Vertex Count']]
        for index, row in df_pre_cut.iterrows():
            cut = row['Prediction']
            df_after_cut = df_pre_cut[df_pre_cut['Prediction'] > cut]
            nsig = df_after_cut[sr].sum()
            nbkg = df_after_cut['Background Vertex Count'].sum()

            try:
                #if nsig.n <= 0:
                #    continue
                if nbkg.n < min_background:
                    continue
            except AttributeError:
                continue
            significance = umath.sqrt(2*((nsig+nbkg)*umath.log(1+nsig/nbkg)-nsig))
            if significance.n > bestSignificance:
                bestSignificance = significance
                bestCut = cut
                bestnSig = nsig.n
                bestnBkg = nbkg.n

        svalues, bvalues, centers = \
            dataToHistogramArray(unumpy.nominal_values(lumidf[sr]), unumpy.nominal_values(lumidf['Background Vertex Count']), nbins=40)
        plt.suptitle("{} and Background FFNN prediction histograms".format(sr))
        plt.hist(centers, bins=40, color='blue', label=sr, alpha=0.5, density=True, weights=svalues)
        if plotBackground:
            plt.hist(centers, bins=40, color='orange', label='Background', alpha=0.5, density=True, weights=bvalues)
        print(centers, svalues)
        plt.ylim(bottom=0)
        plt.xlim(0, 1)
        plt.ylabel('Density')
        plt.xlabel('FFNN Prediction Score')
        plt.legend()
        if outputFilepath != None:
            plt.savefig(outputFilepath)
        else:
            plt.show()
        print("signalROOT = {}, nSig = {}, nBkg = {} with significance = {} +/- {} for FFNN score > {}".format(sr, bestnSig, bestnBkg,
                                                                                                        round(bestSignificance.n, 2),
                                                                                                        round(bestSignificance.s, 2), bestCut))


def dataToHistogramArray(signalLumiVertices, backgroundLumiVertices, nbins=50):
    '''
    :param signalLumiVertices: An array of the number of signal vertices at each FFNN prediction post lumi scale. This
            calculation is done in the significance function
    :param backgroundLumiVertices: An array of the number of background vertices at each FFNN prediction post lumi scale. This
            calculation is done in the significance function
    :param outputFilepath: Filepath to save the bar graph to
    :return: Signal N values per bin, signal error per bin, signal bin centers per bin,
             background N values per bin, background error per bin, background bin centers per bin
    '''
    df = pd.DataFrame({'Prediction': np.linspace(0, 1, 1000), 'Signal': signalLumiVertices, 'Background': backgroundLumiVertices})
    pred_width = 1/nbins
    lower_bound = 0
    upper_bound = lower_bound + pred_width
    svalues = []
    bvalues = []
    centers = []
    while lower_bound < 1:
        bin_center = (lower_bound + upper_bound)/2
        if upper_bound >= 1:
            dfcut = df[(df['Prediction'] >= lower_bound) & (df['Prediction'] <= upper_bound)]
        else:
            dfcut = df[(df['Prediction'] >= lower_bound) & (df['Prediction'] < upper_bound)]
        svalues.append(dfcut['Signal'].sum())
        bvalues.append(dfcut['Background'].sum())
        centers.append(bin_center)
        lower_bound += pred_width
        upper_bound += pred_width

    return svalues, bvalues, centers


def pygram11(x, nbins, range, underflow=True, overflow=True, weights=None):
    """Histogram weighted data with potential under/overflow.

    Parameters
    ----------
    x : array_like
        Data to histogram.
    nbins : int
        Total number of bins.
    range : (float, float)
        Definition of binning max and min.
    underflow : bool
        Include undeflow data in the first bin.
    overflow : bool
        Include overflow data in the last bin.
    weights : array_like, optional
        Weights associated with each element of ``x``.

    Returns
    -------
    numpy.ndarray
        Total bin values.
    numpy.ndarray
        Poisson uncertainty on each bin count.
    numpy.ndarray
        Bin centers.
    numpy.ndarray
        Bin edges.

    """
    if weights is not None:
        if weights.shape != x.shape:
            raise ValueError(
                "Unequal shapes x: {}; weights: {}".format(
                    x.shape, weights.shape
                )
            )
    xmin, xmax = range
    edges = np.linspace(xmin, xmax, nbins + 1)
    neginf = np.array([-np.inf], dtype=np.float32)
    posinf = np.array([np.inf], dtype=np.float32)
    bins = np.concatenate([neginf, edges, posinf])
    if weights is None:
        hist, bin_edges = np.histogram(x, bins=bins)
    else:
        hist, bin_edges = np.histogram(x, bins=bins, weights=weights)

    n = hist[1:-1]
    if underflow:
        n[0] += hist[0]
    if overflow:
        n[-1] += hist[-1]

    if weights is None:
        u = np.sqrt(n)
    else:
        bin_sumw2 = np.zeros(nbins + 2, dtype=np.float32)
        digits = np.digitize(x, edges)
        for i in range(nbins + 2):
            bin_sumw2[i] = np.sum(
                np.power(weights[np.where(digits == i)[0]], 2)
            )
        u = bin_sumw2[1:-1]
        if underflow:
            u[0] += bin_sumw2[0]
        if overflow:
            u[-1] += bin_sumw2[-1]
        u = np.sqrt(u)

    centers = np.delete(edges, [0]) - (np.ediff1d(edges) / 2.0)
    return n, u, centers, edges