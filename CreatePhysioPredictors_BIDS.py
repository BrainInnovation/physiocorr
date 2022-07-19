# -*- coding: utf-8 -*-
"""
CREATE PHYSIOLOGICAL NOISE PREDICTORS
This script can be used to:
   1.  Import *.json and *.tsv.gz physiological recordings that follow BIDS standard as well as the corresponding FMR (generated with BV 21.4 or newer)
   2.  Preprocess the cardiac signal derived from a PPU (peripheral pulse unit) and the respiratory signal derived from a breathing belt
   3.  Perform peak detection on the preprocessed physiological signals and extract different physiological variables (e.g. heart rate, breathing rate, respiratory volume time, etc.)
   4.  Extract volume- (or slice-) based fMRI triggers from the *.json and *.tsv.gz physiological recording files
   5.  Extract fMRI aquisition parameters from the FMR and create volume-based physiological predictors
   6.  Save SDM files with the filtered physiological signals
   7.  Save RETROICOR SDM predictors based on the method proposed by Glover:
       Glover, G. H., Li, T.-Q., & Ress, D. (2000). Image-based method for retrospective correction of physiological motion effects in fMRI: RETROICOR.
       Magnetic Resonance in Medicine, 44(1), 162-167.
   8.  Create physiological noise predictors saved in single sdm files, including:
	 	- (shifted) heart rate (HR), 
		- HR convolved with the cardiac response function (CRF),
		- breathing rate (BR),
		- respiratory flow (RF), 
		- the (shifted) envelope of the respiratory signal (ENV),
        - the (shifted) respiration variation (RV),
		- (shifted) respiration volume per time (RVT), 
        - RV convolved with the respiratory response function (RRF)
		- RVT convolved with the RRF
    9. if specified by the user, compute Pearson correlations of the task design (specified in a SDM file) with the derived physiological regressors
   10. if specified by the user, plot stimulation protocol (PRT) together with computed heart rate (HR) and/or breathing rate (BR) and compute the mean HR and BR per event in the PRT

Resulting files will be saved in a directory called 'PhysioOut', which is created in the same folder as the specified FMR file:
   11. save PNG figures showing the cardiac and respiratory signal, the identified peaks, the functional scan time and some derived physiological noise regressors
   12. save input, output and processing parameters in JSON files (*_InputParameters.json, *_OutputParameters.json) in the 'PhysioOut' folder
   13. save the resulting physiological noise regressors as SDM files and as *_PhysiologicalNoiseRegressors.tsv
   14. save the task x noise correlation matrix with the corresponding p-values as *_CorrelationMatrixNoiseTaskRegressors.tsv
   15. save a PNG file of the plotted stimulation protocol together with HR and/or BR
"""

__author__ = "Judith Eck"
__version__ = "0.1.0"
__date__ = "18-07-2022"
__name__ = "CreatePhysioPredictors_BIDS.py"


# =============================================================================
# Import required packages
# =============================================================================
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.ndimage.filters import uniform_filter1d
import pandas as pd
import copy
# needed for warning messages outside of BV
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication
# needed for warning message in BV
# from PythonQt.QtGui import QmessageBox


import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle, Patch
import seaborn as sns
import json
import os.path
from datetime import datetime
import sys
from brainvoyagertools import sdm, prt

# this sets the Physio, FMR, SDM, PRT file names, if this script is called
# via the batch processing script
if len(sys.argv) > 2:
    physio_json_name = sys.argv[1]
    fmr_file_name = sys.argv[2]
    sdm_task_file_name = sys.argv[3]
    prt_task_file_name = sys.argv[4]
    plotdisp = sys.argv[5]

# change default plotting properties
mpl.rcParams['figure.figsize'] = (10, 6)     # set figure size
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['lines.linewidth'] = 0.5
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['axes.labelsize'] = 'small'
mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rcParams['ytick.labelsize'] = 'small'
mpl.rcParams['axes.titlesize'] = 'medium'
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.framealpha'] = 1
mpl.rcParams['legend.fontsize'] = 'small'
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['legend.loc'] = 'upper center'

app = QApplication(sys.argv)


# =============================================================================
# User-specified parameters
# =============================================================================

# specifies if cardiac respiratory raw data are saved together in a 
# single JSON/TSV file
# if only one of these measures exist please set this to TRUE
physio_1file = True

# cutoff frequencies in Hz for zero-phase second-order bandpass butterworth
cardiac_low = 0.5
cardiac_high = 8

# Definition of RETROICOR model, effects of cardiac and respiratory cycles
# estimated as a linear combination of sinusoidal signals, default order of
# correction terms (3c4r1i) based on: 
    # Harvey, A. K., Pattinson, K. T. S., Brooks, J. C. W., Mayhew, S. D., Jenkinson, M., & Wise, R. G. (2008). 
    # Brainstem functional magnetic resonance imaging: Disentangling signal from physiological noise. 
    # Journal of Magnetic Resonance Imaging, 28(6), 1337–1344. https://doi.org/10.1002/jmri.21623
# for alternative numbers (2c2r0i) see:
    # # Power, J. D., Plitt, M., Laumann, T. O., & Martin, A. (2017). Sources and implications of whole-brain fMRI 
    # signals in humans. NeuroImage, 146, 609-625. https://doi.org/10.1016/J.NEUROIMAGE.2016.09.038
order_cardiac = 2      # number of cardiac harmonics (based on brainstem imaging), change order if needed
order_resp = 2     # number of respiratory harmonics (based on brainstem imaging), change order if needed
order_cardresp = 0    # number of multiplicative harmonics (based on brainstem imaging)

# Outlier detection in identified heartbeats based on percentage rule
# Only used to let the user decide between pulse peak detection approaches
# e.g. Forcolin, F., Buendia, R., Candefjord, S., Karlsson, J., Sjöqvist, B. A., & Anund, A. (2018). Comparison of 
# outlier heartbeat identification and spectral transformation strategies for deriving heart rate variability indices 
# for drivers at different stages of sleepiness. Https://Doi.Org/10.1080/15389588.2017.1393073, 19, S112–S119. https://doi.org/10.1080/15389588.2017.1393073
# if outliers are detected within the heartbeats, the user can decide to try an alternative peak detection for the cardiac signal 
# inter-beat-intervals (IBIs) that differ by more than 30 percent (outlier_cardiac_threshold) from the mean of 
# their 5 neighboring IBIs (nIBIs) are considered outliers
outlier_cardiac_threshold = 30 # specified in percentage
nIBIs = 5   # number of neighboring IBIs taken into account

# Minimum interval between heartbeats in seconds, normal values vary 
# from 0.40 to 0.90 seconds (values depend on age, gender, health, level of training, physical and emotional state)
# this value is used for the alternative peak detection method only
min_hbi = 0.60

# Values for outlier removal in HR, based on Kassinopoulos, M., & Mitsis, G. D., 2019.
# used to remove outliers in the RESULTING Heartrate (HR) signal if there are sudden changes in HR due to noise.
# It needs to be visually inspected whether these sudden changes are indeed noise or true sudden changes in the PPU signal
hr_filloutliers_window = 25             # given in seconds
# outliers are defined as elements more than "hr_filloutliers_threshold"
# median absolute deviations (MAD) from the median
hr_filloutliers_threshold = 10          # normal range between 3-20


# Shifting of heartate (HR) regressor based on Shmueli et al., 2007
hr_shifts = np.arange(0, 25, 2)         # 0:2:24 seconds, or -12:6:12 seconds (when referring to Biancardi et al., 2009)
                                        # if no temporal shifts are required use np.arange(0,1)

# Values for outlier removal in respiratory signal, based on Power et al., 2020
# window and threshold to eliminate spike artifacts
resp_filloutliers_window = 0.25         # given in seconds
# outliers are defined as elements more than "resp_filloutliers_threshold"
# median absolute deviations (MAD) from the median
resp_filloutliers_threshold = 3         # default is 3


# Shifting of (respiration volume time (RVT) regressor based on Jo et al., 2010
rvt_shifts = np.arange(0, 21, 5)     # 0:5:20 seconds, or -24:6:18 seconds (when referring to Biancardi et al., 2009)
                                     # if no temporal shifts required use np.arange(0,1)
                                     
# Shifting of respiration variation (RV) regressor based on:
# Power, J. D., Plitt, M., Laumann, T. O., & Martin, A. (2017). Sources and implications of whole-brain fMRI 
# signals in humans. NeuroImage, 146, 609-625. https://doi.org/10.1016/J.NEUROIMAGE.2016.09.038
rv_shifts = np.arange(-7, 8, 7)     # -7:7:7 seconds
                                    # if no temporal shifts required use np.arange(0,1)

# Shifting of the respiratory envelope (ENV) regressor, similar to RV:
env_shifts = np.arange(-7, 8, 7)    # -7:7:7 seconds
                                    # if no temporal shifts required use np.arange(0,1)

# Minimum correlation value to be shown in heatmap of task-noise correlations
min_pvalue = 0.05



# =============================================================================
# Define some simple warning messages for the user
# =============================================================================


def showdialog_info(string_physio):
    '''
    messagebox used to inform user about missing physiological measures for the available functional run
    string_physio: unavailable physiological measure, e.g. "cardiac" or "respiratory"
    '''
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    msg.setText("No " + string_physio + " data available for this functional run")
    msg.setWindowTitle("Information about unavailable physiological measure")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


def showdialog_peakdetect():
    '''
    messagebox used to inform user about potential problems with the applied pulse peak detection approach
    '''
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText("There have been outliers detected in the calculated heart rate. \n\nPlease check the identified pulse peaks in the cardiac signal of the next figure to rule out a sub-optimal peak-detection.")
    msg.setWindowTitle("Check identified pulse peaks")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


def userinput_peakdetect():
    '''
    messagebox to ask user for a potential switch to an alternative peak detection method
    '''   
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setText("Would you like to use an alternative peak detection approach?")
    msg.setWindowTitle("Change of Peak Detection Approach")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    retval = msg.exec()
    if retval == QMessageBox.Yes:
        new = True
    else:
        new = False
    return(new)


def showdialog_triggererr():
    '''
    show an error message to the user if not for every functional volume in the fmr
    a trigger has been saved
    '''
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("There is not for every recorded volume a trigger saved in the Physio TSV file!")
    msg.setWindowTitle("Recorded Scan Trigger Error")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()


# =============================================================================
# Define some simple functions for the script
# =============================================================================

def shift_preds(a, b, hz):
    '''
    Shifting predictors in steps of seconds
    
    Parameters:
        a: arrary_like
               Array containing the predictor to be shifted
        b: arry_like
               shifts in the form of an array of int32 containing the temporal shifts in seconds
               e.g. [-5 0 5 10 15]
        sampling: int
                sampling rate of the signal in Hz
    '''
    a_shiftedfuncs = np.zeros((len(b), len(a)))
    for i in range(len(b)):
        shift = int(np.round(abs(b[i]*hz)))
        if b[i] < 0:
            a_shiftedfuncs[i, 0:-shift] = a[shift::]
            a_shiftedfuncs[i, -shift::] = np.mean(a)
        elif b[i] > 0:
            a_shiftedfuncs[i, 0:shift] = np.mean(a)
            a_shiftedfuncs[i, shift::] = a[0:-shift]
        else:
            a_shiftedfuncs[i, :] = a
    a_final = np.transpose(a_shiftedfuncs)
    return(a_final)

# copied from https://stackoverflow.com/a/46940319
# define hampel filter
def hampel(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on
    either side of value)
    '''
    # Make copy so original not edited
    vals = vals_orig.copy()
    # Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(k).median()
    difference = np.abs(rolling_median-vals)
    median_abs_deviation = difference.rolling(k).median()
    threshold = t0 * L * median_abs_deviation
    outlier_idx = difference > threshold
    vals[outlier_idx] = np.nan
    return(vals)



# =============================================================================
# Create Dictionary for Input and Output Parameters
# =============================================================================

# Input Parameters
physio_input_parameters = {
    "CurrentTime": datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
    "Scriptname": __name__,
    "Scriptversion": __version__,
    "Scriptdate": __date__,
    "ScriptParameters": {
        "CardiacBandpassFilterCutOff": str(cardiac_low) + "-" + str(cardiac_high),
        "RetroicorCardiacTerm": order_cardiac,
        "RetroicorRespiratoryTerm": order_resp,
        "RetroicorInteractionTerm": order_cardresp,
        "HeartRateOutlierWindowSec": hr_filloutliers_window,
        "HeartRateOutlierThresholdMAD": hr_filloutliers_threshold,
        "PulsePeakDetectionBased": [],
        "MinimumHeartBeatIntervalSec": min_hbi,
        "HeartBeatInterval%OutlierThreshold": outlier_cardiac_threshold,
        "HeartBeatInterval%OutlierWindowSize": nIBIs,
        "ShiftsOfHeartRateRegressorSec": list(hr_shifts),
        "RespiratoryOutlierWindowSec": resp_filloutliers_window,
        "RespiratoryOutlierThresholdMAD": resp_filloutliers_threshold,
        "ShiftsOfRVTRegressorSec": list(rvt_shifts),
        "ShiftsOfRVRegressorSec": list(rv_shifts),
        "ShiftsOfENVRegressorSec": list(env_shifts),
        },
    "PhysioJsonFile": {
        "Name": [],
        "SamplingFrequency": [],
        "StartTime": [],
        "PhysioColumnHeaders": [],
        "PhysioData": [],
        "NumberOfStartScanTriggersSaved": [],
        "IndicesScanTriggers": [],
        "IndicesVolumeTriggers": [],
        "PhysioTimeSec": []
        }
    }

# Dictionary for Output Parameters
physio_output_parameters = {
    "CurrentTime": datetime.now().strftime("%d-%m-%Y, %H:%M:%S"),
    "Scriptname": __name__,
    "Scriptversion": __version__,
    "Scriptdate": __date__,
    "NoiseRegressors": []}


# =============================================================================
# Read in all neccessary files
# =============================================================================

# LOAD PHYSIO DATA FROM BIDS-COMPATIBLE FILES

if physio_1file:
    physio_loop = 1
else:
    physio_loop = 2
    

for pl in range(physio_loop):

    # read json
    # physio_json_name = brainvoyager.choose_file(
    #     'Select the JSON File(s) of the Physiorecordings of a Single Run',
    #     '*.json'
    #     )
    
    if not 'physio_json_name' in locals() or pl == 1: 
        physio_json_name, _ = QFileDialog.getOpenFileName(None, 'Select Physio JSON File', os.getcwd(), 'JSON Files (*.json)')
    physio_input_parameters['PhysioJsonFile']['Name'].append(physio_json_name)
    with open(physio_json_name) as _json_file:
        _temp = json.load(_json_file)
        physio_input_parameters['PhysioJsonFile']['SamplingFrequency'].append(_temp['SamplingFrequency'])
        physio_input_parameters['PhysioJsonFile']['StartTime'].append(_temp['StartTime'])
        physio_input_parameters['PhysioJsonFile']['PhysioColumnHeaders'].append (_temp['Columns'])
        del [_temp, _json_file]
    
    # Save the index of the pulse and respiratory data within the dictionary
    test_card = [i for i, s in enumerate(physio_input_parameters['PhysioJsonFile']['PhysioColumnHeaders'][pl]) if 'card' in s.lower()]
    if np.size(test_card):
        pulse_col_dict = [pl, test_card[0]]
    test_resp = [i for i, s in enumerate(physio_input_parameters['PhysioJsonFile']['PhysioColumnHeaders'][pl]) if 'resp' in s.lower()]
    if np.size(test_resp):
        resp_col_dict = [pl, test_resp[0]]
    del (test_card, test_resp)

    # read tsv
    physio_tsv_name = physio_json_name.rsplit('.', 1)[0] + '.tsv.gz'
    temp_data = np.genfromtxt(fname=physio_tsv_name, delimiter='\t')
    physio_input_parameters['PhysioJsonFile']['PhysioData'].append(temp_data)

    _temp = np.diff(np.append(0, temp_data[:, -1]), n=1)
    physio_input_parameters['PhysioJsonFile']['NumberOfStartScanTriggersSaved'].append(int((_temp == 1).sum()))
    # find MRI trigger locations in Physio Data
    physio_input_parameters['PhysioJsonFile']["IndicesScanTriggers"].append(np.squeeze(np.nonzero(temp_data[:, -1])))
    # sometimes (e.g for the CMRR Sequence) there are 1s written for the entire
    # time a slice acquisition is on and not just for the start of a slice/volume,
    # by finding all indices in the last physio_data column where the data
    # trace changes from 0 to 1, it is possible identify just the beginning of the
    # slice/volume acquisition
    physio_input_parameters['PhysioJsonFile']["IndicesVolumeTriggers"].append(np.squeeze(np.nonzero(
        np.diff(
            np.insert(temp_data[:, -1], 0, 0)
            )
        == 1)))
    del _temp


    # save time stamps of the physio data in seconds 
    # with 0 being the start of the functional run
    physio_input_parameters['PhysioJsonFile']["PhysioTimeSec"].append(np.linspace(
        list(physio_input_parameters['PhysioJsonFile']['StartTime'])[-1],
        list(physio_input_parameters['PhysioJsonFile']['StartTime'])[-1] + (1/list(physio_input_parameters['PhysioJsonFile']['SamplingFrequency'])[-1]) * (temp_data.shape[0]-1),
        temp_data.shape[0]
        ))
    del temp_data

# LOAD FMR TO EXTRACT AND COMPARE TIMING INFORMATION

# fmr_file_name = brainvoyager.choose_file(
#     'Select the correspding FMR file', '*.fmr'
#     )
if not 'fmr_file_name' in locals():
    fmr_file_name, _ = QFileDialog.getOpenFileName(None, 'Select the Corresponding FMR File', physio_json_name.rsplit('/', 1)[0], 'FMR Files (*.fmr)')
with open(fmr_file_name) as _file:
    _fmr = _file.readlines()
    _fmr = [line.strip().replace(' ', '') for line in _fmr]

fmr_name = ''.join([line for line in _fmr if "Prefix:" in line]).split(':')[1]
fmr_name = fmr_name.replace('"', '')
fmr_time_repeat = float(
    ''.join([line for line in _fmr if 'TR:' in line]).split(':')[1]
    )
fmr_no_volumes = int(
    ''.join([line for line in _fmr if 'NrOfVolumes:' in line]).split(':')[1]
    )
fmr_no_slices = int(
    ''.join([line for line in _fmr if 'NrOfSlices:' in line]).split(':')[1]
    )
fmr_sli_table_size = int(
    ''.join([line for line in _fmr if 'SliceTimingTableSize:' in line]
            ).split(':')[1])

_index = [n for n, line in enumerate(_fmr) if 'SliceTimingTableSize:' in line]
if int(_fmr[_index[0]].split(':')[-1]) == 0:
    fmr_slice_times = np.zeros(fmr_no_slices)
    _temp = []
else:
    _temp = np.array(_fmr[_index[0]+1: _index[0]+1+fmr_no_slices])
    fmr_slice_table = _temp.astype(float)
    fmr_slice_times = np.unique(fmr_slice_table)

del [_temp, _index, _file,  _fmr]

physio_input_parameters["ScanningParameters"] = {
    'FmrName': fmr_file_name, 'NoSlices': fmr_no_slices, 
    'NoVolumes': fmr_no_volumes, 'RepetitionTimeSec': fmr_time_repeat/1000, 
    'MBfactor': int(fmr_no_slices/len(fmr_slice_times)),
    'UniqueSliceAcquisitionTimesSec': list(fmr_slice_times/1000)
    }

# LAOD TASK DESIGN MATRIX (SDM)

# TASK SDM: in case you would like to correlate the resulting physiological
# measures with your task predictors, select the task SDM file
# sdm_task_file_name = brainvoyager.choose_file(
#     "Select the SDM file of your task design if you would like to
#     correlate your predictors with the physiological measures,
#     if not click Cancel", "*.sdm"
#     )
if not 'sdm_task_file_name' in locals():
    sdm_task_file_name, _ = QFileDialog.getOpenFileName(None, 'Select the Corresponding Task Design', physio_json_name.rsplit('/', 1)[0], 'SDM Files (*.sdm)')
if sdm_task_file_name.endswith('.sdm'):
    with open(sdm_task_file_name) as _file:
        lines = _file.readlines()

        sdm_task_no_predictors = int(
            ''.join([line for line in lines if "NrOfPredictors:" in line]
                    ).split(":")[1])
        sdm_task_no_volumes = int(
            ''.join([line for line in lines if "NrOfDataPoints:" in line]
                    ).split(":")[1])
        sdm_task_incl_constant = bool(
            ''.join([line for line in lines if "IncludesConstant:" in line]
                    ).split(":")[1])
        sdm_task_first_confound = int(
            ''.join([line for line in lines if "FirstConfoundPredictor:" in line]
                    ).split(":")[1])

        for counter, line in enumerate(lines):
            if line.startswith('FirstConfoundPredictor:'):
                break
        sdm_task_colours = lines[counter+2].strip().split('   ')
        sdm_task_name = lines[counter+3].strip().strip('"').split('" "')
        t_data = []
        for line in lines[counter + 4:]:
            t_data.append([float(s) for s in line.split()[:]])
        sdm_task_data = np.array(t_data)

    del (counter, t_data)
    
    physio_input_parameters['TaskDesign'] = {
        'SdmName': sdm_task_file_name, 'NumberPredictors': sdm_task_no_predictors,
        'NumberVolumes': sdm_task_no_volumes, 'PredictorNames': sdm_task_name}



# LOAD Stimulation Protocol (PRT)

# PRT: Extract the mean heart- and respiratory rate per
# condition and per event in the physio structure
# prt_task_file_name = brainvoyager.choose_file(
#     "Select the stimulation protocol if you would like to
#     compute the mean heart- and/or respiratory rate per condition,
#     if not click Cancel", "*.prt"
#     )
if not 'prt_task_file_name' in locals():
    prt_task_file_name, _ = QFileDialog.getOpenFileName(None, 'Select the Corresponding Stimulation Protocol', physio_json_name.rsplit('/', 1)[0], 'PRT Files (*.prt)')
if prt_task_file_name.endswith('.prt'):
    # load PRT
    protocol = prt.StimulationProtocol(load=prt_task_file_name)
    # save some important parameters to the input dictionary
    physio_input_parameters['StimulationProtocol'] = {
        'PrtName': prt_task_file_name, 'NumberConditions': len(protocol.conditions),
        'ResolutionOfTime': protocol.time_units, 'ConditionNames': list(set(protocol.event_names))}
    # save also some of these parameters for reference to the output dict    
    physio_output_parameters['StimulationProtocol'] = {
       "PrtName": prt_task_file_name,
       "ConditionNames": [protocol.conditions[cond].name for cond in range(len(protocol.conditions))]}
   
    # if PRT resolution = Volumes, convert condition onsets and offsets to
    # extract the heart- and/or breathing rates in these intervals
    if protocol.time_units == "Volumes":
        protocol.convert_to_msec(fmr_time_repeat)
    

# =============================================================================
# Define Output Variables
# =============================================================================

# create output Physio folder in subfolder of FMR (if it does not exist)
physio_out_path = fmr_file_name.rsplit("/", 1)[0] + '/PhysioOut'
if not os.path.isdir(physio_out_path):
    os.mkdir(physio_out_path)

# define a counter to keep track of the number of noise models created
noise_model_no = 0
physio_regressors_matrix = np.zeros((fmr_no_volumes, 1))
physio_regressors_names = []


# =============================================================================
# Start the Processing
# =============================================================================


# =============================================================================
# Pulse Data
# =============================================================================

physio_output_parameters['CardiacOutput'] = {}
# if there is no pulse data provide feedback to the user
if 'pulse_col_dict' not in locals():
    physio_output_parameters['CardiacOutput'].update({"Error": "No cardiac data provided for this functional run"})
    if len(sys.argv) == 1:
        showdialog_info('cardiac')
    else:
        print("No cardiac data provided for this functional run\n")
else:
    # reorganize data for easier use
    physio_hz = physio_input_parameters['PhysioJsonFile']['SamplingFrequency'][pulse_col_dict[0]]
    physio_nyq = physio_hz/2
    physio_data = physio_input_parameters['PhysioJsonFile']['PhysioData'][pulse_col_dict[0]]
    physio_triggers_sum = physio_input_parameters['PhysioJsonFile']['NumberOfStartScanTriggersSaved'][pulse_col_dict[0]]
    physio_triggers_ind = physio_input_parameters['PhysioJsonFile']['IndicesScanTriggers'][pulse_col_dict[0]]
    physio_triggers_startslice_ind = physio_input_parameters['PhysioJsonFile']['IndicesVolumeTriggers'][pulse_col_dict[0]]
    physio_time = physio_input_parameters['PhysioJsonFile']['PhysioTimeSec'][pulse_col_dict[0]]
    
    physio_hz_10 = 10
    physio_ts_10 = 1/physio_hz_10  # sampling steps for 10Hz signal
    # time vector for 10Hz
    physio_time_10 = np.arange(physio_time[0], physio_time[-1], physio_ts_10)
    # extract z-transformed cardiac signal
    cardiac = stats.zscore(np.squeeze(physio_data[:, pulse_col_dict[1]]))

    # check whether there are any missing values in the cardiac signal
    assert ~np.sum(np.isnan(cardiac)), 'Nan values in the cardiac signal'


# __________________________________________________________________________
# 4.1. SYSTOLIC PEAK DETECTION based on:
#  Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D. Systolic
#  peak detection in acceleration photoplethysmograms measured from
#  emergency responders in tropical conditions. PLoS ONE. 2013;8(10):76585.
#  doi: 10.1371/journal.pone.0076585
#  Three-Stage method to get indices of systolic peaks:
#  1. Preprocessing (bandpass filtering and squaring)
#  2. Feature extraction (generating potential blocks using 2 moving averages)
#  3. Classification (thresholding)

    physio_input_parameters["ScriptParameters"]["PulsePeakDetectionBased"] = "Elgendi M, Norton I, Brearley M, Abbott D, Schuurmans D. Systolic peak detection in acceleration photoplethysmograms measured from emergency responders in tropical conditions. PLoS ONE. 2013;8(10):76585."

    # 1. PREPROCESSING: zero-phase second-order Butterworth filter
    # removing baseline wander and high frequencies not
    # contributing to systolic peaks

    # order = 2, normalized cut-off frequency between 0 & 1 (1 = nyquist freq)
    b, a = signal.butter(
        2, [cardiac_low/physio_nyq, cardiac_high/physio_nyq], btype='band')
    # changed method from pad (in Matlab) to gust to improve the filtering at the end of the signal
    cardiac_filt = signal.filtfilt(b, a, cardiac, method='gust') # method='pad' is also possible
    del b, a
    cardiac_filt_zscore = stats.zscore(cardiac_filt)

    cardiac_filt[cardiac_filt < 0] = 0  # clip to ouput signal > 0
    # squaring signal emphasizing large differences from systolic wave and
    # suppressing small differences from diastolic wave and noise
    cardiac_filt = cardiac_filt**2

    # 2. FEATURE EXTRACTION: Blocks of interest are generated using
    # two moving averages marking systolic and heartbeat areas
    w1 = 0.111  # sec (window size of one systolic peak duration)
    w1_norm = round(w1/(1/physio_hz))  # n data points in physio for w1
    w2 = 0.667  # in sec (window size of one beat duration)
    w2_norm = round(w2/(1/physio_hz))  # n data points in physio vector for w2
    # 1st moving average - emphasizing the systolic peak area in the signal
    ma_peak = uniform_filter1d(cardiac_filt, w1_norm, mode='reflect')
    # 2nd moving average - emphasizing the beat area to be used as a threshold
    # for the first moving average
    ma_beat = uniform_filter1d(cardiac_filt, w2_norm, mode='reflect')
    # plt.plot(ma_beat2)

    # 3. TRHESHOLDING: equation determining offset level 'beta' is based
    # on a brute force search
    beta = 0.02  # offset level
    # thresholding, MAbeat + a small offset
    thr1 = ma_beat + beta*np.mean(cardiac_filt)
    # generate block variable with indices that contain possible peaks in
    # cardiac_filt (1 = true), (0 = false)
    blocks = ma_peak > thr1
    blocks = blocks.astype(int)
    # get onsets, offset and durations of blocks of interest
    onset = np.flatnonzero(np.diff(blocks) == 1) + 1
    offset = np.flatnonzero(np.diff(blocks) == -1)

    # if the onset of the first peak was not recorded
    if onset[1] > offset[1]:
        onset = np.insert(onset, 0, 1)
    # if the offset of the last peak was not recorded
    if onset[-1] > offset[-1]:
        offset = np.append(offset, len(cardiac))

    duration = offset - onset+1

    # get the indices of the pulse peaks
    peaks_ind = []
    counter = 0
    for elem in duration:
        if elem >= w1_norm:
            ind = np.argmax(cardiac_filt_zscore[onset[counter]: offset[counter]])
            ind = ind + onset[counter]
            peaks_ind = np.append(peaks_ind, ind)
            del ind
        counter = counter + 1
    peaks_ind = peaks_ind.astype(int)
    # save indices (within the data vector) of identified peaks
    physio_ppg_peaks_ind = peaks_ind
    #  save timings of the peaks in seconds
    physio_ppg_peaks_time = physio_time[peaks_ind]
    # get peaks within the functional scan
    peaks_ind_run = peaks_ind[
        (peaks_ind >= physio_triggers_ind[0])
        &
        (peaks_ind <= physio_triggers_ind[-1])]

    del (w1, w1_norm, w2, w2_norm, beta, blocks, thr1, onset, offset, duration,
         ma_beat, ma_peak, counter, elem)

# ____________________________________________________________________________
# 4.2. HEART RATE and HEART RATE VARIABILITY to cross-check and possibly
#  choose different peak-detection

    hr = 60/np.diff(physio_ppg_peaks_time)
    # physio_time is sampled in middle of two consecutive peaks
    hr_time = np.diff(physio_ppg_peaks_time) / 2 + physio_ppg_peaks_time[0:-1]
    hr_time = np.insert(hr_time, 0, physio_time[0])
    hr_time = np.append(hr_time,  physio_time[-1])

    f = interpolate.interp1d(hr_time, np.block([hr[0], hr, hr[-1]]))
    # interpolate heart rate values to original sampling time
    hr_raw_fs = f(physio_time)

    # perform outlier correction similar to this function in Matlab:
    # HR_filloutl_Fs = filloutliers(HR_raw_Fs,'linear','movmedian',
    #  round(hr_filloutliers_window*physio_Hz),'ThresholdFactor',
    #  hr_filloutliers_threshold)
    #  here performed in two steps:
    df = pd.DataFrame({'hr_raw_fs': hr_raw_fs})
    # 1. outlier detection: outliers are defined as elements more than 3
    #    MAD from the median. The scaled MAD is defined
    #    as c*median(abs(A-median(A))), (see Matlab)
    #    c = -1/(2**0.5*special.erfcinv(3/2)) = 1.4826

    # apply hampel filter
    df['hr_raw_fs_outlier'] = hampel(df['hr_raw_fs'],
                                     k=int(hr_filloutliers_window * physio_hz),
                                     t0=hr_filloutliers_threshold)

    # 2. filling outliers using linear interpolation
    df['hr_raw_fs_filloutlier'] = df['hr_raw_fs_outlier'].interpolate(
        method='linear', limit_direction="both")
    hr_raw_fs_filloutlier = df['hr_raw_fs_filloutlier'].to_numpy()

    del df
    
    # compute heartbeat interval (hbi) in seconds
    physio_hbi = np.diff(physio_ppg_peaks_time)
    
    # compute Heart Rate Varibility (measure of the autonomic nervous activity) as RMSDD (root mean square of successive differences between hearbeats in ms)
    # e.g. van den Berg, M. E., Rijnbeek, P. R., Niemeijer, M. N., Hofman, A., Herpen, G. van, Bots, M. L., Hillege, H., Swenne, C. A., Eijgelsheim, M., Stricker, 
    # B. H., & Kors, J. A. (2018). Normal values of corrected heart-rate variability in 10-second electrocardiograms for all ages. Frontiers in Physiology, 9(APR), 424. 
    # https://doi.org/10.3389/FPHYS.2018.00424/BIBTEX
    physio_hbi_rmssd = np.sqrt(
        np.mean((physio_hbi[1::] - physio_hbi[0:-1])**2)) * 1000
    
    # identify possible outliers in identified heart beats (only used to decide between peak detection approaches)
    df = pd.DataFrame({'physio_hbi': physio_hbi})
    # apply rolling mean
    df['mean_physio_hbi'] = df['physio_hbi'].rolling(window=nIBIs, min_periods=1, center=True).mean()
    mean_physio_hbi = df['mean_physio_hbi'].to_numpy()
    hbi_outlier = np.logical_or((physio_hbi > mean_physio_hbi + mean_physio_hbi/100*outlier_cardiac_threshold), (physio_hbi < mean_physio_hbi - mean_physio_hbi/100*outlier_cardiac_threshold))
    del (df, mean_physio_hbi)
    
# PLOT HR and cardiac signal
    fig_cardiac = plt.figure('Cardiac', constrained_layout=True)

    ax_card_raw = fig_cardiac.add_subplot(211)
    plot_raw, = ax_card_raw.plot(
        physio_time, cardiac, zorder=1)
    plot_filt, = ax_card_raw.plot(physio_time, cardiac_filt_zscore,
                                  color='cyan', zorder=4)
    plot_startscan, = ax_card_raw.plot([physio_time[physio_triggers_ind[0]], physio_time[physio_triggers_ind[0]]], np.squeeze([
        ax_card_raw.get_ylim()]), color='red', zorder=2)
    plot_endscan, = ax_card_raw.plot([physio_time[physio_triggers_ind[-1]], physio_time[physio_triggers_ind[-1]]],
                                     np.squeeze([ax_card_raw.get_ylim()]), color='red', zorder=3)
       
    plot_peaks, = ax_card_raw.plot(
        physio_ppg_peaks_time, cardiac_filt_zscore[peaks_ind], 'g.', zorder=5)
    ax_card_raw.set_title('Cardiac raw signal')
    ax_card_raw.set_xlabel('Time (s)')
    ax_card_raw.set_ylabel('Normalized amplitude')
    ax_card_raw.legend([plot_startscan, plot_raw, plot_filt, plot_peaks], ['run length', 'cardiac raw data', 'filtered cardiac data', 'peaks'],
                       ncol=4)

    ax_card_hr = fig_cardiac.add_subplot(212)
    plot_hr, = ax_card_hr.plot(physio_time, hr_raw_fs)
    ax_card_hr.set_xlabel('Time (s)')
    ax_card_hr.set_ylabel('Heart beats per minutes')

    a = int(np.round(np.mean(hr_raw_fs_filloutlier)))
    b = int(np.round(np.std(hr_raw_fs_filloutlier)))
    c = int(np.round(physio_hbi_rmssd))
    ax_card_hr.set_title(
        'Heart rate ({0} +/- {1} bpm), heart rate variability as RMSSD ({2} ms)'.format(a, b, c))
    
    # if outliers have been detected in the identified inter-beat-intervals (heart rate), let user decide to switch the peak
    # detection method
    if (hbi_outlier.sum() == 0 and len(sys.argv) == 1) or (hbi_outlier.sum() == 0 and len(sys.argv) > 1 and plotdisp.lower() == 'true' ):
        plt.show()
    elif (hbi_outlier.sum() > 0 and len(sys.argv) == 1) or (hbi_outlier.sum() > 0 and len(sys.argv) > 1 and plotdisp.lower() == 'true' ):
        showdialog_peakdetect()
        plt.show()
        new = userinput_peakdetect()
        
        if new:
            physio_input_parameters["ScriptParameters"]["PulsePeakDetectionBased"] = "Kassinopoulos, M., & Mitsis, G. D. (2019). Identification of physiological response functions to correct for fluctuations in resting-state fMRI related to heart rate and respiration. NeuroImage, 202, 116150. https://doi.org/10.1016/j.neuroimage.2019.116150"
            # peak detection based on Kassinopoulos et al., 2019
            min_peak = int(np.round(min_hbi*physio_hz))  # minimum number of datapoints in between heartbeats
            peaks_ind, _ = signal.find_peaks(cardiac_filt_zscore, distance=min_peak, prominence=0.4)
            # get peaks within the functional scan
            peaks_ind_run = peaks_ind[
                 (peaks_ind >= physio_triggers_ind[0])
                 &
                 (peaks_ind <= physio_triggers_ind[-1])]
            physio_ppg_peaks_ind = peaks_ind
            physio_ppg_peaks_time = physio_time[peaks_ind]
            hr = 60/np.diff(physio_ppg_peaks_time)
            # physio_time is sampled in middle of two consecutive peaks
            hr_time = np.diff(physio_ppg_peaks_time) / 2 + physio_ppg_peaks_time[0:-1]
            hr_time = np.insert(hr_time, 0, physio_time[0])
            hr_time = np.append(hr_time,  physio_time[-1])

            f = interpolate.interp1d(hr_time, np.block([hr[0], hr, hr[-1]]))
            # interpolate heart rate values to original sampling time
            hr_raw_fs = f(physio_time)
            df = pd.DataFrame({'hr_raw_fs': hr_raw_fs})
             
            # apply hampel filter
            df['hr_raw_fs_outlier'] = hampel(df['hr_raw_fs'],
                                              k=int(hr_filloutliers_window * physio_hz),
                                              t0=hr_filloutliers_threshold)

            # 2. filling outliers using linear interpolation
            df['hr_raw_fs_filloutlier'] = df['hr_raw_fs_outlier'].interpolate(
                 method='linear', limit_direction="both")
            hr_raw_fs_filloutlier = df['hr_raw_fs_filloutlier'].to_numpy()

            del df
             
            # delete old heartbeat interval and its corresponding measures and compute new heartbeat interval (hbi) in seconds
            del (physio_hbi, physio_hbi_rmssd, hbi_outlier)
            physio_hbi = np.diff(physio_ppg_peaks_time)
            # calculate RMSSD based on all identified heartbeats
            physio_hbi_rmssd = np.sqrt(
                 np.mean((physio_hbi[1::] - physio_hbi[0:-1])**2)) * 1000
            
            # identify possible outliers in newly computed heartbeat signal
            df = pd.DataFrame({'physio_hbi': physio_hbi})
            # apply rolling mean
            df['mean_physio_hbi'] = df['physio_hbi'].rolling(window=nIBIs, min_periods=1, center=True).mean()
            mean_physio_hbi = df['mean_physio_hbi'].to_numpy()
            hbi_outlier = np.logical_or((physio_hbi > mean_physio_hbi + mean_physio_hbi/100*outlier_cardiac_threshold), (physio_hbi < mean_physio_hbi - mean_physio_hbi/100*outlier_cardiac_threshold))
            del (df, mean_physio_hbi)

        del(new)
    else:
        plt.close()
# ____________________________________________________________________________
# 4.3. CARDIAC PHASE: compute based on systolic peaks
    cardiac_phase = np.zeros(len(cardiac))
    for ind in range(len(cardiac)):
        if ind < peaks_ind[0] or ind >= peaks_ind[-1]:
            cardiac_phase[ind] = float('NaN')
        else:
            prev_peak = np.argwhere(peaks_ind <= ind)[-1]
            t1 = peaks_ind[prev_peak]
            t2 = peaks_ind[prev_peak+1]
            # phase coded between 0 and 2pi (see Glover et al., 2000)
            cardiac_phase[ind] = 2*np.pi*(ind - t1)/(t2-t1)
            del (t1, t2)
    del ind
    


# ____________________________________________________________________________
# 4.4. HEART RATE computation based on Kassinopoulos et al., 2019
    f = interpolate.interp1d(physio_time, hr_raw_fs_filloutlier)
    hr_10 = f(physio_time_10)
    physio_hr_mean = np.mean(hr_raw_fs_filloutlier)
    physio_hr_std = np.std(hr_raw_fs_filloutlier)
    # generate shifted versions of the cleaned HR signal
    # (hr_raw_fs_filloutlier) in the original sampling rate
    hr_final = shift_preds(hr_raw_fs_filloutlier, hr_shifts, physio_hz)
    
# ____________________________________________________________________________
# 4.5. Plotting Cardiac Measures

    fig_cardiac_final = plt.figure('Cardiac', constrained_layout=True)

    ax_card_raw = fig_cardiac_final.add_subplot(211)
    plot_raw, = ax_card_raw.plot(
        physio_time, cardiac, zorder=1)
    plot_filt, = ax_card_raw.plot(physio_time, cardiac_filt_zscore,
                                  color='cyan', zorder=4)
    plot_startscan, = ax_card_raw.plot([physio_time[physio_triggers_ind[0]], physio_time[physio_triggers_ind[0]]], np.squeeze([
        ax_card_raw.get_ylim()]), color='red', zorder=2)
    plot_endscan, = ax_card_raw.plot([physio_time[physio_triggers_ind[-1]], physio_time[physio_triggers_ind[-1]]],
                                     np.squeeze([ax_card_raw.get_ylim()]), color='red', zorder=3)
    plot_peaks, = ax_card_raw.plot(
        physio_time[peaks_ind], cardiac_filt_zscore[peaks_ind], 'r.', zorder=5)
    plot_peaks_run, = ax_card_raw.plot(
        physio_time[peaks_ind_run], cardiac_filt_zscore[peaks_ind_run], 'g.', zorder=6)
    y = ax_card_raw.get_ylim()
    ax_card_raw.set_ylim(y[0], y[1]+2)
    ax_card_raw.set_title('Cardiac signal')
    ax_card_raw.set_xlabel('Time (s)')
    ax_card_raw.set_ylabel('Normalized amplitude')
    ax_card_raw.legend([plot_startscan, plot_raw, plot_filt, plot_peaks, plot_peaks_run], ['run length', 'cardiac raw data', 'filtered cardiac data', 'peaks', 'peaks in run'],
                       ncol=5)

    ax_card_hr = fig_cardiac_final.add_subplot(212)
    plot_hr, = ax_card_hr.plot(physio_time, hr_raw_fs)
    plot_hr_cor, = ax_card_hr.plot(physio_time_10, hr_10)
    y = ax_card_hr.get_ylim()
    ax_card_hr.set_ylim(y[0], y[1]+10)
    ax_card_hr.set_xlabel('Time (s)')
    ax_card_hr.set_ylabel('Heart beats per minute')

    a = int(np.round(np.mean(hr_raw_fs_filloutlier)))
    b = int(np.round(np.std(hr_raw_fs_filloutlier)))
    c = int(np.round(physio_hbi_rmssd))
    ax_card_hr.set_title(
        'Heart rate ({0} +/- {1} bpm), heart rate variability as RMSSD ({2} ms)'.format(a, b, c))
    ax_card_hr.legend([plot_hr, plot_hr_cor], ['raw heart rate', 'heart rate corrected for outliers'],
                      ncol=2)

    fig_cardiac_final.savefig((physio_out_path + '/' + fmr_name + '_cardiac.png'), dpi=600, format='png')
    if (len(sys.argv) == 1) or (len(sys.argv) > 1 and plotdisp.lower() == 'true'):
        plt.show()
    
    del(a, b, c, y)


# ____________________________________________________________________________
# 4.6. HR*CRF: Apply standard PRF model (cardiac response function, CRF)
#  as defined by: Chang, C., Cunningham, J. P., & Glover, G. H. (2009).
#  Influence of heart rate on the BOLD signal: the cardiac response function.
#  NeuroImage, 44(3), 857?869. https://doi.org/10.1016/j.neuroimage.2008.09.029
#  code based on Kassinopoulos et al., 2019

    # time vector for impulse response
    t_ir = np.linspace(0, 60, physio_hz_10*60+1)
    # cardiac response function as defined by Chang et al., 2009
    crf = 0.6*t_ir**2.7 *np.exp(-t_ir/1.6)-(16/(np.sqrt(2*np.pi*9)))* np.exp(-(t_ir-12)**2/18)
    crf = crf/max(crf)
    del t_ir

    # smoothing HR data with 6 seconds
    hr_10_sm = uniform_filter1d(hr_10, int(6*physio_hz_10), mode='reflect')

    # in order to avoid the convolution with the zero-padded edges of the data
    # vectors, the approach by the Tapas PhysIO-toolbox is used, using as
    # padding value the mean of the HR
    temp = np.mean(hr_10_sm)
    hr_pad = np.concatenate((temp * np.ones(len(crf)-1), hr_10_sm))

    # create HR*CRF
    hr_conv = np.convolve(hr_pad, crf, 'valid')
    del(temp, hr_pad)

# ____________________________________________________________________________
# 4.7 VOLUME-BASED REGRESSORS: sample volume-based values of all
#  calculated cardiac measures and save as SDM files   

# Sample volume-based cardiac phase, hr, hr*crf

    # get trigger indices for 10 Hz signal
    time_trigger = physio_time[physio_triggers_startslice_ind]
    # get matrix with diff values of size time_triggers x physio_time_10 
    temp = np.absolute(time_trigger - physio_time_10[:, np.newaxis])
    # indices of volume values for 10 Hz signal
    trig_ind_time10 = np.argmin(temp, axis=0)
    del(temp, time_trigger)
    
    temp = int(physio_triggers_sum/fmr_no_volumes)
    
    # not for every recorded volume a trigger saved in the TSV file
    if temp < 1:
        showdialog_triggererr()
    # only volume triggers saved in physio_data
    elif temp == 1:
        # filtered and z-scored cardiac values, sampled at volume times
        cardiac_filt_vol = cardiac_filt_zscore[physio_triggers_startslice_ind]
        # cardiac phase values, sampled at volume times
        cardiac_phase_vol = cardiac_phase[physio_triggers_startslice_ind]
        # heart rate values, sampled at volume times
        hr_vol = hr_final[physio_triggers_startslice_ind]
        # heart rate values convolved with the cardiac response function,
        # sampled at volume times
        hr_conv_vol = hr_conv[trig_ind_time10]
    # slice triggers saved in physio_data:
    elif temp > 1:
        # sampled at the start of each volume
        cardiac_filt_vol = cardiac_filt_zscore[physio_triggers_startslice_ind[0::temp]]
        cardiac_phase_vol = cardiac_phase[physio_triggers_startslice_ind[0::temp]]
        # cardiac_phase_vol = cardiac_phase[physio_triggers_startslice_ind[round(temp/2)-1::temp]]  # sample the middle of the volume
        hr_vol = hr_final[physio_triggers_startslice_ind[0::temp]]
        hr_conv_vol = hr_conv[trig_ind_time10[0::temp]]

    del(temp)

    # Rescale and Detrend predictors
    hr_vol = hr_vol/np.amax(hr_vol, axis=0)
    hr_conv_vol = signal.detrend(hr_conv_vol)
    hr_conv_vol = hr_conv_vol / max(hr_conv_vol)


    # Fit Xth order fourier series to estimate cardiac phase
    dm_phs_c = np.zeros((fmr_no_volumes, order_cardiac*2))
    if order_cardiac > 0:
        for i in range(order_cardiac):
            dm_phs_c[:, (i*2)] = np.cos((i+1)*cardiac_phase_vol)
            dm_phs_c[:, (i*2)+1] = np.sin((i+1)*cardiac_phase_vol)
        del(i)


# SAVE CARDIAC SDM FILES

        # Save Cardiac RETROICOR Predictors
        sdm_cp = sdm.DesignMatrix()
        colour_temp = np.linspace(225, 30, order_cardiac*2)
        counter = 1
        for i in range(0, order_cardiac*2, 2):
            sdm_cp.add_predictor(sdm.Predictor('Cardiac_Cos' + str(counter), dm_phs_c[:,i], colour=[int(colour_temp[i]), 0, 0]))
            sdm_cp.add_predictor(sdm.Predictor('Cardiac_Sin' + str(counter), dm_phs_c[:,i+1], colour=[int(colour_temp[i+1]), 0, 0]))
            counter = counter + 1
        del(i, counter, colour_temp)
        physio_regressors_names.extend(sdm_cp.names)
        sdm_cp.add_constant()
        sdm_cp.save(physio_out_path + '/' + fmr_name + '_cardiac_RETROICOR.sdm')
        physio_regressors_matrix = np.append(physio_regressors_matrix, dm_phs_c, axis=1)
        noise_model_no = noise_model_no + order_cardiac*2
               
        temp = np.sum(np.isnan(sdm_cp.data))
        if temp > 0:
            print('NaNs in cardiac RETROICOR predictors \n')
            physio_output_parameters['CardiacOutput'].update({
                'Error_RETROICOR': 'NaNs in cardiac RETROICOR predictors'
                })
        del(temp)

    # Save Filtered and Z-transformed Cardiac Signal
    sdm_cfilt = sdm.DesignMatrix()
    sdm_cfilt.add_predictor(sdm.Predictor((
        physio_input_parameters['PhysioJsonFile']['PhysioColumnHeaders'][pulse_col_dict[0]][pulse_col_dict[1]] + 
        '_BP' + str(cardiac_low) + '-' + str(cardiac_high) + 'Hz_zscore'), 
        cardiac_filt_vol, colour=[np.random.randint(30,225), 0, 0]))
    physio_regressors_names.extend(sdm_cfilt.names)
    sdm_cfilt.add_constant()
    sdm_cfilt.save(physio_out_path + '/' + fmr_name + '_cardiac_BP' + str(cardiac_low) + '-' + str(cardiac_high) + 'Hz_z.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, np.reshape(cardiac_filt_vol,(fmr_no_volumes,1)), axis=1)
    noise_model_no = noise_model_no + 1


    # Save Shifted HR Predictors
    sdm_hr = sdm.DesignMatrix()
    colour_temp = np.linspace(225, 30, len(hr_shifts))
    for i in range(len(hr_shifts)):
        sdm_hr.add_predictor(sdm.Predictor('HR_shift_' + str(hr_shifts[i]) + 'sec', hr_vol[:,i], colour=[int(colour_temp[i]), 0, 0]))
    del(i, colour_temp)
    physio_regressors_names.extend(sdm_hr.names)
    sdm_hr.add_constant()
    sdm_hr.save(physio_out_path + '/' + fmr_name + '_HR.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, hr_vol, axis=1)
    noise_model_no = noise_model_no + len(hr_shifts)


    # Save HR*CRF Predictor
    sdm_crf = sdm.DesignMatrix()
    sdm_crf.add_predictor(sdm.Predictor('HR*CRF', hr_conv_vol, colour=[np.random.randint(30,225), 0, 0]))
    physio_regressors_names.extend(sdm_crf.names)
    sdm_crf.add_constant()
    sdm_crf.save(physio_out_path + '/' + fmr_name + '_HRCRF.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, np.reshape(hr_conv_vol,(fmr_no_volumes,1)), axis=1)
    noise_model_no = noise_model_no + 1
    
    
    # Fill the physio_output_parameters dict
    physio_output_parameters['CardiacOutput'].update({
        "HbiRmssd": round(physio_hbi_rmssd, 2), "HeartRateAverage(BPM)": round(physio_hr_mean,2),
        "HeartRateStd(BPM)": round(physio_hr_std, 2), "PulsePeaksInRun": len(peaks_ind_run), "HbiOutliersCount": int(hbi_outlier.sum())
        })
    
    
    # If a PRT was loaded, sample the heart rate in ms resolution
    if prt_task_file_name.endswith('.prt'):
        physio_time_hr_ms = np.arange(0, physio_time[-1], 1/1000)
        f = interpolate.interp1d(physio_time, hr_raw_fs_filloutlier)
        hr_ms = f(physio_time_hr_ms)
    
    del(f, peaks_ind, peaks_ind_run, physio_hz, physio_nyq, 
        physio_data, physio_triggers_ind, physio_triggers_sum,
        physio_triggers_startslice_ind, physio_time, physio_time_10,
        physio_hz_10, physio_ts_10)


# =============================================================================
# Respiratory Data
# =============================================================================

physio_output_parameters['RespiratoryOutput'] = {}
# if there is no respiratory data provide feedback to the user
if 'resp_col_dict' not in locals():
    physio_output_parameters['RespiratoryOutput'].update({"Error": "No respiratory data provided for this functional run"})
    if len(sys.argv) == 1:
        showdialog_info('respiratory')
    else:
        print("No respiratory data provided for this functional run")
else:
    # reorganize data for easier use
    physio_hz = physio_input_parameters['PhysioJsonFile']['SamplingFrequency'][resp_col_dict[0]]
    physio_data = physio_input_parameters['PhysioJsonFile']['PhysioData'][resp_col_dict[0]]
    physio_triggers_sum = physio_input_parameters['PhysioJsonFile']['NumberOfStartScanTriggersSaved'][resp_col_dict[0]]
    physio_triggers_ind = physio_input_parameters['PhysioJsonFile']['IndicesScanTriggers'][resp_col_dict[0]]
    physio_triggers_startslice_ind = physio_input_parameters['PhysioJsonFile']['IndicesVolumeTriggers'][resp_col_dict[0]]
    physio_time = physio_input_parameters['PhysioJsonFile']['PhysioTimeSec'][resp_col_dict[0]]
    
    physio_hz_10 = 10
    physio_ts_10 = 1/physio_hz_10  # sampling steps for 10Hz signal
    # time vector for 10Hz
    physio_time_10 = np.arange(physio_time[0], physio_time[-1], physio_ts_10)
    # extract z-transformed respiratory signal
    resp = stats.zscore(np.squeeze(physio_data[:, resp_col_dict[1]]))

    # check whether there are any missing values in the respiratory signal
    assert ~np.sum(np.isnan(resp)), 'Nan values in the respiratory signal'

# __________________________________________________________________________
# 5.1. PREPROCESSING of respiratory signal

#  detrend data (adapted from Kassinopoulos et al, 2019)
    resp = signal.detrend(resp)

#  outlier replacement and filtering adpated from Power, J. D., Lynch, C. J.,
#  Dubin, M. J., Silver, B. M., Martin, A., & Jones, R. M. (2020).
#  Characteristics of respiratory measures in young adults scanned at rest,
#  including systematic changes and missed deep breaths. NeuroImage, 204,
#  116234. https://doi.org/10.1016/j.neuroimage.2019.116234

    #  outlier replacement filter to eliminate spurious spike artifacts
    df = pd.DataFrame({'resp': resp})
    # 1. apply hampel filter
    df['resp_outlier'] = hampel(
        df['resp'], k=int(resp_filloutliers_window * physio_hz),
        t0=resp_filloutliers_threshold)

    # 2. filling outliers using linear interpolation
    df['resp_filloutlier'] = df['resp_outlier'].interpolate(
         method='linear', limit_direction="both")
    resp_filloutlier = df['resp_filloutlier'].to_numpy()
    del df

    # blurring, using a 1 second window to aid peak detection
    resp_filt = signal.savgol_filter(
        resp_filloutlier, window_length=int((np.ceil(physio_hz) // 2 * 2 + 1)),
        polyorder=2, mode='interp')


# __________________________________________________________________________
# 5.2. RESPIRATORY PHASE computation

#  respiratory phase needs to be calculated taking not only the times of peak
#  inspiration into account (peak location), but also the amplitude of
#  inspiration, since the depth of breathing also influences the amount of
#  head motion

#  1. amplitude of respiratory signal from the pneumatic belt, is normalized
#     to the range (0, Rmax), i.e. find max and min amplitudes and normalize
#     amplitude
    resp_norm = (resp_filt-min(resp_filt)) / (max(resp_filt)-min(resp_filt))

#  2. Calculate the histogram from the number of occurrences of specific
#     respiratory amplitudes in bins 1:100 and the bth bin is accordingly
#     centered at bRmax/100
    resp_hist, _ = np.histogram(resp_norm, 100)

#  3. Calculate running integral of the histogram, creating an equalized
#     transfer function between the breathing amplitude and respiratory phase,
#     where end-expiration is assigned a phase of 0 and peak inspiration has
#     phase of +/-pi. While inhaling the phase spans 0 to pi and during
#     expiration the phase is negated.
    resp_transfer_func = np.insert(
        (np.cumsum(resp_hist) / np.sum(resp_hist)), 0, 0)
    kern_size = int(round(physio_hz - 1))
    # smoothed version for taking derivative
    resp_smooth = np.convolve(resp_norm, np.ones(kern_size), 'same')
    # derivative dR/dt
    resp_diff = np.append(np.diff(resp_smooth), 0)
    # for phase calculation +pi was added -> so the adapted range is from 0
    # to 2pi (as defined in the Tapas PhysIO toolbox), for plotting use
    # resp_phase - pi to get the original range defined by Glover
    indices = np.round(resp_norm * 100).astype(int)
    resp_phase = np.pi * np.take(resp_transfer_func,indices) * np.sign(resp_diff) + np.pi
    del(indices, kern_size, resp_transfer_func)

# __________________________________________________________________________
# 5.3. BREATHING RATE (BR), RESPIRATION VARIATION (RV), RESPIRATION VOLUME 
#      PER TIME (RVT), RESPIRATORY FLOW (RF), WINDOWED ENVELOPE OVER THE
#      RESPIRATORY TRACE (ENV)

# peak detection adapted from Power, J. D., Lynch, C. J., Dubin, M. J., Silver,
# B. M., Martin, A., & Jones, R. M. (2020). Characteristics of respiratory
# measures in young adults scanned at rest, including systematic changes and
# missed deep breaths. NeuroImage, 204, 116234.
# https://doi.org/10.1016/j.neuroimage.2019.116234

# RVT, BR and RF computation adapted from Kassinopoulos et al., 2019

# RV as defined by: Chang, C., & Glover, G. H. (2009). Relationship between 
# respiration, end-tidal CO2, and BOLD signals in resting-state fMRI. 
# NeuroImage, 47(4), 1381-1393. 
# https://doi.org/10.1016/j.neuroimage.2009.04.048
# RV = standard deviation of the respiratory signal over 6 seconds
# computation based on Power et al., 2020

# ENV: windowed envelope of the respiratory signal over a 10-s window
# adapted from Power et al., 2020

    # z-scoring of filtered respiratory data
    resp_filtz = stats.zscore(resp_filt)
    f = interpolate.interp1d(physio_time, resp_filtz)
    resp_10 = f(physio_time_10)

    # calculate windowed envelope of the respiratory signal over a 10-s window
    def envelope_rms(a, window_size):
        a2 = np.power(a, 2)
        window = np.ones(window_size)/float(window_size)
        return np.sqrt(np.convolve(a2, window, 'same'))
    env = envelope_rms(resp_filtz, int(physio_hz*10))
    
    # generate shifted versions of the respiration envelope
    env_final = shift_preds(env, env_shifts, physio_hz)


# RV: calculate respiration variation (RV) over 6 seconds window

    df = pd.DataFrame({'resp_filtz': resp_filtz})
    # apply rolling standard deviation 
    df['rv'] = df['resp_filtz'].rolling(window=int(physio_hz*6), min_periods=1, center=True).std()
    rv = df['rv'].to_numpy()
    del df
    
    # generate shifted versions of the respiration variation
    rv_final = shift_preds(rv, rv_shifts, physio_hz)
    

# PEAKS: find peaks and troughs in respiratory signal to calculate RVT and BR
    # minpeakdistance = 1.8 sec, presumes breaths occur more than 1.8 s apart
    resp_peaks_ind, _ = signal.find_peaks(
        resp_filtz, distance=physio_hz*1.8, prominence=0.5)
    resp_peaks = resp_filtz[resp_peaks_ind]
    resp_troughs_ind, _ = signal.find_peaks(
        -resp_filtz, distance=physio_hz*1.8, prominence=0.5)
    resp_troughs = resp_filtz[resp_troughs_ind]

    temp = (resp_peaks_ind >= physio_triggers_ind[0]) * (resp_peaks_ind <= physio_triggers_ind[-1])
    resp_peaks_ind_run = resp_peaks_ind[temp]
    del(temp)
    temp = (resp_troughs_ind >= physio_triggers_ind[0]) * (resp_troughs_ind <= physio_triggers_ind[-1])
    resp_troughs_ind_run = resp_troughs_ind[temp]
    del(temp)

    physio_resp_peaks_ind = resp_peaks_ind
    physio_resp_peaks_time = physio_time[resp_peaks_ind]

    temp_time_peaks = np.concatenate(([physio_time[0]], physio_time[resp_peaks_ind], [physio_time[-1]]))
    temp_peaks = np.concatenate(([resp_peaks[0]], resp_peaks, [resp_peaks[-1]]))
    f = interpolate.interp1d(temp_time_peaks, temp_peaks)
    resp_up_10 = f(physio_time_10)
    del(temp_time_peaks, temp_peaks, f)

    temp_time_troughs = np.concatenate(([physio_time[0]], physio_time[resp_troughs_ind], [physio_time[-1]]))
    temp_troughs = np.concatenate(([resp_troughs[0]], resp_troughs, [resp_troughs[-1]]))
    f = interpolate.interp1d(temp_time_troughs, temp_troughs)
    resp_low_10 = f(physio_time_10)
    del(temp_time_troughs, temp_troughs, f)

# BR: calculate breathing rate (BR)
    br = 60 / np.diff(physio_time[resp_peaks_ind])
    time_br = np.concatenate(([physio_time[0]], (np.diff(physio_time[resp_peaks_ind])) /2 + physio_time[resp_peaks_ind[:-1]], [physio_time[-1]]))
    f = interpolate.interp1d(time_br, np.concatenate(([br[0]], br, [br[-1]])))
    br_10 = f(physio_time_10)
    del(time_br)

    physio_br_mean = np.mean(br_10)
    physio_br_std = np.std(br_10)
    
# RVT:  calculate respiratory volume per time (RVT), i.e. change in breath
#       amplitude over one breath cycle
    rvt = (resp_up_10 - resp_low_10) * br_10

    # generate shifted versions of the cleaned RVT signal at 10 Hz
    rvt_final = shift_preds(rvt, rvt_shifts, physio_hz_10)


# RF: calculate respiratory flow (RF)
    # using a moving average window of 1.5 s to avoid spike artifacts
    # might be not even necessary as the original respiratory data has been
    # smoothed already (resp_filt)
    resp_s = uniform_filter1d(resp_10, int(1.5*physio_hz_10), mode='reflect')
    rf = np.diff(resp_s)
    rf = np.insert(rf, 0, 0)
    rf = rf**2
    del(resp_s)


# ____________________________________________________________________________
# 5.4. Plotting Respiratory Measures

    fig_respiratory = plt.figure('Respiratory', figsize=(10, 8), constrained_layout=True)

    ax_resp_hist = fig_respiratory.add_subplot(5, 1, 1)
    ax_resp_hist.hist(resp_norm, bins=100, edgecolor='b')
    ax_resp_hist.set_title('Histogram of detrended, outlier-corrected, smoothed and amplitude-normalized respiration')

    ax_resp_raw = fig_respiratory.add_subplot(5, 1, 2)
    plot_raw, = ax_resp_raw.plot(
        physio_time, resp, zorder=1)
    plot_filt, = ax_resp_raw.plot(
        physio_time, resp_filtz, color='cyan', zorder=4)
    plot_startscan, = ax_resp_raw.plot([physio_time[physio_triggers_ind[0]], physio_time[physio_triggers_ind[0]]], np.squeeze([
        ax_resp_raw.get_ylim()]), color='red', zorder=2)
    plot_endscan, = ax_resp_raw.plot([physio_time[physio_triggers_ind[-1]], physio_time[physio_triggers_ind[-1]]],
                                     np.squeeze([ax_resp_raw.get_ylim()]), color='red', zorder=3)
    plot_peaks, = ax_resp_raw.plot(
        physio_time[resp_peaks_ind], resp_filtz[resp_peaks_ind], 'r.', zorder=5)
    plot_peaks_run, = ax_resp_raw.plot(
        physio_time[resp_peaks_ind_run], resp_filtz[resp_peaks_ind_run], 'g.', zorder=7)
    plot_troughs, = ax_resp_raw.plot(
        physio_time[resp_troughs_ind], resp[resp_troughs_ind], 'r.', zorder=6)
    plot_troughs_run, = ax_resp_raw.plot(
        physio_time[resp_troughs_ind_run], resp[resp_troughs_ind_run], 'g.', zorder=8)
    ax_resp_raw.plot(physio_time_10, resp_up_10)
    ax_resp_raw.plot(physio_time_10, resp_low_10)
    y = ax_resp_raw.get_ylim()
    ax_resp_raw.set_ylim(y[0], y[1]+2)
    ax_resp_raw.set_title('Respiratory signal')
    ax_resp_raw.set_xlabel('Time (s)')
    ax_resp_raw.set_ylabel('Normalized amplitude')
    ax_resp_raw.legend([plot_startscan, plot_raw, plot_filt, plot_peaks, plot_peaks_run], ['run length', 'respiratory raw data', 'filtered respiratory data', 'peaks/troughs', 'peaks/troughs in run'], 
                       ncol=5)

    ax_resp_br = fig_respiratory.add_subplot(5, 1, 3)
    plot_br, = ax_resp_br.plot(physio_time_10, br_10)
    ax_resp_br.set_title(
        'Breathing rate (BR): {0} +/- {1} rpm'.format(np.round(physio_br_mean, decimals=1), np.round(physio_br_std, decimals=1)))
    ax_resp_br.set_xlabel('Time (s)')
    ax_resp_br.set_ylabel('Rpm')  # Respirations per Minute (rpm)

    ax_resp_rvt_env = fig_respiratory.add_subplot(5, 1, 4)
    plot_resp, = ax_resp_rvt_env.plot(physio_time, resp_filtz)
    plot_rvt, = ax_resp_rvt_env.plot(physio_time_10, stats.zscore(rvt), linewidth=1)
    plot_env, = ax_resp_rvt_env.plot(physio_time, stats.zscore(env), linewidth=1)
    plot_rv, = ax_resp_rvt_env.plot(physio_time, stats.zscore(rv), linewidth=1, color='y')
    y = ax_resp_rvt_env.get_ylim()
    ax_resp_rvt_env.set_ylim(y[0], y[1]+2)
    ax_resp_rvt_env.set_ylabel('Normalized amplitude')
    ax_resp_rvt_env.set_xlabel('time(s)')
    ax_resp_rvt_env.set_title('Respiration volume per time (RVT), windowed Envelope over respiration signal (ENV) and Respiration variation (RV)')
    ax_resp_rvt_env.legend([plot_resp, plot_env, plot_rvt, plot_rv], ['filtered respiratory data', 'ENV', 'RVT', 'RV'], 
                           ncol=4)

    ax_resp_rf = fig_respiratory.add_subplot(5, 1, 5)
    ax_resp_rf.plot(physio_time_10, rf, 'g')
    ax_resp_rf.set_title('Respiratory flow (RF)')
    ax_resp_rf.set_ylabel('RF (a.u.)')
    ax_resp_rf.set_xlabel('Time (s)')
    
    fig_respiratory.savefig((physio_out_path + '/' + fmr_name + '_resp.png'), dpi=600, format='png')
    if (len(sys.argv) == 1) or (len(sys.argv) > 1 and plotdisp.lower() == 'true'):
        plt.show()
    del(y)


# __________________________________________________________________________
# 5.5. RVT*RRF, RV*RRF: Apply standard RRF model (respiratory response function, RRF)
#      as defined by: Birn, R. M., Smith, M. A., Jones, T. B., & Bandettini, P. A. (2008). 
#      The respiration response function: The temporal dynamics of fMRI signal fluctuations 
#      related to changes in respiration. NeuroImage, 40(2), 644-654. 
#      https://doi.org/10.1016/j.neuroimage.2007.11.059
#      code based on Kassinopoulos et al., 2019

    # downsample respiration variation to 10 Hz before convolution with RRF
    f = interpolate.interp1d(physio_time, rv)
    rv_10 = f(physio_time_10)
    del(f)
    
    # time vector for impulse response (sampled at 10 Hz)
    t_ir = np.linspace(0, 60, physio_hz_10*60+1)
    # respiratory response function as defined by Birn et al., 2008
    # (sampled at 10 Hz)
    rrf = 0.6*t_ir**2.1*np.exp(-t_ir/1.6)-0.0023*t_ir**3.54*np.exp(-t_ir/4.25)
    # normalize by max = 1
    rrf = rrf/max(rrf)
    del(t_ir)
    
    # in order to avoid the convolution with the zero-padded edges of the data
    # vectors, the approach by the Tapas PhysIO-toolbox is used, using as
    # padding value the mean of the RVT and RV
    temp_rvt = np.mean(rvt)
    rvt_pad = np.concatenate((temp_rvt * np.ones(len(rrf)-1), rvt))
    temp_rv = np.mean(rv_10)
    rv_pad = np.concatenate((temp_rv * np.ones(len(rrf)-1), rv_10))

    # create RVT*RRF, RV*RRF
    rvt_conv = np.convolve(rvt_pad, rrf, 'valid')
    rv_conv = np.convolve(rv_pad, rrf, 'valid')
    del(temp_rvt, temp_rv, rvt_pad, rv_pad)
    
    

# ____________________________________________________________________________
# 5.6 VOLUME-BASED REGRESSORS: sample volume-based values of all
#  calculated respiratory measures and save as SDM files   

# Sample volume-based filtered respiratory signal, respiratory phase, 
# BR, RVT, RF, ENV, RVT*RRF

    # get trigger indices for 10 Hz signal
    time_trigger = physio_time[physio_triggers_startslice_ind]
    # get matrix with diff values of size time_triggers x physio_time_10 
    temp = np.absolute(time_trigger - physio_time_10[:, np.newaxis])
    # indices of volume values for 10 Hz signal
    trig_ind_time10 = np.argmin(temp, axis=0)
    del(temp, time_trigger)

    
    temp = int(physio_triggers_sum/fmr_no_volumes)
    
    # not for every recorded volume a trigger saved in the TSV file
    if temp < 1:
        showdialog_triggererr()
        
    # only volume triggers saved in physio_data
    elif temp == 1:
        # filtered and z-scored respiratory values, sampled at volume times
        resp_filt_vol = resp_filtz[physio_triggers_startslice_ind]
        # respiratory phase values, sampled at volume times
        resp_phase_vol = resp_phase[physio_triggers_startslice_ind]
        # breathing rate values, sampled at volume times
        br_vol = br_10[trig_ind_time10]
        # respiration volume per time values, sampled at TR
        rvt_vol = rvt_final[trig_ind_time10,:]
        # windowed envelope values of respiratory signal, sampled at TR
        env_vol = env_final[physio_triggers_startslice_ind,:]
        # respiration variation, sampled at TR
        rv_vol = rv_final[physio_triggers_startslice_ind,:]
        # respiratory flow values, sampled at TR
        rf_vol = rf[trig_ind_time10]
        # respiration volume per time values convolved with the respiratory 
        # response function, sampled at volume times (rvt*rrf)
        rvt_conv_vol = rvt_conv[trig_ind_time10]
        # respiration variation convolved with the respiratory 
        # response function, sampled at volume times (rv*rrf)
        rv_conv_vol = rv_conv[trig_ind_time10]
        
        
    # slice triggers saved in physio_data:
    elif temp > 1:
        # sampled at the start of each volume
        resp_filt_vol = resp_filtz[physio_triggers_startslice_ind[0::temp]]
        resp_phase_vol = resp_phase[physio_triggers_startslice_ind[0::temp]]
        # resp_phase_vol = resp_phase[physio_triggers_startslice_ind[round(temp/2)-1::temp]]  # sample the middle of the volume
        br_vol = br_10[trig_ind_time10[0::temp]]
        rvt_vol = rvt_final[trig_ind_time10[0::temp],:]
        env_vol = env_final[physio_triggers_startslice_ind[0::temp],:]
        rv_vol = rv_final[physio_triggers_startslice_ind[0::temp],:]
        rf_vol = rf[trig_ind_time10[0::temp]] 
        rvt_conv_vol = rvt_conv[trig_ind_time10[0::temp]]
        rv_conv_vol = rv_conv[trig_ind_time10[0::temp]]      
    del(temp)

    # Rescale and Detrend predictors
    rvt_vol = rvt_vol/np.amax(rvt_vol, axis=0)
    rv_vol = rv_vol/np.amax(rv_vol, axis=0)
    env_vol = env_vol / np.amax(env_vol, axis=0)
    br_vol = br_vol / max(br_vol)
    rf_vol = rf_vol / max(rf_vol)
    rvt_conv_vol = signal.detrend(rvt_conv_vol)
    rvt_conv_vol = rvt_conv_vol / max(rvt_conv_vol)
    rv_conv_vol = signal.detrend(rv_conv_vol)
    rv_conv_vol = rv_conv_vol / max(rv_conv_vol)


# Fit Xth order fourier series to estimate respiratory phase
    dm_phs_r = np.zeros((fmr_no_volumes, order_resp*2))
    if order_resp > 0:
        for i in range(order_resp):
            dm_phs_r[:, (i*2)] = np.cos((i+1)*resp_phase_vol)
            dm_phs_r[:, (i*2)+1] = np.sin((i+1)*resp_phase_vol)
        del(i)



# SAVE RESPIRATORY SDM FILES

        # Save Respiratory RETROICOR Predictors
        sdm_rp = sdm.DesignMatrix()
        colour_temp = np.linspace(225, 30, order_resp*2)
        counter = 1
        for i in range(0, order_resp*2, 2):
            sdm_rp.add_predictor(sdm.Predictor('Resp_Cos' + str(counter), dm_phs_r[:,i], colour=[0, 0, int(colour_temp[i])]))
            sdm_rp.add_predictor(sdm.Predictor('Resp_Sin' + str(counter), dm_phs_r[:,i+1], colour=[0, 0, int(colour_temp[i+1])]))
            counter = counter + 1
        del(i, counter, colour_temp)
        physio_regressors_names.extend(sdm_rp.names)
        sdm_rp.add_constant()
        sdm_rp.save(physio_out_path + '/' + fmr_name + '_resp_RETROICOR.sdm')
        physio_regressors_matrix = np.append(physio_regressors_matrix, dm_phs_r, axis=1)
        noise_model_no = noise_model_no + order_resp*2
               
        temp = np.sum(np.isnan(sdm_rp.data))
        if temp > 0:
            print('NaNs in respiratory RETROICOR predictors \n')
            physio_output_parameters['RespiratoryOutput'].update({
                'Error_RETROICOR': 'NaNs in repiratory RETROICOR predictors'
                })
        del(temp)

    # Save Filtered and Z-transformed Respiratory Signal
    sdm_rfilt = sdm.DesignMatrix()
    sdm_rfilt.add_predictor(sdm.Predictor((
        physio_input_parameters['PhysioJsonFile']['PhysioColumnHeaders'][resp_col_dict[0]][resp_col_dict[1]]
        + '_filt_zscore'), resp_filt_vol, colour=[0, 0, np.random.randint(30,225)]))
    physio_regressors_names.extend(sdm_rfilt.names)
    sdm_rfilt.add_constant()
    sdm_rfilt.save(physio_out_path + '/' + fmr_name + '_resp_filtz.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, np.reshape(resp_filt_vol,(fmr_no_volumes,1)), axis=1)
    noise_model_no = noise_model_no + 1

    # Save BR predictor
    sdm_br = sdm.DesignMatrix()
    sdm_br.add_predictor(sdm.Predictor('BreathingRate', br_vol, colour=[0, 0, np.random.randint(30,225)]))
    physio_regressors_names.extend(sdm_br.names)
    sdm_br.add_constant()
    sdm_br.save(physio_out_path + '/' + fmr_name + '_BR.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, np.reshape(br_vol,(fmr_no_volumes,1)), axis=1)
    noise_model_no = noise_model_no + 1

    # Save RF predictor
    sdm_rf = sdm.DesignMatrix()
    sdm_rf.add_predictor(sdm.Predictor('RespiratoryFlow', rf_vol, colour=[0, 0, np.random.randint(30,225)]))
    physio_regressors_names.extend(sdm_rf.names)
    sdm_rf.add_constant()
    sdm_rf.save(physio_out_path + '/' + fmr_name + '_RF.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, np.reshape(rf_vol,(fmr_no_volumes,1)), axis=1)
    noise_model_no = noise_model_no + 1

    # Save Shifted ENV Predictors
    sdm_env = sdm.DesignMatrix()
    colour_temp = np.linspace(225, 30, len(env_shifts))
    for i in range(len(env_shifts)):
        sdm_env.add_predictor(sdm.Predictor('ENV_shift_' + str(env_shifts[i]) + 'sec', env_vol[:,i], colour=[0, 0, int(colour_temp[i])]))
    del(i, colour_temp)
    physio_regressors_names.extend(sdm_env.names)
    sdm_env.add_constant()
    sdm_env.save(physio_out_path + '/' + fmr_name + '_ENV.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, env_vol, axis=1)
    noise_model_no = noise_model_no + len(env_shifts)
    
    # Save Shifted RV Predictors
    sdm_rv = sdm.DesignMatrix()
    colour_temp = np.linspace(225, 30, len(rv_shifts))
    for i in range(len(rv_shifts)):
        sdm_rv.add_predictor(sdm.Predictor('RV_shift_' + str(rv_shifts[i]) + 'sec', rv_vol[:,i], colour=[0, 0, int(colour_temp[i])]))
    del(i, colour_temp)
    physio_regressors_names.extend(sdm_rv.names)
    sdm_rv.add_constant()
    sdm_rv.save(physio_out_path + '/' + fmr_name + '_RV.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, rv_vol, axis=1)
    noise_model_no = noise_model_no + len(rv_shifts)

    # Save Shifted RVT Predictors
    sdm_rvt = sdm.DesignMatrix()
    colour_temp = np.linspace(225, 30, len(rvt_shifts))
    for i in range(len(rvt_shifts)):
        sdm_rvt.add_predictor(sdm.Predictor('RVT_shift_' + str(rvt_shifts[i]) + 'sec', rvt_vol[:,i], colour=[0, 0, int(colour_temp[i])]))
    del(i, colour_temp)
    physio_regressors_names.extend(sdm_rvt.names)
    sdm_rvt.add_constant()
    sdm_rvt.save(physio_out_path + '/' + fmr_name + '_RVT.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, rvt_vol, axis=1)
    noise_model_no = noise_model_no + len(rvt_shifts)


    # Save RVT*RRF Predictor
    sdm_rvtrrf = sdm.DesignMatrix()
    sdm_rvtrrf.add_predictor(sdm.Predictor('RVT*RRF', rvt_conv_vol, colour=[0, 0, np.random.randint(30,225)]))
    physio_regressors_names.extend(sdm_rvtrrf.names)
    sdm_rvtrrf.add_constant()
    sdm_rvtrrf.save(physio_out_path + '/' + fmr_name + '_RVTRRF.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, np.reshape(rvt_conv_vol,(fmr_no_volumes,1)), axis=1)
    noise_model_no = noise_model_no + 1
    
    # Save RV*RRF Predictor
    sdm_rvrrf = sdm.DesignMatrix()
    sdm_rvrrf.add_predictor(sdm.Predictor('RV*RRF', rv_conv_vol, colour=[0, 0, np.random.randint(30,225)]))
    physio_regressors_names.extend(sdm_rvrrf.names)
    sdm_rvrrf.add_constant()
    sdm_rvrrf.save(physio_out_path + '/' + fmr_name + '_RVRRF.sdm')
    physio_regressors_matrix = np.append(physio_regressors_matrix, np.reshape(rv_conv_vol,(fmr_no_volumes,1)), axis=1)
    noise_model_no = noise_model_no + 1
    
    # Fill the physio_output_parameters dict
    physio_output_parameters['RespiratoryOutput'].update({
        "RespiratoryRateAverage(BPM)": round(physio_br_mean,2),
        "RespiratoryRateStd(BPM)": round(physio_br_std, 2), "RespiratoryPeaksInRun": len(resp_peaks_ind_run)})
    
    # If a PRT was loaded, sample the respiratory rate in ms resolution
    if prt_task_file_name.endswith('.prt'):
        physio_time_br_ms = np.arange(0, physio_time_10[-1], 1/1000)
        f = interpolate.interp1d(physio_time_10, br_10)
        br_ms = f(physio_time_br_ms)

    del(physio_hz, physio_data, physio_triggers_ind, physio_triggers_sum,
        physio_triggers_startslice_ind, physio_time)

# ____________________________________________________________________________
# 6. If Both Cardiac and Respiratory Signal are available merge cardiac and 
#    respiratory RETROICOR SDMs,
#    if a multplicative term has been specified by the user, include it in the 
#    final RETROICOR SDM
#    see Harvey et al., 2008:
#    interaction between cardiac and respiratory processes, giving rise to
#    amplitude modulation of the cardiac signal by the respiratory waveform
#    (1. heart rate varies with respiration - known as respiratory sinus
#    arrhythmia, 2. venous return to the heart is facilitated during
#    inspiration - known as intrathoracic pump)
#    see also: Kasper, L., Bollmann, S., Diaconescu, A. O., Hutton, C.,
#    Heinzle, J., Iglesias, S., Stephan, K. E. (2017). The PhysIO Toolbox for
#    Modeling Physiological Noise in fMRI Data. Journal of Neuroscience
#    Methods, 276, 56-72. https://doi.org/10.1016/J.JNEUMETH.2016.10.019

#  Merge and Save Complete RETROICOR Model
    if 'pulse_col_dict' in locals() and 'resp_col_dict' in locals() and order_cardiac > 0:
        sdm_crp = copy.deepcopy(sdm_cp)
        colour_temp = np.linspace(225, 30, order_resp*2)
        counter = 1
        for i in range(0, order_resp*2, 2):
            sdm_crp.add_predictor(sdm.Predictor('Resp_Cos' + str(counter), dm_phs_r[:,i], colour=[0, 0, int(colour_temp[i])]))
            sdm_crp.add_predictor(sdm.Predictor('Resp_Sin' + str(counter), dm_phs_r[:,i+1], colour=[0, 0, int(colour_temp[i+1])]))
            counter = counter + 1
        del(i, counter, colour_temp)
        
        # add multiplicative term to RETROICOR Model if specified by the user
        # a first order interaction model results in 4 predictors
        if order_cardresp > 0:
            dm_phs_cr = np.zeros((fmr_no_volumes, order_cardresp*4))
            for i in range(order_cardresp):
                dm_phs_cr[:, (i*4)] = np.cos((i+1)*cardiac_phase_vol + (i+1)*resp_phase_vol)
                dm_phs_cr[:, (i*4)+1] = np.sin((i+1)*cardiac_phase_vol + (i+1)*resp_phase_vol)
                dm_phs_cr[:, (i*4)+2] = np.cos((i+1)*cardiac_phase_vol - (i+1)*resp_phase_vol)
                dm_phs_cr[:, (i*4)+3] = np.sin((i+1)*cardiac_phase_vol - (i+1)*resp_phase_vol)
            del(i)
            colour_temp = np.linspace(225, 30, order_cardresp*4)
            for i in range(0, order_cardresp*4, 4):
                sdm_crp.add_predictor(sdm.Predictor('CardiacResp_' + str(i+1), dm_phs_cr[:,i], colour=[0, int(colour_temp[i]), 0]))
                sdm_crp.add_predictor(sdm.Predictor('CardiacResp_' + str(i+2), dm_phs_cr[:,i+1], colour=[0, int(colour_temp[i+1]), 0]))
                sdm_crp.add_predictor(sdm.Predictor('CardiacResp_' + str(i+3), dm_phs_cr[:,i+2], colour=[0, int(colour_temp[i+2]), 0]))
                sdm_crp.add_predictor(sdm.Predictor('CardiacResp_' + str(i+4), dm_phs_cr[:,i+3], colour=[0, int(colour_temp[i+3]), 0]))
            del(i, colour_temp)            
            physio_regressors_names.extend(sdm_crp.names[-(order_cardresp*4)-1:-1])
            physio_regressors_matrix = np.append(physio_regressors_matrix, dm_phs_cr, axis=1)
            noise_model_no = noise_model_no + order_cardresp*4
        sdm_crp.save(physio_out_path + '/' + fmr_name + '_cardiacresp_RETROICOR.sdm')


physio_regressors_matrix = np.delete(physio_regressors_matrix, 0, 1)

# ____________________________________________________________________________
# 7. Correlate the resulting physiological measures with the task predictors
#    specified in the SDM file
#    if the user has specified a task design matrix, Pearson correlations
#    between all task predictors and all physio predictors will be calculated
#    and saved     

if sdm_task_file_name.endswith('.sdm'):
    # calcualte Pearson correlation coefficients and corresponding p-values
    # in physio_task_noise_corr_matrix and physio_task_noise_corrpval_matrix
    
    if sdm_task_incl_constant:
        temp = len(sdm_task_name)-1
        physio_task_noise_corr_matrix = np.zeros((len(physio_regressors_names), len(sdm_task_name)-1))
        physio_task_noise_corrpval_matrix = np.zeros((len(physio_regressors_names), len(sdm_task_name)-1))
        task_names = sdm_task_name[0:-1]
    else: 
        temp = len(sdm_task_name)
        physio_task_noise_corr_matrix = np.zeros((len(physio_regressors_names), len(sdm_task_name)))
        physio_task_noise_corrpval_matrix = np.zeros((len(physio_regressors_names), len(sdm_task_name)))
        task_names = sdm_task_name
    for n in range(len(physio_regressors_names)):
        for t in range(temp):
            # if there are NaNs in the created predictors, perform the correlation without the affected volumes
            if np.sum(np.isnan(physio_regressors_matrix[:,n])) > 0:
                ind_nan = np.argwhere(np.isnan(physio_regressors_matrix[:,n]))
                x = np.delete(physio_regressors_matrix[:,n], ind_nan)
                y = np.delete(sdm_task_data[:,t], ind_nan)
                corr_p = stats.pearsonr(x,y)
                del(x,y,ind_nan)
            else:
                corr_p = stats.pearsonr(physio_regressors_matrix[:,n], sdm_task_data[:,t])
            physio_task_noise_corr_matrix[n,t] = corr_p[0]
            physio_task_noise_corrpval_matrix[n,t] = corr_p[1]
            del(corr_p)
            
    df_corr = pd.DataFrame(physio_task_noise_corr_matrix, columns = ["correlation_" + item for item in task_names], index = physio_regressors_names)
    df_pval = pd.DataFrame(physio_task_noise_corrpval_matrix, columns = ["pval_" + item for item in task_names], index = physio_regressors_names)
    # Save resulting correlation matrix and corresponding p-values in tsv file
    df_temp = pd.concat([df_corr, df_pval], axis=1)
    df_temp.to_csv(physio_out_path + '/' + fmr_name + '_CorrelationMatrixNoiseTaskRegressors.tsv', sep="\t")
     
# Plot the the Pearson correlation coefficients in a heatmap
    fig_tasknoisecorr = plt.figure('Task-Noise-Correlation', figsize=(10, 8))

    temp = np.ma.masked_outside(physio_task_noise_corrpval_matrix, -min_pvalue, min_pvalue)
    mask_sig = np.ma.getmaskarray(temp)
    temp = np.ma.masked_inside(physio_task_noise_corrpval_matrix, -min_pvalue, min_pvalue)
    mask_nonsig = np.ma.getmaskarray(temp)
    del(temp)

    ax = sns.heatmap(
        physio_task_noise_corr_matrix,
        vmin=np.min(physio_task_noise_corr_matrix),
        vmax=np.max(physio_task_noise_corr_matrix),
        center=0,
        linewidth=0.3,
        linecolor='black',
        mask=mask_sig,
        cmap='coolwarm',
        annot=True,
        annot_kws={'size':6, 'color':'black', 'fontweight':'bold'})  # sns.diverging_palette(20, 220, n=200))

    ax2 = ax.twinx()
    sns.heatmap(
        np.round(physio_task_noise_corr_matrix, decimals=2),
        center=0,
        linewidth=0.3,
        linecolor='black',
        mask=mask_nonsig,
        cmap=mpl.colors.ListedColormap(['white']),
        annot=True,
        cbar=False,
        alpha=1,
        ax = ax2,
        xticklabels=False,
        yticklabels=False,
        annot_kws={'size':6, 'color':'black'})  # sns.diverging_palette(20, 220, n=200))
    
    ax.set_xticks(np.arange(len(task_names)), labels=task_names, fontsize = 'xx-small')
    ax.set_yticks(np.arange(len(physio_regressors_names)), labels=physio_regressors_names, fontsize = 'xx-small')
    ax.set_title("Pearson Correlation Values of Task and Noise Regressors, with p < {0} ".format(min_pvalue))
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")
    plt.setp(ax.get_yticklabels(), rotation=360, va="top")
    
    fig_tasknoisecorr.tight_layout()
    fig_tasknoisecorr.savefig((physio_out_path + '/' + fmr_name + '_TaskNoiseCorr.png'), dpi=600, format='png')
    if (len(sys.argv) == 1) or (len(sys.argv) > 1 and plotdisp.lower() == 'true'):
        plt.show()
    

# ____________________________________________________________________________
# 8. Plot and calculate the heart and respiratory rate in relation to the
#    stimulation protocol,
#    if the user has specified a stimulation protocol, the PRT will be plotted
#    and the mean heart- and/or respiratory rate per condition is saved         

if prt_task_file_name.endswith('.prt'):
        
# Compute heart rate per event and condition if cardiac data exists
   if 'pulse_col_dict' in locals():
       physio_output_parameters['StimulationProtocol']['Cardiac'] = {}
       
       for cond in range(len(protocol.conditions)):
           physio_output_parameters['StimulationProtocol']['Cardiac'].update({('Mean_HeartRate_PerEvent ' + protocol.conditions[cond].name): []})
           for event in range(np.size(protocol.conditions[cond].data,0)):
                temp = np.mean(hr_ms[(physio_time_hr_ms*1000 >= protocol.conditions[cond].data[event,0]) & (physio_time_hr_ms*1000 <= protocol.conditions[cond].data[event,1])])
                physio_output_parameters['StimulationProtocol']['Cardiac'][('Mean_HeartRate_PerEvent ' + protocol.conditions[cond].name)].append(np.round(temp,decimals=2))
                del(temp)
           temp_cond = np.mean(physio_output_parameters['StimulationProtocol']['Cardiac'][('Mean_HeartRate_PerEvent ' + protocol.conditions[cond].name)])
           physio_output_parameters['StimulationProtocol']['Cardiac'].update({('Mean_HeartRate ' + protocol.conditions[cond].name): np.round(temp_cond,decimals=2)})
           del(temp_cond)
            
# Compute breathing rate per event and condition if respiratory data exists
   if 'resp_col_dict' in locals():
       physio_output_parameters['StimulationProtocol']['Respiratory'] = {}
       
       for cond in range(len(protocol.conditions)):
           physio_output_parameters['StimulationProtocol']['Respiratory'].update({('Mean_BreathingRate_PerEvent ' + protocol.conditions[cond].name): []})
           for event in range(np.size(protocol.conditions[cond].data,0)):
                temp = np.mean(br_ms[(physio_time_br_ms*1000 >= protocol.conditions[cond].data[event,0]) & (physio_time_br_ms*1000 <= protocol.conditions[cond].data[event,1])])
                physio_output_parameters['StimulationProtocol']['Respiratory'][('Mean_BreathingRate_PerEvent ' + protocol.conditions[cond].name)].append(np.round(temp,decimals=2))
                del(temp)
           temp_cond = np.mean(physio_output_parameters['StimulationProtocol']['Respiratory'][('Mean_BreathingRate_PerEvent ' + protocol.conditions[cond].name)])
           physio_output_parameters['StimulationProtocol']['Respiratory'].update({('Mean_BreathingRate ' + protocol.conditions[cond].name): np.round(temp_cond,decimals=2)})
           del(temp_cond)
                

### PLOTTING PRT
   fig_prt = plt.figure('Stimulation-Protocol', figsize=(10, 8))
   ax_prt = fig_prt.add_subplot(111)
   # set the limits to the scan duration
   ax_prt.set_xlim(0, fmr_no_volumes*fmr_time_repeat/1000)
   ax_prt.set_ylim(0, 2)
   ax_prt.set_title('Stimulation Protocol: '+ fmr_name)
   
   counter = 0
   
   # plot heart rate, rescaled to arbitrary range
   if 'pulse_col_dict' in locals():
       hr_ms_norm = (((1.95-1.05)*(hr_ms-min(hr_ms))) / (max(hr_ms) - min(hr_ms))) + 1.05
       plot_hr, = ax_prt.plot(physio_time_hr_ms, hr_ms_norm, linewidth=1, color='red', label='heart rate')
       counter = counter + 1
   # plot breathing rate, rescaled to arbitrary range    
   if "resp_col_dict" in locals():
       br_ms_norm = (((0.95-0.05)*(br_ms-min(br_ms))) / (max(br_ms) - min(br_ms))) + 0.05
       plot_br, = ax_prt.plot(physio_time_br_ms, br_ms_norm, linewidth=1, color='blue', label='respiratory rate')
       counter = counter + 1
   # get current handles of heart and/or respiratory rate
   handles, labels = ax_prt.get_legend_handles_labels()
   # add vertical separation between respiratory and heart rate
   ax_prt.add_patch(Rectangle((0,1),fmr_no_volumes*fmr_time_repeat/1000,0.0005,
                       facecolor = 'black', edgecolor = 'black',
                       alpha=1))
   # add text to indicate heart rate and breathing rate box
   ax_prt.text(((fmr_no_volumes*fmr_time_repeat/1000)/2), 1.02, 'Cardiac', horizontalalignment ='center', fontsize=10)
   ax_prt.text(((fmr_no_volumes*fmr_time_repeat/1000)/2), 0.02, 'Respiratory', horizontalalignment ='center', fontsize=10)
   # plot the single PRT events
   for cond in range(len(protocol.conditions)):
       for event in range(np.size(protocol.conditions[cond].data,0)):
          ax_prt.add_patch(Rectangle((protocol.conditions[cond].data[event,0]/1000,0),
                              (protocol.conditions[cond].data[event,1] - protocol.conditions[cond].data[event,0])/1000, 2,
                              facecolor = np.array(protocol.conditions[cond].colour)/255,
                              edgecolor = 'none',
                              alpha=0.3))
   # create handles of the PRT conditions for the legend
   prt_handles = [Patch(color= np.array(protocol.conditions[cond].colour)/255, label=protocol.conditions[cond].name) for cond in range(len(protocol.conditions))]
   # add condition handles to original plot handles
   handles =  handles + prt_handles

   ax_prt.set_yticks([])
   ax_prt.set_ylabel('Signal Variation')
   ax_prt.set_xlabel('Time (s)')
   fig_prt.legend(handles= handles, ncol=len(protocol.conditions)+counter, loc=8)
   fig_prt.savefig((physio_out_path + '/' + fmr_name + '_PRT_PhysioRate.png'), dpi=600, format='png')
   if (len(sys.argv) == 1) or (len(sys.argv) > 1 and plotdisp.lower() == 'true'):
       plt.show()
    


### SAVING PHYSIO-PARAMETER FILES

# int32 is not JSON serializable -> convert to int 
physio_input_parameters["ScriptParameters"]["ShiftsOfHeartRateRegressorSec"] = [int(item) for item in physio_input_parameters["ScriptParameters"]["ShiftsOfHeartRateRegressorSec"]] 
physio_input_parameters["ScriptParameters"]["ShiftsOfRVTRegressorSec"] = [int(item) for item in physio_input_parameters["ScriptParameters"]["ShiftsOfRVTRegressorSec"]] 
physio_input_parameters["ScriptParameters"]["ShiftsOfRVRegressorSec"] = [int(item) for item in physio_input_parameters["ScriptParameters"]["ShiftsOfRVRegressorSec"]] 
physio_input_parameters["ScriptParameters"]["ShiftsOfENVRegressorSec"] = [int(item) for item in physio_input_parameters["ScriptParameters"]["ShiftsOfENVRegressorSec"]] 

# remove the ndarrays from physio_input_parameters to save it as a json file 
keys = ['IndicesScanTriggers', 'IndicesVolumeTriggers', 'PhysioData', 'PhysioTimeSec']
for key in keys:
    physio_input_parameters['PhysioJsonFile'].pop(key, None)  
    
# Save the physio_input_parameters as json file
with open((physio_out_path + '/' + fmr_name + '_PhysioProcessing_InputParameters.json'), 'w') as write_file:
    json.dump(physio_input_parameters, write_file, indent=4)


physio_output_parameters['NoiseRegressors'] = physio_regressors_names
# Save the physio_output_parameters as json file
with open((physio_out_path + '/' + fmr_name + '_PhysioProcessing_OutputParameters.json'), 'w') as write_file:
    json.dump(physio_output_parameters, write_file, indent=4)

# Save all created noise regressors in a single file for external use    
df = pd.DataFrame(physio_regressors_matrix)
df.columns = physio_regressors_names
df.to_csv(physio_out_path + '/' + fmr_name + '_PhysiologicalNoiseRegressors.tsv', sep="\t")

# Restore the default plotting parameters after script is finished
mpl.rcParams.update(mpl.rcParamsDefault)
    