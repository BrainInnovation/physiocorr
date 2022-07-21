# -*- coding: utf-8 -*-
"""
Save output from readCMRRPhysio.py in BIDS format, taking number of skipped
volumes in FMR into account.

This script runs with physiological data (pulse and/or respiratory) saved as
_Info.log, _PULS.log, _RESP.log or for physiological data saved in DICOM/IMA

As input it requires the physiological recordings (as specified above) and the
corresponding FMR file (generated with BV 21.4 or newer).

The output (BIDS-compatible physiological recordings) is saved in the same
folder as the FMR and with the same file-ID as the FMR appended by _Physio.json
and _Physio.tsv.gz, e.g. 
sub-01_ses-01_task-emotion_run-01_bold_Physio.tsv.gz
sub-01_ses-01_task-emotion_run-01_bold_Physio.json

If used in batch mode, a TSV file as input is required. This TSV file lists
in each line the full path information of the physiorecording (IMA/DCM or _Info.log) 
and the full path information of the corresponding FMR file, e.g.

C:/Users/BV/Documents/Physio/JUDECK_Test.0009.0001.2021.08.30.12.42.16.691467.153832522.IMA \TAB C:/Users/BV/Documents/Physio/sub-01_ses-01_task-emotion_run-01_bold.fmr
C:/Users/BV/Documents/Physio/JUDECK_Test.0006.0001.2021.10.08.12.25.28.25127.192152643.IMA \TAB C:/Users/BV/Documents/Physio/sub-02_ses-01_task-emotion_run-01_bold.fmr

or

C:/Users/BV/Documents/Physio/Physio_20220101_133820_f5a92f1f-bbfd-453f-b040-694202db6740_Info.log \TAB C:/Users/BV/Documents/Physio/sub-01_ses-01_task-emotion_run-01_bold.fmr
C:/Users/BV/Documents/Physio/Physio_20220202_110519_7e3388f0-859a-43ee-92a3-6f522353a933_Info.log \TAB C:/Users/BV/Documents/Physio/sub-02_ses-01_task-emotion_run-01_bold.fmr

Works with readCMRRPhysio.py (@author: Marcel Zwiers) retrieved from 
https://github.com/CMRR-C2P/MB/blob/master/readCMRRPhysio.py
(13-07-2022)

Please make sure that readCMRRPhysio.py is saved in the same folder as this
script!

"""

__author__ = "Judith Eck"
__version__ = "0.1.0"
__date__ = "21-07-2022"
__name__ = "SavePhysiologInBIDSfrom_readCMRRPhysio"

# =============================================================================
# Import required packages
# =============================================================================
import numpy as np
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QApplication
import json
import os.path
import sys
from readCMRRPhysio import readCMRRPhysio


app = QApplication(sys.argv)

def userinput_batchmode():
    '''
    messagebox to ask user to use manual or batch mode
    '''   
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Question)
    msg.setText("Would you like to convert the physio recordings from a single run (yes) or use the batch mode (no)?")
    msg.setWindowTitle("Choice of Processing Mode")
    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    retval = msg.exec()
    if retval == QMessageBox.Yes:
        single = True
    else:
        single = False
    return(single)

def showdialog_slicetimes():
    '''
    messagebox used to inform user about missing slice time information in FMR
    '''
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText("'No Slice Acquisition Information saved in Slice Timing Table of FMR'")
    msg.setWindowTitle("Missing Slice Times in FMR")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()
    
def showdialog_nophysio():
    '''
    messagebox used to inform user about missing physio data in logfiles
    '''
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Warning)
    msg.setText("'No physiological data has been recorded. Please check for any data acquisition problems'")
    msg.setWindowTitle("No Physiological Data")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()

single = userinput_batchmode()

# select the Physiofile and the corresponding FMR
if single:
    tempfiles = np.empty((1,2), dtype="object")
    logfile, _ = QFileDialog.getOpenFileName(None, 'Select the Physio recordings', os.getcwd(), filter="IMA (*.ima);; LOG (*_Info.log);; DCM (*.dcm)")
    tempfiles[0,0] = logfile
    fmrfile, _ = QFileDialog.getOpenFileName(None, 'Select FMR File of corresponding fMRI dataset', os.getcwd(), filter="FMR (*.fmr)")
    tempfiles[0,1] = fmrfile
    del (logfile, fmrfile)
    
# select the TSV file with the path information of all physiorecordings and their corresponding FMR files
else:
    tsv_name, _ = QFileDialog.getOpenFileName(None, 'Select TSV File', os.curdir, filter='TSV Files (*.tsv)')
    tempfiles = np.genfromtxt(fname=tsv_name, delimiter='\t', dtype=str, autostrip=True)
    if np.shape(tempfiles) == (2,):
        tempfiles = tempfiles[np.newaxis]


# loop through all physiorecordings and the corresponding FMRs to create the BIDS-compatible physio files
for counter in range(np.size(tempfiles, axis=0)):
    with open(tempfiles[counter,1]) as _file:
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
    fmr_skip_volumes = int(
        ''.join([line for line in _fmr if 'NrOfSkippedVolumes:' in line]).split(':')[1]
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
        # if it is a MB sequence several slices have been acquired at the same time, only unique entries from the SliceTimeTable of the FMR will be used
        fmr_slice_times = np.unique(fmr_slice_table)
        
    if np.size(fmr_slice_table) is not fmr_no_slices:
        showdialog_slicetimes()

    del [_temp, _index, _file,  _fmr]
    
    # extract the information from the physio log files, using the CMRR script by Marcel Zwiers
    if tempfiles[counter,0].lower().endswith('.log'):
        temp = readCMRRPhysio(tempfiles[counter,0][:-9])
    else:
        temp = readCMRRPhysio(tempfiles[counter,0])

    '''
    delete all triggers before the first slice acquisition of the first volume to 0 
    (taking into account skipped volumes in the FMR)
    (or in case a trigger is sent for start of the scanning or something comparable)
    '''
    first_data_val = int(temp['SliceMap'][0,fmr_skip_volumes,0])
    temp['ACQ'][:first_data_val] = False

    ''' 
    In case the number of volumes at the scanner console had been set larger than the actual experimental run duration and the scanner was stopped manually, trigger
    information of several additional complete or partial volumes are saved in the physio recordings.
    To avoid any processing problems all trigger information will be removed before the acquisition start of the first slice of the next volume not 
    included anymore in the FMR
    '''
    vol_onsets = np.int0(np.insert(temp['ACQ'],0,0))
    if sum(np.diff(vol_onsets) == 1)/np.size(fmr_slice_times) > fmr_no_volumes:
        last_data_val = int(temp['SliceMap'][0,fmr_no_volumes,0])-1
        temp['ACQ'][last_data_val+1:] = False

    '''
    Save output in TSV and JSON file
    '''
    outputname = tempfiles[counter,1].rsplit(".")[0]
    
    if not 'RESP' in temp and 'PULS' in temp:
        data = np.column_stack((temp['PULS'],temp['ACQ']))
        np.savetxt((outputname + '_Physio.tsv.gz'), data, fmt= ['%f','%f'], delimiter="\t")
        col = ["cardiac", "trigger"]
    elif not 'PULS' and 'RESP' in temp:
        data = np.column_stack((temp['RESP'],temp['ACQ']))
        np.savetxt((outputname + '_Physio.tsv.gz'), data, fmt= ['%f','%f'], delimiter="\t")
        col = ["respiratory", "trigger"]
    elif not 'PULS' and not 'RESP' in temp:
        data = np.int0(temp['ACQ'])
        np.savetxt((outputname + '_Physio.tsv.gz'), data, fmt= '%f', delimiter="\t")
        col = ["trigger"]
        showdialog_nophysio()
    else:
        data = np.column_stack((temp['PULS'],temp['RESP'],temp['ACQ']))
        np.savetxt((outputname + '_Physio.tsv.gz'), data, fmt= ['%f','%f', '%f'], delimiter="\t")
        col = ["cardiac", "respiratory", "trigger"]
        
    # Save the corresponding json file
    jsoninput = {"SamplingFrequency": temp['Freq'], "StartTime": -(np.where(temp['ACQ']==1)[0][0]+1)/temp['Freq'], "Columns": col}
    with open((outputname + '_Physio.json'), 'w') as write_file:
        json.dump(jsoninput, write_file, indent=0)
        