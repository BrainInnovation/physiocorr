# -*- coding: utf-8 -*-
"""
Batch Processing of Physio Files
This script can be used to loop over the script "CreatePhysioPredictors_BIDS.py" and create the physiological
regressors for multiple functional runs at the same time.
Please note: To date this script works only if the pulse and respiratory data are not saved in separate files.

The user can choose to display the figures during the processing of the separate functional runs by setting the 
plotdisp parameter in this file to True. When no plots shall be displayed, please use plotdisp = False

As input it requires a TSV file containing the path and filename of:
    the physio recordings, the corresponding FMR file, as well as a task design matrix SDM (optional) and a PRT file (optional). 
Each row should represent the files of a single dataset, in the order of JSON FMR SDM PRT. 
Please make sure to indicate missing SDM or PRT files with 'na' or any other missing data identifier that you specify in the 
header of this script, e.g.
    sub-01_ses-01_task-rest_run-01_Physio.json \TAB sub-01_ses-01_task-rest_run-01_BOLD.fmr \TAB sub-01_ses-01_task-rest_run-01_design.sdm \TAB sub-01_ses-01_task-rest_run-01_stimulationprotocol.prt
    sub-01_ses-01_task-rest_run-02_Physio.json \TAB sub-01_ses-01_task-rest_run-02_BOLD.fmr \TAB sub-01_ses-01_task-rest_run-01_design.sdm \TAB na
    sub-01_ses-01_task-rest_run-03_Physio.json \TAB sub-01_ses-01_task-rest_run-03_BOLD.fmr \TAB na \TAB sub-01_ses-01_task-rest_run-01_stimulationprotocol.prt

In case your study does not include any experimental manipulation (e.g. resting state data), please omit the SDM and PRT column completely, e.g.:
    sub-01_ses-01_task-rest_run-01_Physio.json \TAB sub-01_ses-01_task-rest_run-01_BOLD.fmr
    sub-01_ses-01_task-rest_run-02_Physio.json \TAB sub-01_ses-01_task-rest_run-02_BOLD.fmr
    
 Make sure to check the quality of the input and output data or you might run the risk of using incorrect physiological noise measures!
"""

__author__ = "Judith Eck"
__date__ = "19-07-2022"
__name__ = "BatchProcessingPhysioFiles.py"
__version__ = "0.1.0"


from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
import os.path
import sys
import numpy as np

app = QApplication(sys.argv)
    
na_identifier = 'na'
plotdisp = True


def showdialog_fileerror():
    '''
    show an error message to the user if there is not at least a JSON and FMR file specified for every dataset
    '''
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setText("No FMR and Physio file specified for every dataset, please check the input TSV file!")
    msg.setWindowTitle("FileError")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec()

pyscript_name, _ = QFileDialog.getOpenFileName(None, 'Select CreatePhysioPredictors_BIDS.py', os.getcwd(), filter='PY Files (*.py)')

tsv_name, _ = QFileDialog.getOpenFileName(None, 'Select TSV file with path information for every dataset', os.curdir, filter='TSV Files (*.tsv)')
temp_files = np.genfromtxt(fname=tsv_name, delimiter='\t', dtype=str, missing_values=na_identifier, autostrip=True)

for counter in range(np.size(temp_files, axis=0)):
    if not (np.size(temp_files, axis=1) == 2 or np.size(temp_files, axis=1) == 4):
        showdialog_fileerror()
        
    if temp_files[counter,0].endswith('.json'):
        physio_json_name = str(temp_files[counter,0].astype(str))
    else:
        physio_json_name = 'na'
    
    if temp_files[counter,1].endswith('.fmr'):
        fmr_file_name = str(temp_files[counter,1].astype(str))
    else:
        fmr_file_name = 'na'
        
    if np.size(temp_files, axis=1) == 4:
        if temp_files[counter,2].endswith('.sdm'):
            sdm_task_file_name = str(temp_files[counter,2].astype(str))
        else:
            sdm_task_file_name = 'na'
        
        if temp_files[counter,3].endswith('.prt'):
            prt_task_file_name = str(temp_files[counter,3].astype(str))
        else:
            prt_task_file_name = 'na'
    else:
            sdm_task_file_name = 'na'
            prt_task_file_name = 'na'
            
    print("\nProcessing of physiological data for run: \n" + fmr_file_name)        
    os.system('python ' + pyscript_name + ' ' + physio_json_name + ' ' + fmr_file_name + ' ' +sdm_task_file_name + ' ' + prt_task_file_name + ' ' + str(plotdisp))
    