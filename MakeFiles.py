# import numpy as np
import os
import pandas as pd
# import csv
# Extracted C71 Code descriptions - SQL code in current directory

# Note Text Compilation per patient
    # To be replaced with direct data pull
notes = pd.read_csv('Patient data.csv', encoding = "ISO-8859-1")

PatientList = notes.PAT_MRN_ID.unique()
len(PatientList) # 782 Patients! :D

os.mkdir('C:/Users/srajendr/WFU/PatientData') # These are named after patient MRN
i=0
for i in range(0,len(PatientList)):
    note = notes[notes.PAT_MRN_ID == PatientList[i]]
    filename = 'C:/Users/srajendr/WFU/PatientData'+ str(PatientList[i]) + '.csv'
    note['NOTE_TEXT'].to_csv(filename, index=False, header=False) #, quoting=csv.QUOTE_NONE)


# DX description compilation by DX_ID
    # To be replaced with direct data pull
    
#  Codes = pd.read_csv('Algorithm Development Pull - Aug18/C71 DX Options.csv')
# len(Codes.index)  # 514 codes

# os.mkdir('data/CodeFiles') # These are named after DX_ID
# i=0
# for i in range(0,len(Codes.index)):
#     code = Codes.DX_NAME[i]
#     filename = 'data/CodeFiles/'+ str(Codes.DX_ID[i])
#     file = open(filename, 'w')
#     file.write(code)
#     file.close()


# Noble Coder Concept Extraction
# To be automated - pursued in Sept

# ToDo Fix data pull -- Avoid doing it by hand...
#Current command line solution -- Null pointer exception issue
# java -jar NobleCoder-1.0.jar -terminology 'thesaurus' -input '/CodeFiles/' -output '/CodeFiles/Out/' -search all-match

# # Paramenters fo Note files
# 'thesaurus'
# '/NoteFiles'
# '/NoteFiles/Out'
