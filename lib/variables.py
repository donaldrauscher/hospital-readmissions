
# variable groupings
med_var = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',\
           'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol',\
           'troglitazone', 'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin',\
           'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']

lab_var = ['A1Cresult', 'max_glu_serum']

demo_var = ['race', 'gender', 'age', 'weight']

diag_var = ['diag_1', 'diag_2', 'diag_3']

other_var = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',\
             'payer_code', 'medical_specialty', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient',\
             'number_emergency', 'number_inpatient', 'number_diagnoses', 'diabetesMed', 'change']

xvar = med_var + lab_var + demo_var + diag_var + other_var

cat_var = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',\
           'race', 'gender', 'payer_code', 'medical_specialty', 'diag']

hcc_cat_var = ['diag_first']
