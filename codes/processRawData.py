# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 18:04:32 2020

@author: DanielAvila

        This is the first code from the research on the Vulnerability Index 
        for municipalities in Mexico with regards to the COVID19.
"""


import gc
import numpy as np
import pandas as pd

raw_data = '~/RawData/'

#%%
# Loads age cohorts from SSA from reported COVID19 cases in Mexico
age_cohort = pd.read_csv(raw_data + 'casos_COVID19MX.csv', index_col=0)

# Calculates cases with 60 years or more and appends to dataframe
sixty_more = pd.DataFrame(age_cohort.iloc[6:,:].sum(axis=0)).T
sixty_more.index = ['60 y mas']
age_cohort = age_cohort.iloc[0:6,:].append(sixty_more, ignore_index=False, sort=False)

# Gets number of cases per cohort age
age_cohort_score = pd.DataFrame(age_cohort.sum(axis = 1), columns=['Total'])
score = [((100 / np.power(2,x+1)) / 100) for x in reversed(range(len(age_cohort_score['Total'])))]

#%%
# Loads population from CONAPO
pop_muni_1 = pd.read_csv(raw_data + 'base_municipios_final_datos_01.csv', encoding='latin')
pop_muni_2 = pd.read_csv(raw_data + 'base_municipios_final_datos_02.csv', encoding='latin')

pop_muni = pd.concat([pop_muni_1, pop_muni_2])

pop_muni = pop_muni[pop_muni['ANIO'] == 2018]
pop_muni = pd.concat([pop_muni.iloc[:,1:], pop_muni['EDAD_QUIN'].str.split('_', expand=True)], axis=1)

pop_muni = pop_muni.rename(columns={1:'RANGO'})
pop_muni['RANGO'] = pop_muni['RANGO'].astype(int)

# Adds leading zeros to column
pop_muni['CLAVE'] = pop_muni['CLAVE'].map('{:0>5}'.format)

# Groups by age cohort similar to COVID19 data
pop_muni_cohort = pop_muni.copy()
pop_muni_cohort = pop_muni_cohort.groupby(['CLAVE', 'NOM_ENT', 'CLAVE_ENT',
                     'MUN', 'SEXO', pd.cut(pop_muni_cohort['RANGO'], np.arange(-1, 65 + 5, 10))])['POB'].sum().reset_index()

# Converts age cohort to string
pop_muni_cohort['RANGO'] = pop_muni_cohort['RANGO'].astype(str)

# Creates dict to from age cohort (bins) and age cohort (COVID19) and maps it
range_cohort = dict(zip(list(pop_muni_cohort['RANGO'].unique()), list(age_cohort_score.index.unique())))
pop_muni_cohort['RANGO'] = pop_muni_cohort['RANGO'].replace(range_cohort)

pop_muni_cohort = pop_muni_cohort.groupby(['CLAVE', 'NOM_ENT', 'CLAVE_ENT','MUN', 'SEXO', 'RANGO'])['POB'].sum().reset_index()

print('Municipalities age cohort ready')
#%%
    #######################################
    # Process data concerning comorbility #
    #######################################
    
    
    #######################################
    #      Kids (0 to 19 years old)       #
    #######################################

# Loads ENSANUT data from kids (0 to 9)
ensanut_kid = pd.read_csv(raw_data + 'CS_NINO.csv')

ensanut_kid['CONTEO'] = 1
ensanut_kid['TOTAL'] = ensanut_kid['CONTEO'].mul(ensanut_kid['F_NINO'])

ensanut_kid_cohort = ensanut_kid.groupby(['UPM', 'VIV_SEL', 'HOGAR',
                                          'EDAD', 'EDAD_MESES', 
                                          'SEXO',
                                          'ENT', 'DOMINIO', 'REGION'])['TOTAL'].sum().reset_index()

print('Kids data ready')
#%%
    #######################################
    #   Teenagers (10 to 19 years old)    #
    #######################################
    
# Loads ENSANUT data from teenagers (10 to 19)
ensanut_teen = pd.read_csv(raw_data + 'CS_ADOLESCENTES.csv')
ensanut_teen['CONTEO'] = 1
ensanut_teen['TOTAL'] = ensanut_teen['CONTEO'].mul(ensanut_teen['F_10A19'])

# Selects columns of interest
ensanut_teen_smoking = ensanut_teen.loc[:,['P1_1', 'P1_2', 'P1_3', 'P1_4', 
                                           'P1_5', 'P1_6_1', 'P1_6_2', 'P1_7_1',
                                           'EDAD', 'SEXO', 'ENT', 'DOMINIO',
                                           'REGION', 'F_10A19', 'CONTEO','TOTAL']]

# Deselects non-smoking teenagers
ensanut_teen_smoking_nonA = ensanut_teen_smoking[(ensanut_teen_smoking['P1_1'] == 2) & (ensanut_teen_smoking['P1_2'] == 3) & (ensanut_teen_smoking['P1_4'] == 3)]
ensanut_teen_smoking_nonB = ensanut_teen_smoking[(ensanut_teen_smoking['P1_1'] == 8) & (ensanut_teen_smoking['P1_2'] == 3) & (ensanut_teen_smoking['P1_4'] == 3)]
ensanut_teen_smoking_nonC = ensanut_teen_smoking[(ensanut_teen_smoking['P1_1'] == 9) & (ensanut_teen_smoking['P1_2'] == 3) & (ensanut_teen_smoking['P1_4'] == 3)]
ensanut_teen_smoking_non = pd.concat([ensanut_teen_smoking_nonA, ensanut_teen_smoking_nonB, ensanut_teen_smoking_nonC])

# Selects actual smokers
ensanut_teen_smoking_do = ensanut_teen_smoking[~ensanut_teen_smoking.isin(ensanut_teen_smoking_non.to_dict('l')).all(1)]

# Groups by classifiers as to get total sum
ensanut_teen_smoking_do = ensanut_teen_smoking_do.groupby(['P1_1','P1_2','EDAD','SEXO',
                                                            'ENT','DOMINIO'])['TOTAL'].sum().reset_index() 

# Total smokers in age cohort 10 - 19 per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_teen_smoking_summary = ensanut_teen_smoking_do.groupby(['ENT', 'DOMINIO'])['TOTAL'].sum().reset_index() 
ensanut_teen_smoking_summary_sex = ensanut_teen_smoking_do.groupby(['SEXO', 'ENT', 'DOMINIO'])['TOTAL'].sum().reset_index()                                                         

# Selects columns of interest
ensanut_teen_obesity = ensanut_teen.loc[:,['P4_1_2', 'P4_1_3', 'P4_1_5', 'P4_1_6',
                                           'P4_1_7', 'P4_1_8', 'P4_1_9', 'P4_1_10',
                                           'EDAD', 'SEXO', 'ENT', 'DOMINIO',
                                           'REGION', 'F_10A19', 'CONTEO','TOTAL']]

# Deselects non-obese teenagers
ensanut_teen_obesity_nonA = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 1)]

ensanut_teen_obesity_nonB = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 2) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 1)]

ensanut_teen_obesity_nonC = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 2) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 1)]

ensanut_teen_obesity_nonD = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 2) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 1)]

ensanut_teen_obesity_nonE = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 2) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 1)]

ensanut_teen_obesity_nonF = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 2) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 1)]

ensanut_teen_obesity_nonG = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 2) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 1)]

ensanut_teen_obesity_nonH = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 2)]

ensanut_teen_obesity_nonI = ensanut_teen_obesity[(ensanut_teen_obesity['P4_1_2'] == 2) &
                                                 (ensanut_teen_obesity['P4_1_3'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_5'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_6'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_7'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_8'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_9'] == 1) &
                                                 (ensanut_teen_obesity['P4_1_10'] == 1)]

ensanut_teen_obesity_non = pd.concat([ensanut_teen_obesity_nonA, 
                                      ensanut_teen_obesity_nonB, 
                                      ensanut_teen_obesity_nonC,
                                      ensanut_teen_obesity_nonD,
                                      ensanut_teen_obesity_nonE,
                                      ensanut_teen_obesity_nonF,
                                      ensanut_teen_obesity_nonG,
                                      ensanut_teen_obesity_nonH,
                                      ensanut_teen_obesity_nonI])

ensanut_teen_obesity_do = ensanut_teen_obesity[~ensanut_teen_obesity.isin(ensanut_teen_obesity_non.to_dict('l')).all(1)]

# Groups by classifiers as to get total sum
ensanut_teen_obesity_do = ensanut_teen_obesity_do.groupby(['P4_1_2','P4_1_3','EDAD',
                                                           'SEXO','ENT','DOMINIO'])['TOTAL'].sum().reset_index() 

# Total smokers in age cohort 10 - 19 per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_teen_obesity_summary = ensanut_teen_obesity_do.groupby(['ENT', 'DOMINIO'])['TOTAL'].sum().reset_index() 
ensanut_teen_obesity_summary_sex = ensanut_teen_obesity_do.groupby(['SEXO', 'ENT', 'DOMINIO'])['TOTAL'].sum().reset_index() 
   

# Groups by classifiers as to get total sum 
# Age cohort 10 - 19 per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_teen_cohort = ensanut_teen.groupby(['SEXO', 'ENT', 'DOMINIO'])['TOTAL'].sum().reset_index()
ensanut_teen_cohort_tot = ensanut_teen_cohort['TOTAL'].sum()

print('Teens data ready')
#%%
def createSummary(dataframe):
    """ Process a database sorting and grouping it by cohort age.
        """
    age_dict_cohort = {'(19, 29]': '20 a 29', 
                    '(29, 39]':'30 a 39', 
                    '(39, 49]':'40 a 49', 
                    '(49, 59]':'50 a 59',
                    '(59, 69]':'60 y mas',
                    '(69, 79]':'60 y mas',
                    '(79, 89]':'60 y mas',
                    '(89, 99]':'60 y mas'}
    
    local = dataframe.copy()
    local_summary = local.groupby([pd.cut(local['EDAD'], np.arange(19, 65 + 40, 10)),'SEXO', 'ENT', 'DOMINIO'])['TOTAL'].sum().reset_index()
    local_summary_sex = local.groupby(['SEXO', 'ENT', 'DOMINIO'])['TOTAL'].sum().reset_index()
    
    local_summary['EDAD'] = local_summary['EDAD'].astype(str)
    local_summary['EDAD'] = local_summary['EDAD'].replace(age_dict_cohort)
    
    local_summary = local_summary.groupby(['EDAD','SEXO', 'ENT', 'DOMINIO'])['TOTAL'].sum().reset_index()
     
    gc.collect()
    
    return local_summary, local_summary_sex

#%%
    #######################################
    #    Adults (20 to +60 years old)     #
    #######################################    


# Loads ENSANUT data from adults (20 to +60)
ensanut_adult = pd.read_csv(raw_data + 'CS_ADULTOS.csv')

ensanut_adult['CONTEO'] = 1
ensanut_adult['TOTAL'] = ensanut_adult['CONTEO'].mul(ensanut_adult['F_20MAS'])

# Selects columns of interest for obesity analysis
ensanut_adult_obesity = ensanut_adult.loc[:,['P1_1', 'P1_2', 'P1_3','P1_6',
                                             'P1_7', 'P1_8',
                                             'EDAD', 'SEXO', 'ENT', 'DOMINIO',
                                             'F_20MAS', 'CONTEO','TOTAL']]

ensanut_adult_obesity_nonA = ensanut_adult_obesity[(ensanut_adult_obesity['P1_1'] == 2) &
                                                   (ensanut_adult_obesity['P1_6'] == 3)]

ensanut_adult_obesity_nonB = ensanut_adult_obesity[(ensanut_adult_obesity['P1_1'] == 2) &
                                                   (ensanut_adult_obesity['P1_7'] == 3)]

ensanut_adult_obesity_nonC = ensanut_adult_obesity[(ensanut_adult_obesity['P1_1'] == 2) &
                                                   (ensanut_adult_obesity['P1_7'] == 4)]

ensanut_adult_obesity_nonD = ensanut_adult_obesity[(ensanut_adult_obesity['P1_1'] == 2) &
                                                   (ensanut_adult_obesity['P1_6'] == 2) &
                                                   (ensanut_adult_obesity['P1_7'] == 2)]

ensanut_adult_obesity_nonE = ensanut_adult_obesity[(ensanut_adult_obesity['P1_1'] == 2) &
                                                   (ensanut_adult_obesity['P1_6'] == 2) &
                                                   (ensanut_adult_obesity['P1_7'] == 9)]

ensanut_adult_obesity_nonF = ensanut_adult_obesity[(ensanut_adult_obesity['P1_1'] == 2) &
                                                   (ensanut_adult_obesity['P1_6'] == 9) &
                                                   (ensanut_adult_obesity['P1_7'] == 9)]

ensanut_adult_obesity_non = pd.concat([ensanut_adult_obesity_nonA, 
                                       ensanut_adult_obesity_nonB, 
                                       ensanut_adult_obesity_nonC,
                                       ensanut_adult_obesity_nonD,
                                       ensanut_adult_obesity_nonE,
                                       ensanut_adult_obesity_nonF])

# Only keeps records that are effectively obese
ensanut_adult_obesity_do = ensanut_adult_obesity[~ensanut_adult_obesity.isin(ensanut_adult_obesity_non.to_dict('l')).all(1)]

# Total obese in age cohort per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_adult_obesity_summary, ensanut_adult_obesity_summary_sex = createSummary(ensanut_adult_obesity_do)

print('Adults obesity data ready')
#%%
# Selects columns of interest for diabetes analysis
ensanut_adult_diabetes = ensanut_adult.loc[:,['P3_1', 'P3_2', 'P3_3','P3_8',
                                             'P3_9M', 'P3_9A', 'P3_10M', 'P3_10A',
                                             'P3_11', 'P3_12',
                                             'EDAD', 'SEXO', 'ENT', 'DOMINIO',
                                             'F_20MAS', 'CONTEO','TOTAL']]
    
ensanut_adult_diabetes_nonA = ensanut_adult_diabetes[(ensanut_adult_diabetes['P3_1'] == 3) &
                                                     (ensanut_adult_diabetes['P3_2'].isnull()) &
                                                     (ensanut_adult_diabetes['P3_3'].isnull()) &
                                                     (ensanut_adult_diabetes['P3_8'].isnull()) &
                                                     (ensanut_adult_diabetes['P3_9M'].isnull()) &
                                                     (ensanut_adult_diabetes['P3_9A'].isnull()) &
                                                     (ensanut_adult_diabetes['P3_10M'].isnull()) &
                                                     (ensanut_adult_diabetes['P3_10A'].isnull()) &
                                                     (ensanut_adult_diabetes['P3_11'].isnull()) &
                                                     (ensanut_adult_diabetes['P3_12'].isnull())]

# Only keeps records that are effectively diabetic
ensanut_adult_diabetes_do = ensanut_adult_diabetes[~ensanut_adult_diabetes.isin(ensanut_adult_diabetes_nonA.to_dict('l')).all(1)]

# Total diabetic in age cohort per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_adult_diabetes_summary, ensanut_adult_diabetes_summary_sex = createSummary(ensanut_adult_diabetes_do) 

print('Adults diabetes data ready')
#%%
# Selects columns of interest for arterial hypertension analysis
ensanut_adult_hypertension = ensanut_adult.loc[:,['P4_1', 'P4_2M', 'P4_2A','P4_3',
                                             'P4_4', 'P4_5M', 'P4_5A', 'P4_6',
                                             'P4_9',
                                             'EDAD', 'SEXO', 'ENT', 'DOMINIO',
                                             'F_20MAS', 'CONTEO','TOTAL']]

ensanut_adult_hypertension_nonA = ensanut_adult_hypertension[(ensanut_adult_hypertension['P4_1'] == 2)]

# Only keeps records that are effectively hypertense
ensanut_adult_hypertension_do = ensanut_adult_hypertension[~ensanut_adult_hypertension.isin(ensanut_adult_hypertension_nonA.to_dict('l')).all(1)]

# Total hypertense in age cohort per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_adult_hypertension_summary, ensanut_adult_hypertension_summary_sex = createSummary(ensanut_adult_hypertension_do) 

print('Adults hypertension data ready')
#%%
# Selects columns of interest for cardiovascular disease analysis
ensanut_adult_cardiovascular = ensanut_adult.loc[:,['P5_1', 'P5_2_1', 'P5_2_2','P5_2_3',
                                                    'P5_3', 'P5_4', 'P5_5', 'P5_6',
                                                    'P5_7',
                                                    'EDAD', 'SEXO', 'ENT', 'DOMINIO',
                                                    'F_20MAS', 'CONTEO','TOTAL']]

ensanut_adult_cardiovascular_nonA = ensanut_adult_cardiovascular[(ensanut_adult_cardiovascular['P5_1'] == 2) &
                                                               (ensanut_adult_cardiovascular['P5_2_1'] == 2) &
                                                               (ensanut_adult_cardiovascular['P5_2_2'] == 2) &
                                                               (ensanut_adult_cardiovascular['P5_2_3'] == 2)]

# Only keeps records that are effectively with cardiovascular disease
ensanut_adult_cardiovascular_do = ensanut_adult_cardiovascular[~ensanut_adult_cardiovascular.isin(ensanut_adult_cardiovascular_nonA.to_dict('l')).all(1)]

# Total with cardiovascular disease in age cohort per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_adult_cardiovascular_summary, ensanut_adult_cardiovascular_summary_sex = createSummary(ensanut_adult_cardiovascular_do) 

print('Adults cardiovascular data ready')
#%%
# Selects columns of interest for renal disease analysis
ensanut_adult_renal = ensanut_adult.loc[:,['P6_1_1', 'P6_1_2', 'P6_1_3',
                                           'EDAD', 'SEXO', 'ENT', 'DOMINIO',
                                           'F_20MAS', 'CONTEO','TOTAL']]

ensanut_adult_renal_nonA = ensanut_adult_renal[(ensanut_adult_renal['P6_1_1'] == 2) &
                                               (ensanut_adult_renal['P6_1_2'] == 2) &
                                               (ensanut_adult_renal['P6_1_3'] == 2)]

# Only keeps records that are effectively with cardiovascular disease
ensanut_adult_renal_do = ensanut_adult_renal[~ensanut_adult_renal.isin(ensanut_adult_renal_nonA.to_dict('l')).all(1)]

# Total with cardiovascular disease in age cohort per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_adult_renal_summary, ensanut_adult_renal_summary_sex = createSummary(ensanut_adult_renal_do) 

print('Adults renal data ready')
#%%
# Selects columns of interest for smoking disease analysis
ensanut_adult_smoking = ensanut_adult.loc[:,['P13_1', 'P13_2', 'P13_3',
                                             'P13_4', 'P13_5', 'P13_6',
                                             'P13_6_1', 'P13_6_2',
                                             'EDAD', 'SEXO', 'ENT', 'DOMINIO',
                                             'F_20MAS', 'CONTEO','TOTAL']]

# Deselects non-smoking teenagers
ensanut_adult_smoking_nonA = ensanut_adult_smoking[(ensanut_adult_smoking['P13_1'] == 2) & 
                                                   (ensanut_adult_smoking['P13_2'] == 3) & 
                                                   (ensanut_adult_smoking['P13_4'] == 3)]

ensanut_adult_smoking_nonB = ensanut_adult_smoking[(ensanut_adult_smoking['P13_1'] == 8) & 
                                                   (ensanut_adult_smoking['P13_2'] == 3) & 
                                                   (ensanut_adult_smoking['P13_4'] == 3)]

ensanut_adult_smoking_nonC = ensanut_adult_smoking[(ensanut_adult_smoking['P13_1'] == 9) & 
                                                   (ensanut_adult_smoking['P13_2'] == 3) & 
                                                   (ensanut_adult_smoking['P13_4'] == 3)]

ensanut_adult_smoking_non = pd.concat([ensanut_adult_smoking_nonA, 
                                       ensanut_adult_smoking_nonB, 
                                       ensanut_adult_smoking_nonC])

# Only keeps records that are effectively with cardiovascular disease
ensanut_adult_smoking_do = ensanut_adult_smoking[~ensanut_adult_smoking.isin(ensanut_adult_smoking_non.to_dict('l')).all(1)]

# Total with cardiovascular disease in age cohort per sex (1M: 2:F), entity (State) and domain (1:urban and 2:rural)
ensanut_adult_smoking_summary, ensanut_adult_smoking_summary_sex = createSummary(ensanut_adult_smoking_do) 

print('Adults smoking data ready')
print('Adults data ready')
#%%
    #############################################
    #   Process information of available beds   #
    ############################################# 
    
health_df = pd.read_excel(raw_data + 'ESTABLECIMIENTO_SALUD_202002.xlsx')

# Keeps hospitals that are in operations 
health_df_op = health_df[health_df['ESTATUS DE OPERACION'] == 'EN OPERACION']

# Adds up all available beds & beds available only for hospitalization
total_beds_all = health_df_op.groupby(['NOMBRE DE LA ENTIDAD',
                          'CLAVE DE LA ENTIDAD',
                          'NOMBRE DEL MUNICIPIO',
                          'CLAVE DEL MUNICIPIO'])['TOTAL DE CAMAS'].sum().reset_index()

total_beds_hosp = health_df_op[health_df_op['NOMBRE TIPO ESTABLECIMIENTO'] == 'DE HOSPITALIZACIÓN']

total_beds_hosp = total_beds_hosp.groupby(['NOMBRE DE LA ENTIDAD',
                          'CLAVE DE LA ENTIDAD',
                          'NOMBRE DEL MUNICIPIO',
                          'CLAVE DEL MUNICIPIO'])['TOTAL DE CAMAS'].sum().reset_index()

print('Total beds data ready')

#%%
    #############################################
    #     Saves files to use in another code    #
    ############################################# 
    
age_cohort_score.to_csv(raw_data + 'age_cohort_score.csv')
pop_muni_cohort.to_csv(raw_data + 'pop_muni_cohort.csv')

ensanut_kid_cohort.to_csv(raw_data + 'ensanut_kid_cohort.csv')

ensanut_teen_smoking_summary.to_csv(raw_data + 'ensanut_teen_smoking_summary.csv')
ensanut_teen_smoking_summary_sex.to_csv(raw_data + 'ensanut_teen_smoking_summary_sex.csv')

ensanut_teen_obesity_summary.to_csv(raw_data + 'ensanut_teen_obesity_summary.csv')
ensanut_teen_obesity_summary_sex.to_csv(raw_data + 'ensanut_teen_obesity_summary_sex.csv')

ensanut_adult_obesity_summary.to_csv(raw_data + 'ensanut_adult_obesity_summary.csv')
ensanut_adult_obesity_summary_sex.to_csv(raw_data + 'ensanut_adult_obesity_summary_sex.csv')

ensanut_adult_diabetes_summary.to_csv(raw_data + 'ensanut_adult_diabetes_summary.csv')
ensanut_adult_diabetes_summary_sex.to_csv(raw_data + 'ensanut_adult_diabetes_summary_sex.csv')

ensanut_adult_hypertension_summary.to_csv(raw_data + 'ensanut_adult_hypertension_summary.csv') 
ensanut_adult_hypertension_summary_sex.to_csv(raw_data + 'ensanut_adult_hypertension_summary_sex.csv')

ensanut_adult_cardiovascular_summary.to_csv(raw_data + 'ensanut_adult_cardiovascular_summary.csv') 
ensanut_adult_cardiovascular_summary_sex.to_csv(raw_data + 'ensanut_adult_cardiovascular_summary_sex.csv')

ensanut_adult_renal_summary.to_csv(raw_data + 'ensanut_adult_renal_summary.csv') 
ensanut_adult_renal_summary_sex.to_csv(raw_data + 'ensanut_adult_renal_summary_sex.csv')

ensanut_adult_smoking_summary.to_csv(raw_data + 'ensanut_adult_smoking_summary.csv') 
ensanut_adult_smoking_summary_sex.to_csv(raw_data + 'ensanut_adult_smoking_summary_sex.csv') 

total_beds_all.to_csv(raw_data + 'total_beds_all.csv')
total_beds_hosp.to_csv(raw_data + 'total_beds_hosp.csv')
