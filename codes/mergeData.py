# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 11:34:47 2020

@author: Daniel Itzamna Avila Ortega
@Centre: Mexican Center of Industrial Ecology
@DOI: https://doi.org/10.22201/igg.25940694e.2020.2.73    
@Publication: An index of municipality-level vulnerability to COVID-19 in Mexico.
    
        This is the second code from the research on the Vulnerability Index 
        for municipalities in Mexico with regards to the COVID19.
"""

import gc
import numpy as np
import pandas as pd

ageb = '~/AGEB2018/datos/'                                          # AGEB2018 recovered from INEGI, Marco Geoestadístico: https://www.inegi.org.mx/temas/mg/#Descargas
raw_data = '~/RawData/'
clean_data = '~/CleanData/'

    #######################################
    #        Loads files to process       #
    #######################################
    
age_cohort_score = pd.read_csv(raw_data + 'age_cohort_score.csv', index_col=0)
pop_muni_cohort = pd.read_csv(raw_data + 'pop_muni_cohort.csv', index_col=0)

ensanut_kid_cohort = pd.read_csv(raw_data + 'ensanut_kid_cohort.csv', index_col=0)

ensanut_teen_smoking_summary = pd.read_csv(raw_data + 'ensanut_teen_smoking_summary.csv', index_col=0)
ensanut_teen_smoking_summary_sex = pd.read_csv(raw_data + 'ensanut_teen_smoking_summary_sex.csv', index_col=0)

ensanut_teen_obesity_summary = pd.read_csv(raw_data + 'ensanut_teen_obesity_summary.csv', index_col=0)
ensanut_teen_obesity_summary_sex = pd.read_csv(raw_data + 'ensanut_teen_obesity_summary_sex.csv', index_col=0)

ensanut_adult_obesity_summary = pd.read_csv(raw_data + 'ensanut_adult_obesity_summary.csv', index_col=0)
ensanut_adult_obesity_summary_sex = pd.read_csv(raw_data + 'ensanut_adult_obesity_summary_sex.csv', index_col=0)

ensanut_adult_diabetes_summary = pd.read_csv(raw_data + 'ensanut_adult_diabetes_summary.csv', index_col=0)
ensanut_adult_diabetes_summary_sex = pd.read_csv(raw_data + 'ensanut_adult_diabetes_summary_sex.csv', index_col=0)

ensanut_adult_hypertension_summary = pd.read_csv(raw_data + 'ensanut_adult_hypertension_summary.csv', index_col=0) 
ensanut_adult_hypertension_summary_sex = pd.read_csv(raw_data + 'ensanut_adult_hypertension_summary_sex.csv', index_col=0)

ensanut_adult_cardiovascular_summary = pd.read_csv(raw_data + 'ensanut_adult_cardiovascular_summary.csv', index_col=0) 
ensanut_adult_cardiovascular_summary_sex = pd.read_csv(raw_data + 'ensanut_adult_cardiovascular_summary_sex.csv', index_col=0)

ensanut_adult_renal_summary = pd.read_csv(raw_data + 'ensanut_adult_renal_summary.csv', index_col=0) 
ensanut_adult_renal_summary_sex = pd.read_csv(raw_data + 'ensanut_adult_renal_summary_sex.csv', index_col=0)

ensanut_adult_smoking_summary = pd.read_csv(raw_data + 'ensanut_adult_smoking_summary.csv', index_col=0) 
ensanut_adult_smoking_summary_sex = pd.read_csv(raw_data + 'ensanut_adult_smoking_summary_sex.csv', index_col=0) 

total_beds_all = pd.read_csv(raw_data + 'total_beds_all.csv', index_col=0)
total_beds_hosp = pd.read_csv(raw_data + 'total_beds_hosp.csv', index_col=0)

#%%
# Harmonize dataframes
ensanut_kid_cohort['RANGO'] = '0 a 9'
ensanut_kid_cohort_tot = ensanut_kid_cohort.groupby(['RANGO', 'SEXO', 'ENT', 'DOMINIO'])['TOTAL'].sum().reset_index()

ensanut_teen_obesity_summary_sex['RANGO'] = '10 a 19'
ensanut_teen_smoking_summary_sex['RANGO'] = '10 a 19'

ensanut_teen_obesity_summary_sex = ensanut_teen_obesity_summary_sex.rename(columns={'ENT':'CLAVE_ENT'})
ensanut_teen_smoking_summary_sex = ensanut_teen_smoking_summary_sex.rename(columns={'ENT':'CLAVE_ENT'})

ensanut_adult_obesity_summary = ensanut_adult_obesity_summary.rename(columns={'EDAD':'RANGO', 'ENT':'CLAVE_ENT'})
ensanut_adult_diabetes_summary = ensanut_adult_diabetes_summary.rename(columns={'EDAD':'RANGO', 'ENT':'CLAVE_ENT'})
ensanut_adult_hypertension_summary = ensanut_adult_hypertension_summary.rename(columns={'EDAD':'RANGO', 'ENT':'CLAVE_ENT'})
ensanut_adult_cardiovascular_summary = ensanut_adult_cardiovascular_summary.rename(columns={'EDAD':'RANGO', 'ENT':'CLAVE_ENT'})
ensanut_adult_renal_summary = ensanut_adult_renal_summary.rename(columns={'EDAD':'RANGO', 'ENT':'CLAVE_ENT'})
ensanut_adult_smoking_summary = ensanut_adult_smoking_summary.rename(columns={'EDAD':'RANGO', 'ENT':'CLAVE_ENT'})

# Comorbolities in teenagers and adults
comorbility_obesity = pd.concat([ensanut_teen_obesity_summary_sex, 
                                 ensanut_adult_obesity_summary], sort=False)

comorbility_smoking = pd.concat([ensanut_teen_smoking_summary_sex, 
                                 ensanut_adult_smoking_summary], sort=False)

# Harmonize sex for all dataframes
sex = {1:'Hombres', 2:'Mujeres'}
ensanut_kid_cohort_tot['SEXO'] = ensanut_kid_cohort_tot['SEXO'].replace(sex)

comorbility_obesity['SEXO'] = comorbility_obesity['SEXO'].replace(sex)
comorbility_smoking['SEXO'] = comorbility_smoking['SEXO'].replace(sex)

ensanut_adult_diabetes_summary['SEXO'] = ensanut_adult_diabetes_summary['SEXO'].replace(sex)
ensanut_adult_hypertension_summary['SEXO'] = ensanut_adult_hypertension_summary['SEXO'].replace(sex)
ensanut_adult_cardiovascular_summary['SEXO'] = ensanut_adult_cardiovascular_summary['SEXO'].replace(sex)
ensanut_adult_renal_summary['SEXO'] = ensanut_adult_renal_summary['SEXO'].replace(sex)
#%%

# Standardizes "Unique code" for five digits
pop_muni_cohort['CLAVE'] = pop_muni_cohort['CLAVE'].map('{:0>5}'.format)

# Groups by different keys to get total population within range in keys
pop_muni_cohort_tot = pop_muni_cohort.groupby(['NOM_ENT', 'CLAVE_ENT', 'RANGO'])['POB'].sum().reset_index()
pop_muni_cohort_sex = pop_muni_cohort.groupby(['NOM_ENT', 'CLAVE_ENT', 'RANGO', 'SEXO'])['POB'].sum().reset_index()

# Merges population in municipalities with totals with range, dividing them as to get pop-share per municipality and age cohort 
pop_muni_cohort_variables = pop_muni_cohort.merge(pop_muni_cohort_tot, on=['NOM_ENT', 'CLAVE_ENT', 'RANGO'], how='left')
pop_muni_cohort_variables['%-POP-MUN-AGE'] = (pop_muni_cohort_variables[['POB_x']].div(pop_muni_cohort_variables.pop('POB_y'), axis=0))

pop_muni_cohort_variables = pop_muni_cohort_variables.merge(pop_muni_cohort_sex, on=['NOM_ENT', 'CLAVE_ENT', 'RANGO', 'SEXO'], how='left')
pop_muni_cohort_variables['%-POP-STATE-AGE-SEX'] = (pop_muni_cohort_variables[['POB_x']].div(pop_muni_cohort_variables.pop('POB'), axis=0))

#%%
obesity = pop_muni_cohort_variables.merge(comorbility_obesity, on=['CLAVE_ENT', 'RANGO', 'SEXO'], how='left')
obesity.fillna(0, inplace=True)
obesity['POP-Obesity'] = obesity['%-POP-STATE-AGE-SEX'].mul(obesity['TOTAL']).round(0)
del obesity['TOTAL']

smoking = pd.merge(pop_muni_cohort_variables, comorbility_smoking, on=['CLAVE_ENT', 'RANGO', 'SEXO'], how='left')
smoking.fillna(0, inplace=True)
smoking['POP-Smoking'] = smoking['%-POP-STATE-AGE-SEX'].mul(smoking['TOTAL']).round(0)
del smoking['TOTAL']

diabetes = pop_muni_cohort_variables.merge(ensanut_adult_diabetes_summary, on=['CLAVE_ENT', 'RANGO', 'SEXO'], how='left')
diabetes.fillna(0, inplace=True)
diabetes['POP-Diabetes'] = diabetes['%-POP-STATE-AGE-SEX'].mul(diabetes['TOTAL']).round(0)
del diabetes['TOTAL']

hypertension = pop_muni_cohort_variables.merge(ensanut_adult_hypertension_summary, on=['CLAVE_ENT', 'RANGO', 'SEXO'], how='left')
hypertension.fillna(0, inplace=True)
hypertension['POP-Hypertension'] = hypertension['%-POP-STATE-AGE-SEX'].mul(hypertension['TOTAL']).round(0)
del hypertension['TOTAL']

cardiovascular = pop_muni_cohort_variables.merge(ensanut_adult_cardiovascular_summary, on=['CLAVE_ENT', 'RANGO', 'SEXO'], how='left')
cardiovascular.fillna(0, inplace=True)
cardiovascular['POP-Cardiovascular'] = cardiovascular['%-POP-STATE-AGE-SEX'].mul(cardiovascular['TOTAL']).round(0)
del cardiovascular['TOTAL']

renal = pop_muni_cohort_variables.merge(ensanut_adult_renal_summary, on=['CLAVE_ENT', 'RANGO', 'SEXO'], how='left')
renal.fillna(0, inplace=True)
renal['POP-Renal'] = renal['%-POP-STATE-AGE-SEX'].mul(renal['TOTAL']).round(0)
del renal['TOTAL']

#%%

    #######################################
    #         Save files to process       #
    #######################################
    
pop_muni_cohort.to_csv(clean_data + 'pop_muni_cohort.csv')
obesity.to_csv(clean_data + 'obesity.csv')
smoking.to_csv(clean_data + 'smoking.csv')
diabetes.to_csv(clean_data + 'diabetes.csv')
hypertension.to_csv(clean_data + 'hypertension.csv')
cardiovascular.to_csv(clean_data + 'cardiovascular.csv')
renal.to_csv(clean_data + 'renal.csv')
