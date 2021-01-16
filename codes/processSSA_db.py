# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:02:22 2020

@author: DanielAvila
"""

import functools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

raw_data = 'D:/Users/Daniel_Avila/Documents/CMEI/Proyectos/Covid19/Articulo/data/SSA_DB/'
clean_data = 'D:/Users/Daniel_Avila/Documents/CMEI/Proyectos/Covid19/Articulo/data/clean/'

rawData = pd.read_csv(raw_data + '200819COVID19MEXICO.csv', header='infer', encoding='ANSI')

dataKeep = rawData.loc[:,['SECTOR', 'ENTIDAD_UM', 'SEXO', 'ENTIDAD_RES',
                          'MUNICIPIO_RES', 'TIPO_PACIENTE', 'FECHA_INGRESO', 'FECHA_SINTOMAS',
                          'FECHA_DEF', 'INTUBADO', 'NEUMONIA', 'EDAD', 'EMBARAZO', 'DIABETES',
                          'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR',
                          'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'RESULTADO', 'UCI']]


dataKeep['ENTIDAD_RES'] = dataKeep['ENTIDAD_RES'].map('{:0>2}'.format)
dataKeep['MUNICIPIO_RES'] = dataKeep['MUNICIPIO_RES'].map('{:0>3}'.format)
dataKeep['CVEGEO'] = dataKeep['ENTIDAD_RES'] + dataKeep['MUNICIPIO_RES']

#%%
covid = dataKeep.loc[:, ['SECTOR', 'ENTIDAD_UM', 'SEXO', 'ENTIDAD_RES',
                          'MUNICIPIO_RES', 'TIPO_PACIENTE', 'FECHA_INGRESO', 'FECHA_SINTOMAS',
                          'FECHA_DEF', 'INTUBADO', 'NEUMONIA', 'EDAD', 'EMBARAZO', 'DIABETES',
                          'EPOC', 'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR',
                          'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'RESULTADO', 'UCI',
                          'CVEGEO']]

covid = covid[covid['RESULTADO'] == 1]
covid['CONTEO'] = 1
#%%
# Sex distinction
covid_sex = covid.groupby('SEXO')['CONTEO'].sum()
#%%
covid_age = covid.loc[:,['CVEGEO', 'EDAD', 'CONTEO']]

age_dict_cohort = {'(-1, 9]': '0 a 9',
                   '(9, 19]': '10 a 19',
                   '(19, 29]': '20 a 29', 
                    '(29, 39]':'30 a 39', 
                    '(39, 49]':'40 a 49', 
                    '(49, 59]':'50 a 59',
                    '(59, 69]':'60 y mas',
                    '(69, 79]':'60 y mas',
                    '(79, 89]':'60 y mas',
                    '(89, 99]':'60 y mas',
                    '(99, 109]':'60 y mas',
                    '(109, 119]':'60 y mas'}

covid_age = covid_age.groupby([pd.cut(covid_age['EDAD'], np.arange(-1, 65 + 60, 10)),
                                     'CVEGEO'])['CONTEO'].sum().reset_index()   


covid_age['EDAD'] = covid_age['EDAD'].astype(str)
covid_age['EDAD'] = covid_age['EDAD'].replace(age_dict_cohort)

covid_age = covid_age.groupby(['CVEGEO', 'EDAD'])['CONTEO'].sum().reset_index()

covid_age.rename(columns={'CONTEO': 'CASOS'}, inplace=True)

covid_age = covid_age.pivot(index='CVEGEO', columns='EDAD', values='CASOS')
#%% 
# 1: Counts how many had the comorbidites of interest (all),
#           how many had other comorbidites
def createSummaryShort(dataframe, factor="", name="", condition=True):
    """ Creates a summary of a given dataframe based on requested columns.
        It takes as arguements:
            - a dataframe
            - two strings:
                - column of interest to process
                - name for the resulting column
        """
    
    if condition == True:
    # Cuts the dataframe in chunks grouping by the the chunks and the specified columns
    # adding up a given column
        vIndex_factor = dataframe.groupby(['CVEGEO', factor])['CONTEO'].sum().reset_index()   
    else:
        vIndex_factor = dataframe.groupby(['CVEGEO'])['CONTEO'].sum().reset_index()   

    # Renames column
    vIndex_factor.rename(columns={factor:'KEY', 'CONTEO': name}, inplace=True)
    
    return vIndex_factor  

# Comorbidities of interest
covid_renal_short = createSummaryShort(covid, 'RENAL_CRONICA', 'TOTAL-R', True)
covid_smoking_short = createSummaryShort(covid, 'TABAQUISMO', 'TOTAL-S', True) 
covid_obesity_short = createSummaryShort(covid, 'OBESIDAD', 'TOTAL-O', True) 
covid_diabetes_short = createSummaryShort(covid, 'DIABETES', 'TOTAL-D', True) 
covid_hipertension_short = createSummaryShort(covid, 'HIPERTENSION', 'TOTAL-H', True) 
covid_cardiovascular_short = createSummaryShort(covid, 'CARDIOVASCULAR', 'TOTAL-C', True) 


# Does a quick summary not sorting by age nor sex
data_frames = [covid_renal_short, covid_smoking_short, covid_obesity_short, 
               covid_diabetes_short, covid_hipertension_short, covid_cardiovascular_short]

# Counts per municipality how many had the comorbidities of interest
covid_commorbidity_vi = df_merged = functools.reduce(lambda  left,right: 
                                     pd.merge(left,right, 
                                            on=['CVEGEO', 'KEY'],
                                            how='outer'), data_frames).fillna(0)

# Comorbidities not of interest
covid_neumonia_short = createSummaryShort(covid, 'NEUMONIA', 'TOTAL-N', True) 
covid_epoc_short = createSummaryShort(covid, 'EPOC', 'TOTAL-E', True) 
covid_asma_short = createSummaryShort(covid, 'ASMA', 'TOTAL-A', True) 
covid_inmu_short = createSummaryShort(covid, 'INMUSUPR', 'TOTAL-I', True) 
covid_other_short = createSummaryShort(covid, 'OTRA_COM', 'TOTAL-OC', True)

# Does a quick summary not sorting by age nor sex
data_frames_other = [covid_neumonia_short, covid_epoc_short, covid_asma_short, 
                     covid_inmu_short, covid_other_short]

# Counts per municipality how many had the comorbidities 
covid_commorbidity_other = df_merged = functools.reduce(lambda  left,right: 
                                     pd.merge(left,right, 
                                            on=['CVEGEO', 'KEY'],
                                            how='outer'), data_frames_other).fillna(0)

covid_commorbidities = covid_commorbidity_vi.merge(covid_commorbidity_other,
                                                  on=['CVEGEO', 'KEY'],
                                                  how='left').fillna(0)
#%%
# 2: Gets what was the most common commorbidity per municipality
covid_commorbidity_vi['COMMON'] = covid_commorbidity_vi.iloc[:,2:].columns[covid_commorbidity_vi.iloc[:,2:].values.argsort(1)[:, -1]]

covid_commorbidity_other['COMMON'] = covid_commorbidity_other.iloc[:,2:].columns[covid_commorbidity_other.iloc[:,2:].values.argsort(1)[:, -1]]

covid_commorbidities['COMMON'] = covid_commorbidities.iloc[:,2:].columns[covid_commorbidities.iloc[:,2:].values.argsort(1)[:, -1]] 
#%%
# 3: Counts what was the most common commorbidity per municipality in death people 
covid_death = covid[covid['FECHA_DEF'] != '9999-99-99']

# Comorbidities of interest
covid_death_renal_short = createSummaryShort(covid_death, 'RENAL_CRONICA', 'TOTAL-R', True)
covid_death_smoking_short = createSummaryShort(covid_death, 'TABAQUISMO', 'TOTAL-S', True) 
covid_death_obesity_short = createSummaryShort(covid_death, 'OBESIDAD', 'TOTAL-O', True) 
covid_death_diabetes_short = createSummaryShort(covid_death, 'DIABETES', 'TOTAL-D', True) 
covid_death_hipertension_short = createSummaryShort(covid_death, 'HIPERTENSION', 'TOTAL-H', True) 
covid_death_cardiovascular_short = createSummaryShort(covid_death, 'CARDIOVASCULAR', 'TOTAL-C', True) 

# Does a quick summary not sorting by age nor sex
data_frames_death = [covid_death_renal_short, covid_death_smoking_short, 
                     covid_death_obesity_short, covid_death_diabetes_short, 
                     covid_death_hipertension_short, covid_death_cardiovascular_short]

# Counts per municipality how many had the comorbidities of interest
covid_death_commorbidity_vi = functools.reduce(lambda  left,right: 
                                     pd.merge(left,right, 
                                            on=['CVEGEO', 'KEY'],
                                            how='outer'), data_frames_death).fillna(0)

# Comorbidities not of interest
covid_death_neumonia_short = createSummaryShort(covid_death, 'NEUMONIA', 'TOTAL-N', True) 
covid_death_epoc_short = createSummaryShort(covid_death, 'EPOC', 'TOTAL-E', True) 
covid_death_asma_short = createSummaryShort(covid_death, 'ASMA', 'TOTAL-A', True) 
covid_death_inmu_short = createSummaryShort(covid_death, 'INMUSUPR', 'TOTAL-I', True) 
covid_death_other_short = createSummaryShort(covid_death, 'OTRA_COM', 'TOTAL-OC', True)

# Does a quick summary not sorting by age nor sex
data_frames_death_other = [covid_death_neumonia_short, covid_death_epoc_short, 
                            covid_death_asma_short, covid_death_inmu_short, 
                            covid_death_other_short]

# Counts per municipality how many had the comorbidities 
covid_death_commorbidity_other = functools.reduce(lambda  left,right: 
                                     pd.merge(left,right, 
                                            on=['CVEGEO', 'KEY'],
                                            how='outer'), data_frames_death_other).fillna(0)

covid_death_commorbidities = covid_death_commorbidity_vi.merge(covid_death_commorbidity_other,
                                                  on=['CVEGEO', 'KEY'],
                                                  how='left').fillna(0)

covid_death_commorbidity_vi['COMMON'] = covid_death_commorbidity_vi.iloc[:,2:].columns[covid_death_commorbidity_vi.iloc[:,2:].values.argsort(1)[:, -1]]

covid_death_commorbidity_other['COMMON'] = covid_death_commorbidity_other.iloc[:,2:].columns[covid_death_commorbidity_other.iloc[:,2:].values.argsort(1)[:, -1]]

covid_death_commorbidities['COMMON'] = covid_death_commorbidities.iloc[:,2:].columns[covid_death_commorbidities.iloc[:,2:].values.argsort(1)[:, -1]] 


# Gets summary 
covid_death_summary = covid_death_commorbidities[covid_death_commorbidities['KEY'] == 1]
covid_death_summary = covid_death_summary.sum()
#%%
# 4: Arranges data for correlation analysis

# Selects COVID19 cases that effectively have a given commorbidity
covid_commorbidity_vi_yes = covid_commorbidity_vi[covid_commorbidity_vi['KEY'] == 1]
del covid_commorbidity_vi_yes['KEY']
del covid_commorbidity_vi_yes['COMMON']
covid_commorbidity_vi_yes.set_index('CVEGEO', inplace=True)

# Selects COVID19 cases-deaths that effectively have a given commorbidity
covid_death_commorbidity_vi_yes = covid_death_commorbidity_vi[covid_death_commorbidity_vi['KEY'] == 1]
del covid_death_commorbidity_vi_yes['KEY']
del covid_death_commorbidity_vi_yes['COMMON']
covid_death_commorbidity_vi_yes.set_index('CVEGEO', inplace=True)

# Gets total number of COVID19 cases per municipality and renames column
covid_mun = pd.DataFrame(covid.groupby(['CVEGEO'])['CONTEO'].sum())
covid_mun.rename(columns={'CONTEO':'CASOS'},  inplace=True)


# Loads data with vulnerability index, population, commorbidities
vi_data = pd.read_csv(clean_data + 'vi_commorbities.csv', encoding='ANSI')
vi_data_keep = ['CVEGEO', 'NOM_ENT', 'MUN', 'V_index', 'POB', 
                'POP-Obesity', 'POP-Smoking', 'POP-Diabetes', 'POP-Hypertension',
                'POP-Cardiovascular', 'POP-Renal']

# Keeps data of interest, maps to keep string consistency and sets index
vi_data_interest = vi_data.loc[:,vi_data_keep]
vi_data_interest['CVEGEO'] = vi_data_interest['CVEGEO'].map('{:0>5}'.format)
vi_data_interest.set_index(['CVEGEO'], inplace=True)

#%%
# 5: Normalizes data of interest to further correlation analysis with VI

# Matches municipalities that have COVID19 cases with VI-data
covid_mun_pop = covid_mun.merge(vi_data_interest['POB'], left_index=True, right_index=True)

# Returns non-matching municipalities
df_non_match = pd.merge(covid_mun, vi_data_interest, how='outer', indicator=True, on ='CVEGEO')
df_non_match = df_non_match[(df_non_match._merge == 'left_only')]

# Normalizes each municipality cases' in terms of population
covid_mun_pop['NORM-1'] = covid_mun_pop['CASOS'].div(covid_mun_pop['POB']).round(6)

# Normalizes each municipality cases commorbities' in terms of cases
covid_mun_pop_commorbidity = covid_mun_pop.copy()
covid_mun_pop_commorbidity = covid_mun_pop_commorbidity.merge(covid_commorbidity_vi_yes, left_index=True, right_index=True) 

for i in covid_mun_pop_commorbidity.iloc[:,3:]:
    covid_mun_pop_commorbidity['NORM' + '-' + i] = covid_mun_pop_commorbidity[i].div(covid_mun_pop_commorbidity['CASOS']).round(6)

#%%
# Merges with Vulnerability index
covid_mun_pop = covid_mun_pop.merge(vi_data_interest, left_index=True, right_index=True)
del covid_mun_pop['POB_x']
covid_mun_pop.rename(columns={'POB_y':'POB'},  inplace=True)


covid_mun_pop_commorbidity = covid_mun_pop_commorbidity.merge(vi_data_interest, left_index=True, right_index=True)
del covid_mun_pop_commorbidity['POB_x']
covid_mun_pop_commorbidity.rename(columns={'POB_y':'POB'},  inplace=True)
#%%    
# Applies expansion factor for modeling purposes
#covid_mun_pop_exp = covid_mun_pop.copy()
#covid_mun_pop_exp['CASOS'] = covid_mun_pop_exp['CASOS'].mul(4)
#covid_mun_pop_exp['NORM-EXP'] = covid_mun_pop_exp['CASOS'].div(covid_mun_pop_exp['POB']).round(6)
#%%    

# 6: Correlation analysis
covid_mun_pop_corr = covid_mun_pop.loc[:,['V_index', 'NORM-1']]

covid_mun_pop_com_corr = covid_mun_pop_commorbidity.loc[:, ['V_index', 
                                                 'NORM-TOTAL-R', 'NORM-TOTAL-S',
                                                 'NORM-TOTAL-O', 'NORM-TOTAL-D',
                                                 'NORM-TOTAL-H', 'NORM-TOTAL-C']]

#covid_mun_pop_exp = covid_mun_pop_exp.loc[:,['V_index', 'NORM-EXP']]


covid_mun_pop_corr.to_csv(clean_data + 'covid_mun_pop_corr.csv')
covid_mun_pop_com_corr.to_csv(clean_data + 'covid_mun_pop_com_corr.csv')


#%%

#covid_Index.to_csv(clean_data + 'covid_SSA.csv', encoding='UTF-8')