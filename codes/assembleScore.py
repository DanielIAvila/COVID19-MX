# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:29:37 2020

@author: Daniel Itzamna Avila Ortega
@Institution: Mexican Center of Industrial Ecology
@DOI: https://doi.org/10.22201/igg.25940694e.2020.2.73    
@Publication: An index of municipality-level vulnerability to COVID-19 in Mexico.
    
        This is the third code from the research on the Vulnerability Index 
        for municipalities in Mexico with regards to the COVID19.
"""

import gc
import numpy as np
import pandas as pd

    #######################################
    #        Loads files to process       #
    #######################################

clean_data = '~/CleanData/'

# Loads files
pop_muni_cohort = pd.read_csv(clean_data + 'pop_muni_cohort.csv', index_col=0)

# Load comorbility risks
age_cohort_score = pd.read_csv(clean_data + 'age_cohort_score.csv', index_col=0)
obesity = pd.read_csv(clean_data + 'obesity.csv', index_col=0)
smoking = pd.read_csv(clean_data + 'smoking.csv', index_col=0)
diabetes = pd.read_csv(clean_data + 'diabetes.csv', index_col=0)
hypertension = pd.read_csv(clean_data + 'hypertension.csv', index_col=0)
cardiovascular = pd.read_csv(clean_data + 'cardiovascular.csv', index_col=0)
renal = pd.read_csv(clean_data + 'renal.csv', index_col=0)

#%%
# Commorbities percentage
obesity['%-Obesity-Mun'] = obesity['POP-Obesity'].div(obesity['POB_x'])
smoking['%-Smoking-Mun'] = smoking['POP-Smoking'].div(smoking['POB_x'])
diabetes['%-Diabetes-Mun'] = diabetes['POP-Diabetes'].div(diabetes['POB_x'])
hypertension['%-Hypertension-Mun'] = hypertension['POP-Hypertension'].div(hypertension['POB_x'])
cardiovascular['%-Cardiovascular-Mun'] = cardiovascular['POP-Cardiovascular'].div(cardiovascular['POB_x'])
renal['%-Renal-Mun'] = renal['POP-Renal'].div(renal['POB_x'])

#%%
def stats(dataframe, column='', comorbidity=''):
    """ This function calculates the statistics of a given dataframe.
        """
    
    local = dataframe.copy()
    local_a = local.groupby(['CLAVE','NOM_ENT','CLAVE_ENT', 'MUN','SEXO','RANGO'])[column].sum().reset_index()
    
    local_stats = local_a.groupby(['SEXO', 'RANGO'])[column].describe(include='all').reset_index()
    local_stats['Comorbity'] = comorbidity
    
    local_b = local_a.merge(local_stats, on=['SEXO', 'RANGO'], how='left')
      
    gc.collect()
    return local_b, local_stats

obesity_stats, obesity_descriptive = stats(obesity, 'POP-Obesity','obesity')
smoking_stats, smoking_descriptive = stats(smoking, 'POP-Smoking','smoking')
diabetes_stats, diabetes_descriptive = stats(diabetes, 'POP-Diabetes','diabetes')
hypertension_stats, hypertension_descriptive = stats(hypertension, 'POP-Hypertension','hypertension')
cardiovascular_stats, cardiovascular_descriptive = stats(cardiovascular, 'POP-Cardiovascular','cardiovascular')
renal_stats, renal_descriptive = stats(renal, 'POP-Renal','renal')
#%%
        
def categorize(dataframe, column='', score=''):
    """ This function categorizes the each municipality according with their 
        statistics. """

    conditions = [
            (dataframe[column] > dataframe['25%']) & (dataframe[column] > dataframe['50%']) & (dataframe[column] > dataframe['75%']),
            (dataframe[column] > dataframe['25%']) & (dataframe[column] > dataframe['50%']) & (dataframe[column] < dataframe['75%']),
            (dataframe[column] > dataframe['25%']) & (dataframe[column] < dataframe['50%']),
            (dataframe[column] > dataframe['min']) & (dataframe[column] < dataframe['25%']),
            (dataframe[column] == 0)]
    
    choices = [1, 0.75, 0.5, 0.25, 0]
    
    dataframe[score] = np.select(conditions, choices)
    
    return dataframe

obesity_stats = categorize(obesity_stats, 'POP-Obesity', 'S_obesity')
smoking_stats = categorize(smoking_stats, 'POP-Smoking', 'S_smoking')
diabetes_stats = categorize(diabetes_stats, 'POP-Diabetes', 'S_diabetes')
hypertension_stats = categorize(hypertension_stats, 'POP-Hypertension', 'S_hypertension')
cardiovascular_stats = categorize(cardiovascular_stats, 'POP-Cardiovascular', 'S_cardiovascular')
renal_stats = categorize(renal_stats, 'POP-Renal', 'S_renal')

#%%
# Save information to an Excel document
writer = pd.ExcelWriter(clean_data + 'ComorbitiesStatsV2.xlsx', engine='xlsxwriter')
    
obesity_stats.to_excel(writer, sheet_name='obesity_stats', index=False)
smoking_stats.to_excel(writer, sheet_name='smoking_stats', index=False)
diabetes_stats.to_excel(writer, sheet_name='diabetes_stats', index=False)
hypertension_stats.to_excel(writer, sheet_name='hypertension_stats', index=False)
cardiovascular_stats.to_excel(writer, sheet_name='cardiovascular_stats', index=False)
renal_stats.to_excel(writer, sheet_name='renal_stats', index=False)

writer.save()

#%%

age_cohort_score.reset_index(inplace=True)
age_cohort_score = age_cohort_score.rename(columns={'index':'RANGO'})
del age_cohort_score['Total']

# Merges age cohort score to all population per municipality
pop_muni_cohort_score = pop_muni_cohort.merge(age_cohort_score, on=['RANGO'], how='left')

condition_sex = [pop_muni_cohort_score['SEXO'] == 'Hombres', pop_muni_cohort_score['SEXO'] == 'Mujeres']
choice_sex = [0.67, 0.33]

# Assigns score per sex to dataframe
pop_muni_cohort_score['S-sex'] = np.select(condition_sex, choice_sex)

#%%
on = ['CLAVE', 'NOM_ENT', 'CLAVE_ENT', 'MUN', 'SEXO', 'RANGO']
drop = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']

vulnerability_data = pop_muni_cohort_score.merge(obesity_stats, on=on, how='left')
vulnerability_data = vulnerability_data.drop(drop, axis=1)

#%%
vulnerability_data = vulnerability_data.merge(smoking_stats, on=on, how='left')
vulnerability_data = vulnerability_data.drop(drop, axis=1)

#%%
vulnerability_data = vulnerability_data.merge(diabetes_stats,on=on, how='left')
vulnerability_data = vulnerability_data.drop(drop, axis=1)

#%%
vulnerability_data = vulnerability_data.merge(hypertension_stats, on=on, how='left')
vulnerability_data = vulnerability_data.drop(drop, axis=1)

#%%
vulnerability_data = vulnerability_data.merge(cardiovascular_stats, on=on, how='left')
vulnerability_data = vulnerability_data.drop(drop, axis=1)
#%%
vulnerability_data = vulnerability_data.merge(renal_stats, on=on, how='left')
vulnerability_data = vulnerability_data.drop(drop, axis=1)


#%%

# Calculates vulnerability index
vulnerability_index = vulnerability_data.loc[:,['CLAVE', 'NOM_ENT', 'CLAVE_ENT', 'MUN', 'POB', 'SEXO', 'S-sex',
                                                'RANGO', 'Score_age', 
                                                'POP-Obesity', 'S_obesity', 
                                                'POP-Smoking', 'S_smoking',
                                                'POP-Diabetes', 'S_diabetes',
                                                'POP-Hypertension', 'S_hypertension',
                                                'POP-Cardiovascular', 'S_cardiovascular',
                                                'POP-Renal', 'S_renal']]

# Gets scores of interest to calculate comorbility index
scores = vulnerability_index.loc[:,['Score_age', 'S_obesity', 'S_smoking', 
                                    'S_diabetes', 'S_hypertension', 'S_cardiovascular', 'S_renal']]

# Calculates comorbility index
vulnerability_index['C_index'] = scores.mean(axis=1)

# Calculates comorbility index based on sex
vulnerability_index['C_sex_index'] = vulnerability_index['C_index'].mul(vulnerability_index['S-sex'])

# Maps 'CLAVE' to have five digits
vulnerability_index['CLAVE'] = vulnerability_index['CLAVE'].map('{:0>5}'.format)

# Groups by variables adding up the 'S-sex' and 'C_sex_index'
sex_index_average = vulnerability_index.groupby(['CLAVE', 'NOM_ENT', 'CLAVE_ENT', 'MUN'])['S-sex'].sum().reset_index()
vulnerability_index_summary = vulnerability_index.groupby(['CLAVE', 'NOM_ENT', 'CLAVE_ENT', 'MUN'])['C_sex_index'].sum().reset_index()

# Merges the 'sex index' with the comorbility index, calculating the 'Vulnerability index' by weighted average
vulnerability_index_summary = vulnerability_index_summary.merge(sex_index_average, on=['CLAVE', 'NOM_ENT', 'CLAVE_ENT', 'MUN'], how='left')
vulnerability_index_summary['V_index'] = vulnerability_index_summary['C_sex_index'].div(vulnerability_index_summary['S-sex'])

# Dataframe for graphs in R with ggplot
vulnerability_index_graphs = pd.melt(vulnerability_index, 
                                     id_vars=['CLAVE', 'NOM_ENT', 'CLAVE_ENT', 'MUN', 'POB', 'SEXO', 'RANGO'],
                                     value_vars=['Score_age','S-sex','S_obesity','S_smoking',
                                                 'S_diabetes','S_hypertension','S_cardiovascular','S_renal',
                                                 'C_sex_index'],
                                     var_name='S-Index', value_name='Score')

vulnerability_pop_graphs = pd.melt(vulnerability_index, 
                                   id_vars=['CLAVE', 'NOM_ENT', 'CLAVE_ENT', 'MUN', 'POB', 'SEXO', 'RANGO'],
                                   value_vars=['POP-Obesity', 'POP-Smoking', 'POP-Diabetes', 
                                               'POP-Hypertension', 'POP-Cardiovascular', 'POP-Renal'],
                                   var_name='Pop-type', value_name='Population')

#%%

# Factors lack of access to health services
poverty_ic_asalud_summary = pd.read_excel(clean_data + 'poverty_age_cohort.xlsx', sheet_name='poverty_ic_asalud_summary')

# Harmonizes dataframe to be compared against others
sex_dict = {1:'Hombres', 2:'Mujeres'}
poverty_ic_asalud_summary['sexo'] = poverty_ic_asalud_summary['sexo'].replace(sex_dict)
poverty_ic_asalud_summary['ubica_geo'] = poverty_ic_asalud_summary['ubica_geo'].astype(str)
poverty_ic_asalud_summary['ubica_geo'] = poverty_ic_asalud_summary['ubica_geo'].map('{:0>5}'.format)
poverty_ic_asalud_summary = poverty_ic_asalud_summary.rename(columns={'sexo':'SEXO', 'ubica_geo':'CLAVE',
                                                                      'edad':'RANGO'})
# Gets population with no access to health services and groups by variables as to get total population
poverty_ic_asalud = poverty_ic_asalud_summary[poverty_ic_asalud_summary['ic_asalud'] == 1]
poverty_ic_asalud = poverty_ic_asalud.groupby(['CLAVE', 'SEXO', 'RANGO'])['population'].sum().reset_index()
poverty_ic_asalud_stats = poverty_ic_asalud.groupby(['SEXO', 'RANGO'])['population'].describe(include='all').reset_index()

# Merges ic_asalud with their corresponding stats
poverty_ic_asalud_factor = poverty_ic_asalud.merge(poverty_ic_asalud_stats, on=['SEXO', 'RANGO'], how='left')

# Assigns health access index factor based on median from all municipalities in the country
poverty_ic_asalud_factor = categorize(poverty_ic_asalud_factor, 'population', 'S_health_ac')

vulnerability_poverty_index = vulnerability_index.merge(poverty_ic_asalud_factor, on=['CLAVE', 'SEXO', 'RANGO'], how='left')
vulnerability_poverty_index = vulnerability_poverty_index.drop(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'], axis=1)

# Gets average S_health_ac, in order to populate missing values
vulnerability_poverty_index_average = vulnerability_poverty_index['S_health_ac'].median(axis=0)
vulnerability_poverty_index['S_health_ac'].fillna(vulnerability_poverty_index_average, inplace=True)

# Calculates average access to health services index per municipality
vulnerability_poverty_index_summary = vulnerability_poverty_index.groupby(['CLAVE', 'NOM_ENT', 'CLAVE_ENT',
                                                                           'MUN'])['S_health_ac'].mean().reset_index()

# Merges vulnerability index with access to health services index
vulnerability_index_full = vulnerability_index_summary.merge(vulnerability_poverty_index_summary,
                                                             on=['CLAVE', 'NOM_ENT', 'CLAVE_ENT',
                                                                           'MUN'], how='left')

# Erase columns of non-interest    
vulnerability_index_full = vulnerability_index_full.drop(['C_sex_index', 'S-sex'], axis=1)
  
vulnerability_index_full['V_A_index'] = (vulnerability_index_full['V_index'] + vulnerability_index_full['S_health_ac']) / 2  

#%% 
# Available beds
total_beds_hosp = pd.read_csv(clean_data + 'total_beds_hosp.csv', index_col=0)

# Formats columns to future merge with population per municipality
total_beds_hosp['CVE_ENT'] = total_beds_hosp['CVE_ENT'].astype(str)
total_beds_hosp['CVE_ENT'] = total_beds_hosp['CVE_ENT'].map('{:0>2}'.format)

total_beds_hosp['CVE_MUN'] = total_beds_hosp['CVE_MUN'].astype(str)
total_beds_hosp['CVE_MUN'] = total_beds_hosp['CVE_MUN'].map('{:0>3}'.format)

total_beds_hosp['CVEGEO'] = total_beds_hosp['CVE_ENT'] + total_beds_hosp['CVE_MUN']

# Calculates total beds per municipality
total_beds_hosp_summary = total_beds_hosp.groupby(['CVEGEO'])['TOTAL DE CAMAS'].sum().reset_index()

# Formats columns and group by municipality to get total population
pop_muni_cohort['CLAVE'] = pop_muni_cohort['CLAVE'].astype(str)
pop_muni_cohort['CLAVE'] = pop_muni_cohort['CLAVE'].map('{:0>5}'.format)
pop_muni_cohort = pop_muni_cohort.rename(columns={'CLAVE':'CVEGEO'}) 
pop_muni = pop_muni_cohort.groupby(['CVEGEO'])['POB'].sum().reset_index()

# Merges population per municipality with total beds, to get available beds x 1k habitants
total_beds_hosp_summary = pop_muni.merge(total_beds_hosp_summary, on=['CVEGEO'], how='left')
total_beds_hosp_summary.fillna(0, inplace=True)
total_beds_hosp_summary['CAMAS_1000hab'] = total_beds_hosp_summary['TOTAL DE CAMAS'].div(total_beds_hosp_summary['POB']).round(5)

# Renames column and merges with total available beds per municipality
vulnerability_index_full = vulnerability_index_full.rename(columns={'CLAVE':'CVEGEO'}) 
vulnerability_index_full = vulnerability_index_full.merge(total_beds_hosp_summary, on=['CVEGEO'], how='left')

# Population at risk (PAR), population access to infrastructure (PAI)
vulnerability_index_full['PAR'] = vulnerability_index_full['V_index'].mul(vulnerability_index_full['POB']).round(0)
vulnerability_index_full['PAI'] = vulnerability_index_full['CAMAS_1000hab'].mul(vulnerability_index_full['PAR']).round(0)

#%%
# Erase total available beds per municipality
vulnerability_index_full = vulnerability_index_full.drop(['TOTAL DE CAMAS'], axis=1)
vulnerability_index_full_graphs = pd.melt(vulnerability_index_full, 
                                  id_vars=['CVEGEO', 'NOM_ENT', 'CLAVE_ENT', 'MUN', 'POB'],
                                   value_vars=['V_index', 'S_health_ac', 'V_A_index', 'CAMAS_1000hab',
                                               'PAR', 'PAI'],
                                   var_name='V_type', value_name='Value')

vulnerability_index_full_summary = vulnerability_index_full.groupby(['CLAVE_ENT'])['POB', 'PAR', 'PAI'].sum().reset_index()

#%%
# Population at risk and with access to health services per State, Municipality and Sex

vulnerability_index_sex = vulnerability_index.merge(
        vulnerability_poverty_index.loc[:,['CLAVE', 'NOM_ENT', 'CLAVE_ENT',
                                           'MUN', 'SEXO', 'RANGO', 'S_health_ac']],
        on=['CLAVE', 'NOM_ENT', 'CLAVE_ENT', 'MUN', 'SEXO', 'RANGO'], how='left')


pop_sex_muni = vulnerability_index_sex.groupby(['CLAVE', 'NOM_ENT', 'CLAVE_ENT',
                                           'MUN', 'SEXO'])['POB'].sum().reset_index()

vulnerability_index_sex = vulnerability_index_sex.groupby(['CLAVE', 'NOM_ENT', 'CLAVE_ENT',
                                           'MUN', 'SEXO'])['C_sex_index', 'S-sex', 'S_health_ac'].mean().reset_index()

vulnerability_index_sex['V_index'] = vulnerability_index_sex['C_sex_index'].div(vulnerability_index_sex['S-sex'])

vulnerability_index_sex['V_A_index'] = (vulnerability_index_sex['V_index'] + vulnerability_index_sex['S_health_ac']) / 2  



vulnerability_index_sex = vulnerability_index_sex.merge(pop_sex_muni, on=['CLAVE', 'NOM_ENT', 'CLAVE_ENT',
                                           'MUN', 'SEXO'], how='left')

sex_dict_english = {'Hombres':'Male', 'Mujeres':'Female'}
vulnerability_index_sex['SEXO'] = vulnerability_index_sex['SEXO'].replace(sex_dict_english)
vulnerability_index_sex = vulnerability_index_sex.rename(columns={'SEXO':'Gender', 'CLAVE':'CVEGEO'})

vulnerability_index_sex = vulnerability_index_sex.merge(total_beds_hosp_summary, on=['CVEGEO'], how='left')

vulnerability_index_sex = vulnerability_index_sex.drop(['POB_y', 'TOTAL DE CAMAS'], axis=1)
vulnerability_index_sex = vulnerability_index_sex.rename(columns={'POB_x':'POP'})

vulnerability_index_sex['PAR'] = vulnerability_index_sex['POP'].mul(vulnerability_index_sex['V_index']).round(0)
vulnerability_index_sex['PAI'] = vulnerability_index_sex['CAMAS_1000hab'].mul(vulnerability_index_sex['PAR']).round(0)
vulnerability_index_sex['PAR-A'] = vulnerability_index_sex['POP'].mul(vulnerability_index_sex['V_A_index']).round(0)
vulnerability_index_sex['PAI-A'] = vulnerability_index_sex['CAMAS_1000hab'].mul(vulnerability_index_sex['PAR-A']).round(0)

vulnerability_index_sex['Legend'] = tuple(zip((vulnerability_index_sex['POP'] / 1000), 
                                              (vulnerability_index_sex['PAR'] / 1000), 
                                              (vulnerability_index_sex['PAI'] / 1000)))
#%%
vulnerability_index_sex_state = vulnerability_index_sex.groupby(['CLAVE_ENT', 'NOM_ENT',
                                                                 'Gender'])['POP','PAR','PAI','PAR-A','PAI-A'].sum().reset_index()

#%%
vulnerability_index_sex_state['Legend'] = tuple(zip((vulnerability_index_sex_state['POP'] / 1000), 
                                              (vulnerability_index_sex_state['PAR'] / 1000), 
                                              (vulnerability_index_sex_state['PAI'] / 1000)))
#%%
vulnerability_index_sex_graph = vulnerability_index_sex_state.pivot(index='NOM_ENT',
                                                              columns ='Gender', 
                                                              values =['POP', 'PAR', 'PAI'])

vulnerability_index_sex_graph.columns = ['{}_{}'.format(x[0], x[1]) for x in vulnerability_index_sex_graph.columns]


vulnerability_index_sex_graph2 = pd.melt(vulnerability_index_sex,
                                         id_vars=['CLAVE_ENT', 'NOM_ENT', 'Gender', 'Legend'],
                                         value_vars=['POP', 'PAR', 'PAI'],
                                         value_name='Value')


#%%


# Count municipalities with given index
vulnerability_index_count = vulnerability_index_full.groupby([pd.cut(vulnerability_index_full['V_index'], 
                                                              np.arange(0, 1.20, .2))])['CVEGEO'].count().reset_index()

vulnerability_index_a_count = vulnerability_index_full.groupby([pd.cut(vulnerability_index_full['V_A_index'], 
                                                              np.arange(0, 1.20, .2))])['CVEGEO'].count().reset_index()

vulnerability_municipality = pd.concat([vulnerability_index_count, vulnerability_index_a_count], axis=1)
vulnerability_municipality = vulnerability_municipality.drop(['V_A_index'], axis=1)
vulnerability_municipality.columns = ['Range', 'Mun_V_Index', 'Mun_V_A_Index']

vulnerability_municipality_graphs = pd.melt(vulnerability_municipality, 
                                            id_vars=['Range'],
                                            value_vars=['Mun_V_Index','Mun_V_A_Index'],
                                            var_name = 'Index', value_name='Count')

#%%
# Save files for graphs in R with ggplot2
vulnerability_index.to_csv(clean_data + 'comorbilityIndex.csv')
vulnerability_index_summary.to_csv(clean_data + 'vulnerabilityIndex.csv')
vulnerability_index_graphs.to_csv(clean_data + 'vulnerability_index_graphs.csv')
vulnerability_pop_graphs.to_csv(clean_data + 'vulnerability_pop_graphs.csv')
vulnerability_poverty_index.to_csv(clean_data + 'vulnerability_poverty_index.csv')
total_beds_hosp_summary.to_csv(clean_data + 'total_beds_hosp_summary.csv')
vulnerability_index_full.to_csv(clean_data + 'vulnerability_index_full.csv')
vulnerability_index_full_graphs.to_csv(clean_data + 'vulnerability_index_full_graphs.csv')
vulnerability_index_sex.to_csv(clean_data + 'vulnerability_index_sex.csv', encoding='latin')
vulnerability_index_sex_state.to_csv(clean_data + 'vulnerability_index_sex_state.csv', encoding='latin')
vulnerability_index_sex_graph.to_csv(clean_data + 'vulnerability_index_sex_graph.csv', encoding='latin')
vulnerability_index_sex_graph2.to_csv(clean_data + 'vulnerability_index_sex_graph2.csv', encoding='latin')
vulnerability_municipality.to_csv(clean_data + 'vulnerability_municipality.csv', encoding='utf-8')
vulnerability_municipality_graphs.to_csv(clean_data + 'vulnerability_municipality_graphs.csv', encoding='utf-8')



