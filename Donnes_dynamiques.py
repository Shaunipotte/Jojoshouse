
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dm4bem import read_epw, sol_rad_tilt_surf
from Rayonnement import donnees


#start_date = '2000-06-29 12:00'
#end_date = '2000-06-30 12:00'


def donnees_dynamique(start_date,end_date) : 
    
    filename = './weather_data/FRA_Lyon.074810_IWEC.epw'
    [data, meta] = read_epw(filename, coerce_year=None)
    data
    # Extract the month and year from the DataFrame index with the format 'MM-YYYY'
    month_year = data.index.strftime('%m-%Y')
    # Create a set of unique month-year combinations
    unique_month_years = sorted(set(month_year))
    # Create a DataFrame from the unique month-year combinations
    pd.DataFrame(unique_month_years, columns=['Month-Year'])
    # select columns of interest
    weather_data = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
    # replace year with 2000 in the index 
    weather_data.index = weather_data.index.map(
        lambda t: t.replace(year=2000))
    #Pour lire les données à une date et heure précise : 
    weather_data.loc[start_date]
    
    
        # Définition de la durée étudiée 
    
    # Filter the data based on the start and end dates
    weather_data = weather_data.loc[start_date:end_date]
    
    # Remove timezone information from the index
    weather_data.index = weather_data.index.tz_localize(None)
    
    del data
    weather_data
        
    rayonnement = {}
    sud = {}
    valeur = weather_data.index

    
    dico_dyn = {}
    Text = {}
    for val in valeur :
        dico,Tpt = donnees(str(val))
        dico_dyn[str(val)] = dico
        Text[str(val)] = Tpt
    return dico_dyn, Text


def moyenne(start_date,end_date) :
    ray_moyen = {}
    dico, Text = donnees_dynamique(start_date,end_date)
    sommeT = 0
    somme_dir_sud = 0
    somme_dif_sud = 0
    somme_ref_sud = 0
    somme_dir_nord = 0
    somme_dif_nord = 0
    somme_ref_nord = 0
    i = 0
    for key, val in dico.items() :  
        i = i+1
        somme_dir_sud =  somme_dir_sud + val['sud']['dir_rad']
        somme_dif_sud = somme_dif_sud + val['sud']['dif_rad']
        somme_ref_sud = somme_ref_sud + val['sud']['ref_rad']
        somme_dir_nord =  somme_dir_nord + val['nord']['dir_rad']
        somme_dif_nord = somme_dif_nord + val['nord']['dif_rad']
        somme_ref_nord = somme_ref_nord + val['nord']['ref_rad']
        sommeT = sommeT + Text[key]
        
    total_sud = somme_dir_sud + somme_dif_sud + somme_ref_sud
    sud = {}
    sud['dir_rad'] = somme_dir_sud / i
    sud['dif_rad'] = somme_dif_sud / i
    sud['ref_rad'] = somme_ref_sud / i
    sud['total'] = total_sud/i
    ray_moyen['sud']=sud
    
    total_nord = somme_dir_nord + somme_dif_nord + somme_ref_nord
    nord = {}
    nord['dir_rad'] = somme_dir_nord / i
    nord['dif_rad'] = somme_dif_nord / i
    nord['ref_rad'] = somme_ref_nord / i
    nord['total'] = total_nord/i
    ray_moyen['nord']=nord
    
    Tpt_ext = sommeT/i
        
    return ray_moyen, Tpt_ext

#dico_moyen, Text = moyenne(start_date,end_date)
