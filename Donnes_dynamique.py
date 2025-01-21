
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dm4bem import read_epw, sol_rad_tilt_surf
from Rayonnement import donnees


start_date = '2000-06-29 12:00'
end_date = '2000-06-30 12:00'


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
    for val in valeur :
        dico,Text = donnees(str(val))
        dico_dyn[str(val)] = dico

    
    return dico_dyn 
