            #Obtenir des Data
    # Télécharger et lire les data 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dm4bem import read_epw, sol_rad_tilt_surf

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
weather_data.loc['2000-06-29 12:00']

    
    # Définition de la durée étudiée 

# Define start and end dates
start_date = '2000-06-29 12:00'
end_date = '2000-07-02'         # time is 00:00 if not indicated

# Filter the data based on the start and end dates
weather_data = weather_data.loc[start_date:end_date]
del data
weather_data


    # Tracer de la température de l'air extérieure 
    
#Plot outdoor air temperature
weather_data['temp_air'].plot()
plt.xlabel("Time")
plt.ylabel("Dry-bulb air temperature, θ / °C")
plt.legend([])
plt.show()

#Plot solar radiation: normal direct and horizontal diffuse
weather_data[['dir_n_rad', 'dif_h_rad']].plot()
plt.xlabel("Time")
plt.ylabel("Solar radiation, Φ / (W·m⁻²)")
plt.legend(['$Φ_{direct}$', '$Φ_{diffuse}$'])
plt.show()





            #Solar radiation on a tilted surface 
            #Façade SUD
    # Orentiation de la surface 
    
surface_orientation = {'slope': 90,     # 90° car la paroie est verticale 
                       'azimuth': 0,    # 0° car on est orienté plein Sud
                       'latitude': 45.77}  # °
albedo = 0.2


    # Tracer du rayonnement solaire 
    
#Plot solar radiation
rad_surf = sol_rad_tilt_surf(
    weather_data, surface_orientation, albedo)

rad_surf.plot()
plt.xlabel("Time")
plt.ylabel("Solar irradiance,  Φ / (W·m⁻²)")
plt.show()

#Pour ibtenir la valeur à un temps spécifique : 
print(f"{rad_surf.loc['2000-06-29 12:00']['direct']:.0f} W/m²")

#Pour avoir le max et le min, cad une valeur spécifique
print(f"Mean. direct irradiation: {rad_surf['direct'].mean():.0f} W/m²")
print(f"Max. direct irradiation:  {rad_surf['direct'].max():.0f} W/m²")
print(f"Direct solar irradiance is maximum on {rad_surf['direct'].idxmax()}")



                # Calculation of solar radiation on a tilted surface from weather data

β = surface_orientation['slope']
γ = surface_orientation['azimuth']
ϕ = surface_orientation['latitude']

# Transform degrees in radians
β = β * np.pi / 180
γ = γ * np.pi / 180
ϕ = ϕ * np.pi / 180

n = weather_data.index.dayofyear


        # Rayonnement directe 
declination_angle = 23.45 * np.sin(360 * (284 + n) / 365 * np.pi / 180)
δ = declination_angle * np.pi / 180 # C'est l'angele entre l'équateur et la position du soleil lorsqu'il est à son max

hour = weather_data.index.hour
minute = weather_data.index.minute + 60
hour_angle = 15 * ((hour + minute / 60) - 12)   # deg
ω = hour_angle * np.pi / 180                    # rad   # Solar hour angle. négatif le matin, nul à midi, positif l'aprem 


theta = np.sin(δ) * np.sin(ϕ) * np.cos(β) \
    - np.sin(δ) * np.cos(ϕ) * np.sin(β) * np.cos(γ) \
    + np.cos(δ) * np.cos(ϕ) * np.cos(β) * np.cos(ω) \
    + np.cos(δ) * np.sin(ϕ) * np.sin(β) * np.cos(γ) * np.cos(ω) \
    + np.cos(δ) * np.sin(β) * np.sin(γ) * np.sin(ω) # C'est l'angle d'incidence, l'angle entre la normal à la surface et le soleil 

theta = np.array(np.arccos(theta))
theta = np.minimum(theta, np.pi / 2)

dir_rad = weather_data["dir_n_rad"] * np.cos(theta)
dir_rad[dir_rad < 0] = 0



        # Rayonnement diffus 
dif_rad = weather_data["dif_h_rad"] * (1 + np.cos(β)) / 2


        # Rayonnement solaire réfélchis par le sol 
gamma = np.cos(δ) * np.cos(ϕ) * np.cos(ω) \
    + np.sin(δ) * np.sin(ϕ)

gamma = np.array(np.arcsin(gamma))
gamma[gamma < 1e-5] = 1e-5

dir_h_rad = weather_data["dir_n_rad"] * np.sin(gamma)

ref_rad = (dir_h_rad + weather_data["dif_h_rad"]) * albedo \
        * (1 - np.cos(β) / 2)
        
     # Rayonnement total 

total = dir_rad + dif_rad + ref_rad 
total.plot()


