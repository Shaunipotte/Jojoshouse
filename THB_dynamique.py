# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:09:20 2025

@author: chaou
"""
################ Les imports ###################
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt

from dm4bem import read_epw, sol_rad_tilt_surf, tc2ss, inputs_in_time
from dm4bem import *
from Donnes_dynamiques import donnees_dynamique

###############################################################################
############################# Données #########################################
###############################################################################

start_date = '2000-06-29 12:00' # à changer seon la journée que l'on veut
end_date = '2000-06-30 12:00'

dico_dyn, Text_dyn = donnees_dynamique(start_date, end_date)

#en dynamique la température change au fur et à mesure
T_ext = 25
#éclairement
alpha_ext=0.5
alpha_in=0.4
tau=0.3
EN = 332.8795 ###éclairement nord à rédéfinir
ES = 406.649  ###éclairement surd à rédéfinir

largeur = 4     # largeur des pièces
longueur = 8    # longueur de l'appartement
hauteur = 3     # hauteur des murs 

## définitions de dictionnaires des différents composants
air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000,
       'Volume': longueur*largeur*hauteur}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])


concrete = {'Conductivity': 1.400,          # W/(m·K)
            'Density': 2300.0,              # kg/m³
            'Specific heat': 880,           # J/(kg⋅K)
            'Width': 0.2}                   # m
     
insulation = {'Conductivity': 0.027,        # W/(m·K)
              'Density': 55.0,              # kg/m³
              'Specific heat': 1210,        # J/(kg⋅K)
              'Width': 0.08}                # m

glass = {'Conductivity': 1.4,               # W/(m·K)
         'Density': 2500,                   # kg/m³
         'Specific heat': 1210,             # J/(kg⋅K)
         'Width': 0.04,                     # m
         'Surface': 2,
         'Transmission': 0.8}                     # m²

door = {'Conductivity': 0.1,  
        'Width': 0.04,  
       'Surface' : 2}                     # m²

Surface = {'Nord': longueur*hauteur-door['Surface']-glass['Surface'],
           'Sud': longueur*hauteur-glass['Surface'],
           'Milieu':longueur*hauteur-door['Surface'],
           'Lateral':longueur/2*hauteur,
          'Plafond' : longueur*largeur}

### création du panda mur
wall = pd.DataFrame.from_dict({'Layer_in': concrete,
                               'Layer_out': insulation,
                               'Glass': glass,
                                'Door': door},
                              orient='index')

# définition coeff convection 
h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])

#thermostat ######
# Kp = 1e4            # almost perfect controller Kp -> ∞
# Kp = 1e-3           # no controller Kp -> 0
KpN = 1e-4 #pièce nord
KpS = 1e3 #pièce Sud
Tc = 18 #été
#Tc = 21 #hiver

## flux utilisateur
Qa = 80 #~80 par personne, ici c'est celui de la pièce Nord (four, télé, personnes)

###############################################################################
############################# Le schéma général ###############################
###############################################################################
#les noeuds
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14']
# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20']
# temperature nodes
nθ = len(θ)      # number of temperature nodes
#θ = [f'θ{i}' for i in range(nθ)] #autre méthode
# flow-rate branches
nq = len(q)     # number of flow branches
#q = [f'q{i}' for i in range(nq)] #autre méthode


########################### matrice A des flux #############################

A = np.zeros([nq, nθ])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 5] = -1, 1    # branch 5: node 4 -> node 5

A[6, 5], A[6, 6] = 1, -1    # branch 6: node 5 -> node 6
A[7, 6], A[7, 7] = 1, -1    # branch 7: node 6 -> node 7
A[8, 7], A[8, 8] = 1, -1    # branch 8: node 7 -> node 8
A[9, 8], A[9, 9] = 1, -1    # branch 9: node 8 -> node 9
A[10, 9], A[10, 10] = 1, -1    # branch 10: node 9 -> node 10
A[11, 10], A[11, 11] = 1, -1    # branch 11: node 10 -> node 11
A[12, 11], A[12, 12] = 1, -1    # branch 12: node 11 -> node 12
A[13, 12], A[13, 13] = 1, -1    # branch 13: node 12 -> node 13
A[14, 13], A[14, 14] = 1, -1    # branch 14: node 13 -> node 14
A[15, 14]= 1   # branch 15: node 14 -> node 15

# porte, fenetre, ventilation
A[18, 5]= 1
A[17, 5]= 1
A[17, 9]= -1
A[16, 9]= 1

#controler
A[19,5] = 1
A[20,9] = 1

A = pd.DataFrame(A, index=q, columns=θ)

############ Matrice B avec T_ext définit à l'aide du code rayonnement ########
b = np.zeros([nq])
b[0] = T_ext
b[15] = T_ext
b[16] = T_ext
b[18] = T_ext
b[19] = Tc
b[20] = Tc

b = pd.Series(b, index=q)

#################################### Matrice G ################################

# définiton conductance de conduction
G_cd = wall['Conductivity'] / wall['Width']
pd.DataFrame(G_cd, columns=['Conductance'])
##G de des infiltrations d'air pour les différentes parois
ACH = {'S': 2, 
       'I': 2,
      'N':4}
Va_dot = {'S' : ACH['S'] / 3600 * air['Volume'],
              'I' : ACH['I'] / 3600 * air['Volume'],
              'N' : ACH['N'] / 3600 * air['Volume']}
Gv = {'S' :  air['Density'] * air['Specific heat'] * Va_dot['S'],
       'I' : air['Density'] * air['Specific heat'] * Va_dot['I'],
       'N' : air['Density'] * air['Specific heat'] * Va_dot['N']} 

#Gv['S'] = 0 ##ventilation nord vers Sud
Gv['N'] = 0  ##ventilation Sud vers Nord

# glass: convection outdoor & conduction
Gglass16 = wall.loc['Glass', 'Surface'] / (1 / h['out'] + 1 / G_cd['Glass'] + 1 / h['in'])
Gporte16 = wall.loc['Door', 'Surface'] / (1 / h['out'] + 1 / G_cd['Door'] + 1 / h['in'])
Gporte17 = wall.loc['Door', 'Surface'] / (1 / h['in'] + 1 / G_cd['Door'] + 1 / h['in'])
G16 = float(Gv['S'] + Gglass16.iloc[0])
G17 = float(Gv['I'] + Gporte17.iloc[0])
G18 = float(Gv['N'] + Gglass16.iloc[0] + Gporte16.iloc[0])

## remplissage de G
GN = np.array(np.hstack([h['out'].iloc[0] * Surface['Nord'], 
      G_cd['Layer_out']*Surface['Nord']/2,
      G_cd['Layer_out']*Surface['Nord']/2,
      G_cd['Layer_in']*Surface['Nord']/2,
      G_cd['Layer_in']*Surface['Nord']/2,
      h['in'].iloc[0] * Surface['Nord']]))

GM = np.array((h['in'].iloc[0] * Surface['Milieu'],
      G_cd['Layer_in']*Surface['Milieu']/2,
      G_cd['Layer_in']*Surface['Milieu']/2,
      h['in'].iloc[0] * Surface['Milieu']))

GS = np.array((h['in'].iloc[0] * Surface['Sud'],
      G_cd['Layer_in']*Surface['Sud']/2,
      G_cd['Layer_in']*Surface['Sud']/2,
      G_cd['Layer_out']*Surface['Sud']/2,
      G_cd['Layer_out']*Surface['Sud']/2,
      h['out'].iloc[0] * Surface['Sud']))

GP = np.array((G16, G17, G18))
GC = np.array((KpN, KpS))

G = np.array(np.hstack((GN, GM, GS, GP, GC)))
G = pd.DataFrame(G, index=q)

########################## Matrice f des flux apportés ########################
f = np.zeros([nθ])

phi_n=alpha_ext*EN*Surface["Nord"]
phi_s=alpha_ext*ES*Surface["Sud"]
phi_iN=tau*EN*glass["Surface"]
phi_iN1=alpha_in*phi_iN*(Surface["Nord"]/(Surface["Milieu"]+2*Surface["Lateral"]+Surface["Nord"]+2*Surface["Plafond"]))
phi_iN2=alpha_in*phi_iN*(Surface["Milieu"]/(Surface["Milieu"]+2*Surface["Lateral"]+Surface["Nord"]+2*Surface["Plafond"]))
phi_iS=tau*ES*glass["Surface"]
phi_iS1=alpha_in*phi_iS*(Surface["Sud"]/(2*Surface["Lateral"]+Surface["Milieu"]+Surface["Sud"]+2*Surface["Plafond"]))
phi_iS2 = alpha_in*phi_iS*(Surface["Milieu"]/(2*Surface["Lateral"]+Surface["Milieu"]+Surface["Sud"]+2*Surface["Plafond"]))

f[0] = phi_n
f[4] = phi_iN1
f[5] = Qa
f[6] = phi_iN2
f[8] = phi_iS2
f[10] = phi_iS1
f[14] = phi_s

f = pd.Series(f, index=θ)

############# Matrice C des capacités (en statique non utile) #################

# Compute capacities for walls
C_walls = wall['Density'] * wall['Specific heat'] * wall['Width']
# Compute capacity for air
C_air = air['Density'] * air['Specific heat'] * air['Volume']

# Assign non-zero capacities to specific diagonal elements
CN = np.array(np.hstack([0,
                         C_walls.loc['Layer_out']*Surface['Nord'],
                         0,
                         C_walls.loc['Layer_in']*Surface['Nord'],
                         0,
                         C_air]))
CM = np.array(np.hstack([0,
                         C_walls.loc['Layer_in']*Surface['Milieu'], 
                         0,
                         C_air]))
CS = np.array(np.hstack([0,
                         C_walls.loc['Layer_in']*Surface['Sud'],
                         0,
                         C_walls.loc['Layer_out']*Surface['Sud'],
                         0]))


C = np.array(np.hstack((CN, CM, CS)))
C = pd.DataFrame(C, index=θ)

# Matrice des températures
y = np.zeros(len(θ))     # nodes and len(θ) = 15
y = pd.DataFrame(y, index=θ, columns=[1])

###############################################################################
###################### Résolution dynamique ###################################
###############################################################################

# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.DataFrame(G, index=q)
C = pd.DataFrame(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)
TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}
#on se retrouve avec ce circuit : thermal circuit
print("A:", A.shape)
print("G:", G.shape)
print("C:", C.shape)
print("b:", b.shape)
print("f:", f.shape)
print("y:", y.shape)

## système  DAE : utliser dm4bem
[As, Bs, Cs, Ds, us] = tc2ss(TC)

########################## discretisation #######################################

#définition du pas de temps
λ = np.linalg.eig(As)[0]        # eigenvalues of matrix As
dtmax = 2 * min(-1. / λ)        #pas de temps max
print(f"Pas de temps maximal pour stabilité : {dt_max:.2f} s") 

dt = 180 # Choisir un pas de temps inférieur pour garantir la stabilité

# settling time, temps de fin de simulation max
t_f = 4 * max(-1 / λ)
print_rounded_time('t_settle', t_f) 

# duration: next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_f / 3600) * 3600
print_rounded_time('duration', duration)


#maintenant on passe à la définition des points et du dico de u (f et T)
n_points = int((pd.Timestamp(end_date) - pd.Timestamp(start_date)).total_seconds() / pas_temps)

# DateTimeIndex starting at "00:00:00" with a time step of dt
time = pd.date_range(start = start_date,
                           periods = n, freq=f"{int(dt)}S")

Text = np.ones(n)
t = 0
i = 0
k = 3600/dt
j=0
v=0
while i < n : 
    t += dt
    if j!=20 : 
        Text[i] = dico_dyn[1][v]
        i+=1
    else : 
        v+=1
        j = 1
        Text[i] = dico_dyn[1][v]
        i+=1
        
Tc = Tc*np.ones(n)

Φin = alpha_ext*EN*Surface["Nord"]


Qa = 80*np.ones(n) # on peut tenter ensuite de simuler une évolution des consommations selon la nuit ou le jour en remplissant avec une boucle

data = {'Text': Text, 'Tc': Tc, 'Φin': Φin, 'ΦiN1': ΦiN1, 'Qa': Qa, 'ΦiN2': ΦiN2, 'Φis2': ΦiS2, 'ΦiS1': ΦiS1, 'Φis': Φis}
input_data_set = pd.DataFrame(data, index=time)

u = inputs_in_time(us, input_data_set)

# Initialisation des résultats
theta = np.zeros((n_points, n_theta))  # Températures aux noeuds

# Conditions initiales
theta[0, :] = np.zeros(n_theta)  # Par exemple, toutes les températures initiales à 0

# Simulation par Euler explicite
for i in range(1, n_points):
    # Valeurs actuelles
    current_time = time[i]
    u = np.zeros(Bs.shape[1])  # Initialisation des entrées du système

    # Mise à jour des conditions de bord selon le dictionnaire dico_dyn
    time_str = str(current_time)
    if time_str in dico_dyn:
        u[:len(f)] = [dico_dyn[time_str]['nord']['total'],  # Exemple pour l'éclairement nord
                      dico_dyn[time_str]['sud']['total']]  # Exemple pour l'éclairement sud
    else:
        u[:len(f)] = 0  # Valeurs par défaut si les données sont absentes

    # Intégration explicite : theta(t+1) = theta(t) + dt * (As * theta(t) + Bs * u)
    theta[i, :] = theta[i - 1, :] + pas_temps * (As @ theta[i - 1, :] + Bs @ u)

# Résultats : conversion en DataFrame pour faciliter l'analyse
results = pd.DataFrame(theta, index=time, columns=theta)

# Visualisation des résultats
plt.figure(figsize=(10, 6))
for i, var in enumerate(theta):
    plt.plot(results.index, results[var], label=f"{var}")
plt.xlabel("Temps")
plt.ylabel("Température (°C)")
plt.legend()
plt.title("Évolution des températures par Euler explicite")
plt.grid()
plt.show()


