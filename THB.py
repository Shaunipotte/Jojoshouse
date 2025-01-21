# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 12:06:14 2025

@author: chaou
"""
### les import
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt

#################################
########### Données #############
#################################

# moment = '2000-06-29 12:00'
dico_rayonnement, Text = (donnees(moment)) # récupération des données

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

door = {'Conductivity': 1,  
        'Width': 0.04,  
       'Surface' : 2}                     # m²

Surface = {'Nord': longueur*hauteur-door['Surface']-glass['Surface'],
           'Sud': longueur*hauteur-glass['Surface'],
           'Milieu':longueur*hauteur-door['Surface'],
           'Lateral':longueur/2*hauteur}

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
Kp = 0

#éclairement
alpha_ext=0.5
alpha_in=0.4
tau=0.3
EN = 406.649 ###éclairement nord à rédéfinir
ES = 332.8795 ###éclairement surd à rédéfinir


########################################
####### le schéma général ##############
########################################
#les noeuds
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14']
# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18']
# temperature nodes
nθ = len(θ)      # number of temperature nodes
#θ = [f'θ{i}' for i in range(nθ)]
# flow-rate branches
nq = len(q)     # number of flow branches
#q = [f'q{i}' for i in range(nq)]

#matrice A des flux
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

A[18, 5]= 1
A[17, 5]= 1
A[17, 9]= -1
A[16, 9]= 1

pd.DataFrame(A, index=q, columns=θ)

# Matrice B avec T_ext à redéfinir
T_ext = 13
b = np.zeros([nq,1])
b[0,0] = T_ext
b[15,0] = T_ext
b[16,0] = T_ext
b[18,0] = T_ext


# Matrice G #
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

# glass: convection outdoor & conduction
Gglass16 = wall.loc['Glass', 'Surface'] / (1 / h['out'] + 1 / G_cd['Glass'] + 1 / h['in'])
Gporte16 = wall.loc['Door', 'Surface'] / (1 / h['out'] + 1 / G_cd['Door'] + 1 / h['in'])
Gporte17 = wall.loc['Door', 'Surface'] / (1 / h['in'] + 1 / G_cd['Door'] + 1 / h['in'])
G16 = float(Gv['S'] + Gglass16.iloc[0])
G17 = float(Gv['I'] + Gporte17.iloc[0])
G18 = float(Gv['N'] + Gglass16.iloc[0] + Gporte16.iloc[0])

## remplissage de G
G = np.zeros((nq, nq))

G[0, 0] = h['out'].iloc[0] * Surface['Nord']
G[1,1] = G_cd['Layer_out']*Surface['Nord']/2
G[2,2] = G[1,1]
G[3,3] = G_cd['Layer_in']*Surface['Nord']/2
G[4,4] = G[3,3]
G[5, 5] = h['in'].iloc[0] * Surface['Nord']
G[6, 6] = h['in'].iloc[0] * Surface['Milieu']
G[7,7] = G_cd['Layer_in']*Surface['Milieu']/2
G[8,8] = G[7,7]
G[9, 9] = h['in'].iloc[0] * Surface['Milieu']
G[10, 10] = h['in'].iloc[0] * Surface['Sud']
G[11,11] = G_cd['Layer_in']*Surface['Sud']/2
G[12,12] = G[11,11]
G[13,13] = G_cd['Layer_out']*Surface['Sud']/2
G[14,14] = G[13,13]
G[15, 15] = h['out'].iloc[0] * Surface['Sud']
G[16,16] = G16
G[17,17] = G17
G[18,18] = G18

G = pd.DataFrame(G, index=q, columns=q) #### faut comprendre ça fait quoi ?

# Matrice f des flux apportés
phi_n=alpha_ext*EN*Surface["Nord"]
phi_s=alpha_ext*ES*Surface["Sud"]
phi_iN=tau*EN*glass["Surface"]
phi_iN1=alpha_in*phi_iN*(Surface["Nord"]/(Surface["Milieu"]+2*Surface["Lateral"]+Surface["Nord"]))
phi_iN2=alpha_in*phi_iN*(Surface["Milieu"]/(Surface["Milieu"]+2*Surface["Lateral"]+Surface["Nord"]))
phi_iS=tau*ES*glass["Surface"]
phi_iS1=alpha_in*phi_iS*(Surface["Sud"]/(2*Surface["Lateral"]+Surface["Milieu"]+Surface["Sud"]))
phi_iS2 = 1 #### faut définir ça pour la matrice f
f = pd.Series([phi_n, 0, 0, 0, phi_iN1, 0, phi_iN2, 0, phi_iS2, 0, phi_iS1, 0, 0, 0, phi_s], index=θ)

# Matrice C des capacités 'pour l'instant en statique non utile
# Compute capacities for walls
C_walls = wall['Density'] * wall['Specific heat'] * wall['Width']
# Compute capacity for air
C_air = air['Density'] * air['Specific heat'] * air['Volume']

# Initialize the C matrix (2D)
C = np.zeros((nθ, nθ))
# Assign non-zero capacities to specific diagonal elements
C[1, 1] = C_walls.loc['Layer_out']*Surface['Nord'] #isolant Nord
C[3, 3] = C_walls.loc['Layer_in']*Surface['Nord'] #béton nord
C[7, 7] = C_walls.loc['Layer_in']*Surface['Milieu']
C[11, 11] = C_walls.loc['Layer_in']*Surface['Sud']
C[13, 13] = C_walls.loc['Layer_out']*Surface['Sud']
C[5, 5] = C_air #capacité des pièces
C[9, 9] = C_air
#print(C)

# Matrice des températures
y = np.zeros(len(θ))     # nodes and len(θ) = 15
#y[[6]] = 1              # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)

#on se retrouve avec ce circuit
# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.DataFrame(G, index=q, columns=q)

G_np = G.to_numpy()


###########################################################
################ Résolution du cas statique ###############
###########################################################
y = inv(-np.transpose(A) @ G @ A) @ (np.transpose(A) @ G @ b + f)
# ou bien : np.linalg.inv(A.T @ G @ A) @ (A.T @ G @ b + f)

print(y)

###########################################################
################ Résolution du cas Dynamique ############## proposition je pense pas que ça fonctionne vraiment
###########################################################
C_inv = np.linalg.pinv(C) #matrice pseudo_inverse sinon pb de singulat matrix

# State matrix
As = -C_inv @ A.T @ G @ A
# pd.set_option('precision', 1)
pd.DataFrame(As, index=θ, columns=θ)

# Input matrix
Bs = C_inv @ np.block([A.T @ G, np.eye(nθ)])
# pd.set_option('precision', 2)
pd.DataFrame(Bs, index=θ, columns=q + θ)





