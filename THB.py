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

###############################################################################
############################# Données #########################################
###############################################################################

#moment = '2000-06-29 12:00'
#dico_rayonnement, Text = (donnees(moment)) # récupération des données

#en statique on prend la moyenne sur une journée, il devrait y avoir un moyen de faire ça avec le dico
T_ext = 25

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

#éclairement
alpha_ext=0.5
alpha_in=0.4
tau=0.3
EN = 332.8795 ###éclairement nord à rédéfinir
ES = 406.649  ###éclairement surd à rédéfinir


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
b = np.zeros([nq,1])
b[0,0] = T_ext
b[15,0] = T_ext
b[16,0] = T_ext
b[18,0] = T_ext
b[19,0] = Tc
b[20,0] = Tc

b = pd.DataFrame(b, index=q, columns=[1])

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
G[19,19] = KpN
G[20,20] = KpS

G = pd.DataFrame(G, index=q, columns=q) #### faut comprendre ça fait quoi ?

########################## Matrice f des flux apportés ########################
f = np.zeros((nθ,1))

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

f = pd.DataFrame(f, index=θ, columns=[1])

############# Matrice C des capacités (en statique non utile) #################

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

C = pd.DataFrame(C, index=θ, columns=θ)

# Matrice des températures
y = np.zeros(len(θ))     # nodes and len(θ) = 15
pd.DataFrame(y, index=θ)

#on se retrouve avec ce circuit : thermal circuit
print("A:", A.shape)
print("G:", G.shape)
print("b:", b.shape)
print("f:", f.shape)

###############################################################################
###################### Résolution du circuit statique #########################
###############################################################################
y = inv(A.T @ G @ A) @ (A.T @ G @ b + f)
print(y)

plt.plot(y,'*b')
plt.plot([0,14],[T_ext, T_ext],"*r")
plt.plot([5,9],[Tc, Tc],"*g")
plt.title("Du Nord au Sud ->")  ### title : "Température dans les 14 neouds définis dans le logement étudié"
plt.ylabel("Température en °C")
plt.legend(["Températures en chaque point","Températures extérieures","Températures visées par le controller"])

y = pd.DataFrame(y, index=θ, columns=[1])



#recerche des flux
q = G @ (-(A @ y) + b)
print(q)
