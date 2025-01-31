# # -*- coding: utf-8 -*-


################## Ici Cas 0 par défaut  #################


# ### les import
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
from Donnes_dynamiques import moyenne

###############################################################################
############################# Données #########################################
###############################################################################

start_date = '2000-12-21 00:00' # à changer selon la journée que l'on veut
end_date = '2000-12-22 00:00'

dico_moyen_hiver, T_ext_hiver = moyenne(start_date,end_date)  #en statique on prend la moyenne sur une journée


start_date = '2000-06-21 00:00' 
end_date = '2000-06-22 00:00'

dico_moyen_ete, T_ext_ete = moyenne(start_date,end_date)



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
KpN = 1e-4 #pièce nord, no controller Kp -> 0
KpS = 1e3 #pièce Sud, almost perfect controller Kp -> ∞
Tc_hiver = 22
Tc_ete = 18

## flux utilisateur
Qa = 80 #~80 par personne, ici c'est celui de la pièce Nord (four, télé, personnes)

#éclairement
alpha_ext=0.5
alpha_in=0.4
tau=0.3

EN_hiver = dico_moyen_hiver['nord']['total']
ES_hiver = dico_moyen_hiver['sud']['total']  

EN_ete = dico_moyen_ete['nord']['total'] 
ES_ete = dico_moyen_ete['sud']['total']  


###############################################################################
############################# Le schéma général ###############################
###############################################################################
#les noeuds
θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14']
# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18', 'q19', 'q20']
# temperature nodes
nθ = len(θ)      # number of temperature nodes
# flow-rate branches
nq = len(q)     # number of flow branches


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

############ Matrice B  #############
b_hiver = np.zeros([nq,1])
b_hiver[0,0] = T_ext_hiver
b_hiver[15,0] = T_ext_hiver
b_hiver[16,0] = T_ext_hiver
b_hiver[18,0] = T_ext_hiver
b_hiver[19,0] = Tc_hiver
b_hiver[20,0] = Tc_hiver

b_hiver = pd.DataFrame(b_hiver, index=q, columns=[1])



b_ete = np.zeros([nq,1])
b_ete[0,0] = T_ext_ete
b_ete[15,0] = T_ext_ete
b_ete[16,0] = T_ext_ete
b_ete[18,0] = T_ext_ete
b_ete[19,0] = Tc_ete
b_ete[20,0] = Tc_ete

b_ete = pd.DataFrame(b_ete, index=q, columns=[1])

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
Gglass16 = wall.loc['Glass', 'Surface'] / (1 / h['out'] + 1 / h['in'])
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

G = pd.DataFrame(G, index=q, columns=q) 

########################## Matrice f des flux apportés ########################
f_hiver = np.zeros((nθ,1))

phi_n=alpha_ext*EN_hiver*Surface["Nord"]
phi_s=alpha_ext*ES_hiver*Surface["Sud"]
phi_iN=tau*EN_hiver*glass["Surface"]
phi_iN1=alpha_in*phi_iN*(Surface["Nord"]/(Surface["Milieu"]+2*Surface["Lateral"]+Surface["Nord"]+2*Surface["Plafond"]))
phi_iN2=alpha_in*phi_iN*(Surface["Milieu"]/(Surface["Milieu"]+2*Surface["Lateral"]+Surface["Nord"]+2*Surface["Plafond"]))
phi_iS=tau*ES_hiver*glass["Surface"]
phi_iS1=alpha_in*phi_iS*(Surface["Sud"]/(2*Surface["Lateral"]+Surface["Milieu"]+Surface["Sud"]+2*Surface["Plafond"]))
phi_iS2 = alpha_in*phi_iS*(Surface["Milieu"]/(2*Surface["Lateral"]+Surface["Milieu"]+Surface["Sud"]+2*Surface["Plafond"]))

f_hiver[0] = phi_n
f_hiver[4] = phi_iN1
f_hiver[5] = Qa
f_hiver[6] = phi_iN2
f_hiver[8] = phi_iS2
f_hiver[10] = phi_iS1
f_hiver[14] = phi_s

f_hiver = pd.DataFrame(f_hiver, index=θ, columns=[1])


f_ete = np.zeros((nθ,1))

phi_n=alpha_ext*EN_ete*Surface["Nord"]
phi_s=alpha_ext*ES_ete*Surface["Sud"]
phi_iN=tau*EN_ete*glass["Surface"]
phi_iN1=alpha_in*phi_iN*(Surface["Nord"]/(Surface["Milieu"]+2*Surface["Lateral"]+Surface["Nord"]+2*Surface["Plafond"]))
phi_iN2=alpha_in*phi_iN*(Surface["Milieu"]/(Surface["Milieu"]+2*Surface["Lateral"]+Surface["Nord"]+2*Surface["Plafond"]))
phi_iS=tau*ES_ete*glass["Surface"]
phi_iS1=alpha_in*phi_iS*(Surface["Sud"]/(2*Surface["Lateral"]+Surface["Milieu"]+Surface["Sud"]+2*Surface["Plafond"]))
phi_iS2 = alpha_in*phi_iS*(Surface["Milieu"]/(2*Surface["Lateral"]+Surface["Milieu"]+Surface["Sud"]+2*Surface["Plafond"]))

f_ete[0] = phi_n
f_ete[4] = phi_iN1
f_ete[5] = Qa
f_ete[6] = phi_iN2
f_ete[8] = phi_iS2
f_ete[10] = phi_iS1
f_ete[14] = phi_s

f_ete = pd.DataFrame(f_ete, index=θ, columns=[1])


# Matrice des températures
y = np.zeros(len(θ))     # nodes and len(θ) = 15
pd.DataFrame(y, index=θ)

# thermal circuit
print("A:", A.shape)
print("G:", G.shape)
print("b:", b_ete.shape)
print("f:", f_ete.shape)

###############################################################################
###################### Résolution du circuit statique #########################
###############################################################################
y_hiver = inv(A.T @ G @ A) @ (A.T @ G @ b_hiver + f_hiver)
y_ete = inv(A.T @ G @ A) @ (A.T @ G @ b_ete + f_ete)
print(y_ete)

plt.plot(y_hiver,'-b')
plt.plot([0,14],[T_ext_hiver, T_ext_hiver],"*c")
plt.plot([5,9],[Tc_hiver, Tc_hiver],"*g")

plt.plot(y_ete, '-r')
plt.plot([0,14],[T_ext_ete, T_ext_ete], color = 'pink', marker = '*', linestyle = 'None')
plt.plot([5,9],[Tc_ete, Tc_ete], color = 'orange', marker = '*', linestyle = 'None')

plt.title("Température dans les 15 noeuds définis dans le logement étudié")
plt.xlabel("Du Nord au Sud ->") 
plt.ylabel("Température en °C")
plt.legend(["Températures en chaque point en hiver","Températures extérieures en hiver","Températures visées par le controller en hiver","Températures en chaque point en été","Températures extérieures en été","Températures visées par le controller en été"])


#recerche des flux
θ = range(15)
A = np.array(A)
A = pd.DataFrame(A, index=q, columns=θ)

y_hiver = pd.DataFrame(y_hiver, index=θ, columns=[1])
y_ete = pd.DataFrame(y_ete, index=θ, columns=[1])

q_hiver = G @ (b_hiver - (A @ y_hiver))
print(q_hiver)

q_ete = G @ (b_ete - (A @ y_ete))
print(q_ete)
