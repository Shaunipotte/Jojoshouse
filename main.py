# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import dm4bem


Longueur = 8 # longeur de l'appartement
hauteur = 3 # hateur des murs 

air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
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
         'Surface': 2}                     # m²

door = {'Surface' : 2}                     # m²
wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass,
                                'Door': door},
                              orient='index')

Surface = {'Nord': Longueur*hauteur-door['Surface']-glass['Surface'],
           'Sud': Longueur*hauteur-glass['Surface'],
           'Milieu':Longueur*hauteur-door['Surface'],
           'Lateral':Longueur/2*hauteur}


# radiative properties (for the sun radiations)
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)

# conduction
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']
pd.DataFrame(G_cd, columns=['Conductance'])

# convection
Gw = h * wall['Surface'].iloc[0]     # wall
Gg = h * wall['Surface'].iloc[2]     # glass


#### Radiation long ##### Négligé

# T_int = 273.15 + np.array([0, 40])
# coeff = np.round((4 * σ * T_int**3), 1)
# print(f'For 0°C < (T/K - 273.15)°C < 40°C, 4σT³/[W/(m²·K)] ∈ {coeff}')


# T_int = 273.15 + np.array([10, 30])
# coeff = np.round((4 * σ * T_int**3), 1)
# print(f'For 10°C < (T/K - 273.15)°C < 30°C, 4σT³/[W/(m²·K)] ∈ {coeff}')


# T_int = 273.15 + 20
# coeff = np.round((4 * σ * T_int**3), 1)
# print(f'For (T/K - 273.15)°C = 20°C, 4σT³ = {4 * σ * T_int**3:.1f} W/(m²·K)')


# # long wave radiation
# Tm = 20 + 273   # K, mean temp for radiative exchange

# GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['Layer_in']
# GLW12 = 4 * σ * Tm**3 * Fwg * wall['Surface']['Layer_in']
# GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['Glass']

# GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2)

##G de des infiltrations d'air pour les différentes parois
ACH = {'S': 2, 
       'I': 2,
      'N':4}
Va_dot = {'S' = ACH['S'] / 3600 * air['Volume'],
              'I' = ACH['I'] / 3600 * air['Volume'],
              'N' = ACH['N'] / 3600 * air['Volume']}
Gv = {'S' =  air['Density'] * air['Specific heat'] * Va_dot['S'],
       'I' = air['Density'] * air['Specific heat'] * Va_dot['I'],
       'N' = air['Density'] * air['Specific heat'] * Va_dot['N']}

##### P-controler gain ######
# Kp = 1e4            # almost perfect controller Kp -> ∞
# Kp = 1e-3           # no controller Kp -> 0
Kp = 0



# glass: convection outdoor & conduction
G16 = float(  1 / (1 / Gg.loc['h', 'out'] + 1 / G_cd['Glass'] + 1 / Gg.loc['h', 'in'] )  )    #### Why x2 ? ####  
G17 = float(  1 / (1 / Gg.loc['h', 'in'] + 1 / G_cd['Glass'] + 1 / Gg.loc['h', 'in'] )  )
G18 = G16


# Capacity
C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
pd.DataFrame(C, columns=['Capacity'])

C['Air'] = air['Density'] * air['Specific heat'] * Va
pd.DataFrame(C, columns=['Capacity'])







###### Mettre notre cas ######

θ = ['θ0', 'θ1', 'θ2', 'θ3', 'θ4', 'θ5', 'θ6', 'θ7','θ8', 'θ9', 'θ10', 'θ11', 'θ12', 'θ13', 'θ14']

# flow-rate branches
q = ['q0', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11','q12', 'q13', 'q14', 'q15', 'q16', 'q17', 'q18']

# temperature nodes
nθ = len(θ)      # number of temperature nodes
#θ = [f'θ{i}' for i in range(nθ)]

# flow-rate branches
nq = len(q)     # number of flow branches
#q = [f'q{i}' for i in range(nq)]


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

#Matrice B avec T_ext à redéfinir
T_ext = 13
b = np.zeros([nq,1])
b[0,0] = T_ext
b[15,0] = T_ext
b[16,0] = T_ext
b[18,0] = T_ext

pd.DataFrame(A, index=q, columns=θ)

G = np.array(np.hstack(
    [Gw['out'],
     G_cd['Layer_out']/2, G_cd['Layer_out']/2,
     G_cd['Layer_in']/2, G_cd['Layer_in']/2,
     Gw['in'], Gw['in'],
     G_cd['Layer_in']/2, G_cd['Layer_in']/2,
     Gw['in'], Gw['in'],
     G_cd['Layer_in']/2, G_cd['Layer_in']/2,
     G_cd['Layer_out']/2, G_cd['Layer_out']/2,
     Gw['out'],
     G16, G17, G18]
     ))


# np.set_printoptions(precision=3, threshold=16, suppress=True)
# pd.set_option("display.precision", 1)
pd.DataFrame(G, index=q)

neglect_air_glass = False

if neglect_air_glass:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                  0, 0])
else:
    C = np.array([0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                  C['Air'], C['Glass']])

# pd.set_option("display.precision", 3)
pd.DataFrame(C, index=θ)

b = pd.Series(['To', 0, 0, 0, 0, 0, 0, 0, 'To', 0, 'To', 'Ti_sp'],
              index=q)


f = pd.Series(['Φo', 0, 0, 0, 'Φi', 0, 'Qa', 'Φa'],
              index=θ)

y = np.zeros(8)         # nodes
y[[6]] = 1              # nodes (temperatures) of interest
pd.DataFrame(y, index=θ)

# thermal circuit
A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

# TC = dm4bem.file2TC('./toy_model/TC.csv', name='', auto_number=False)

# TC['G']['q11'] = 1e3  # Kp -> ∞, almost perfect controller
TC['G']['q11'] = 0      # Kp -> 0, no controller (free-floating)

#[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
#us

# q0        To
# q8        To
# q10       To
# q11    Ti_sp
# θ0        Φo
# θ4        Φi
# θ6        Qa
# θ7        Φa
# dtype: object
