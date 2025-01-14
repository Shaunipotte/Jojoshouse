air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
air['Volume'] = 3*4*8
air['Capacity'] = air['Density'] * air['Specific heat'] * air['Volume']
pd.DataFrame(air, index=['Air'])

#FACADE SUD
ACH_S = 0.25                     # 1/h, air changes per hour
Va_dotS = ACH_S / 3600 * air['Volume']    # m³/s, infiltration "instantanée"
#ENTRE PIECES
ACH_I = 0.25                    # 1/h, air changes per hour
Va_dotI = ACH_I / 3600 * air['Volume']    # m³/s, infiltration "instantanée"
#FACADE NORD
ACH_N = 0.5                    # 1/h, air changes per hour
Va_dotN = ACH_N / 3600 * air['Volume']    # m³/s, infiltration "instantanée"

# ventilation & infiltration
Gvent_S = air['Density'] * air['Specific heat'] * Va_dotS
Gvent_I = air['Density'] * air['Specific heat'] * Va_dotI
Gvent_N = air['Density'] * air['Specific heat'] * Va_dotN
