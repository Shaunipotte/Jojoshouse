air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)
pd.DataFrame(air, index=['Air'])

# ventilation facade sud
Va = 3*4*8                   # m³, volume d'une pièce
ACH_S = 0.5                     # 1/h, air changes per hour
Va_dotS = ACH / 3600 * Va    # m³/s, infiltration "instantanée"

# ventilation & infiltration
Gvent_S = air['Density'] * air['Specific heat'] * Va_dot

# ventilation facade nord
ACH_N = 0.5                     # 1/h, air changes per hour
Va_dotN = ACH_N / 3600 * Va    # m³/s, infiltration "instantanée"

# ventilation & infiltration
Gvent_N = air['Density'] * air['Specific heat'] * Va_dot
