#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import CoolProp as CP
import lmfit
import scipy as sp
from psychrochart import PsychroChart #, load_config
import matplotlib.pyplot as plt
import pytemperature as temp
import psychrolib
from CoolProp.CoolProp import PropsSI
import math
psychrolib.SetUnitSystem(psychrolib.SI)


# In[29]:


def zerofn(Tqj,params,T_d,T_w,R_H):      # Evaluates the heat transfer, evap rate locaT_error_at_lower_limity
    global h_Primary
    T_s = Tqj['T_s']
    T_i = Tqj['T_i']
    q_tot = Tqj['q_tot']
    q_conv = Tqj['q_conv']
    j = Tqj['j']

    h_fg = CP.CoolProp.PropsSI('H','T',temp.c2k(T_i),'Q',1,'Water') - CP.CoolProp.PropsSI('H','T',temp.c2k(T_i),'Q',0,'Water')
    p_sat_i = CP.CoolProp.PropsSI('P','T',temp.c2k(T_i),'Q',0,'Water')
    p_sat_w = CP.CoolProp.PropsSI('P','T',temp.c2k(T_i),'Q',0,'Water')
    p_w = R_H * p_sat_w

    "Primary channel heat transfer coeff calclulation"
    h_Primary=htcoeff(T_d, up)
    "Secondary channel heat tansfer coeff calculation"
    h_Secondary=htcoeff(T_w, us)
    
    eq1 = q_tot - h_Primary[0]*(T_d - T_s)
    eq2 = q_tot - params['h_f']*(T_s - T_i)
    eq3 = q_tot - (q_conv + j*h_fg)
    eq4 = q_conv - h_Secondary[0]*(T_i - T_w)
    eq5 = j - h_Secondary[0]* (p_sat_i - p_w)/(h_Secondary[2]*287*temp.c2k(T_w)) * 0.91
    eq_array = np.array([eq1,eq2,eq3,eq4,eq5])
    return eq_array


# In[30]:


def htcoeff(T_w,u): #heat transfer coefficient calculation and corresponding air properties
    rho_air=PropsSI('D','T',temp.c2k(T_w),'P',params['Patm'],'Air')
    mu=PropsSI('V','T',temp.c2k(T_w),'P',params['Patm'],'Air')
    cp=PropsSI('C','T',temp.c2k(T_w),'P',params['Patm'],'Air')
    k=PropsSI('L','T',temp.c2k(T_w),'P',params['Patm'],'Air')
    Pr=PropsSI('PRANDTL','T',temp.c2k(T_w),'P',params['Patm'],'Air')
    Re=rho_air*u*params['hydraulic diameter']/mu
    Nu=2+0.6*(Re**0.5)*Pr**0.33
    h=Nu*k/params['hydraulic diameter']
    return[h,rho_air,cp]


# In[31]:


def dhwdx(A,hw,m_d,m_w,w_in,params):    # defines the rate of change of enthalpy, hum rat
    h_d = hw[0]
    h_w = hw[1]
    w_w = hw[2]
    
    
    T_d = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(h_d,w_in) # CP.HumidAirProp.HAPropsSI('T','H',h_d,'P',101325,'W',w_in)
    T_w = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(h_w,abs(w_w)) # CP.HumidAirProp.HAPropsSI('T','H',h_w,'P',101325,'W',w_w)
    RH_w = psychrolib.GetRelHumFromHumRatio(T_w,abs(w_w),params['Patm']) # CP.HumidAirProp.HAPropsSI('R','H',h_w,'P',101325,'W',w_w)

    
    Tqj = lmfit.Parameters()
    Tqj.add('T_s',value=T_d, min=1, max=100)
    Tqj.add('T_i',value=T_w, min=1, max=100)
    Tqj.add('q_tot',value=500)
    Tqj.add('q_conv',value=100)
    Tqj.add('j',value=1E-5)

    out = lmfit.minimize(zerofn, Tqj, args=(params,T_d,T_w,RH_w))
    
    T_s = out.params['T_s'].value
    T_i = out.params['T_i'].value
    q_tot = out.params['q_tot'].value
    q_conv = out.params['q_conv'].value
    j = out.params['j'].value
    
    w_i = psychrolib.GetHumRatioFromRelHum(T_i, 1, params['Patm'])
    h_vap_i = CP.CoolProp.PropsSI('H','T',temp.c2k(T_i),'Q',1,'Water')
    
    p_vap_w = psychrolib.GetVapPresFromHumRatio(abs(w_w), params['Patm'])
    p_vap_0 = psychrolib.GetVapPresFromHumRatio(w_in, params['Patm'])
    h_s=htcoeff(T_w,us)

    h_vap_w = CP.CoolProp.PropsSI('H','T',temp.c2k(abs(T_w)),'Q',1,'Water')
    
    dh_d = 2*q_tot/m_d
    dh_w = 2*(q_conv + j*h_vap_i)/m_w
    dw_w = 2*(j)/m_w
    dhwdA = np.array([dh_d, dh_w, dw_w])
    return dhwdA


# In[32]:


def results(Td,Tw,w_w_in,P): # for parametric values calculation
    Twet=psychrolib.GetTWetBulbFromHumRatio(Td, w_w_in, P)
    Tdew=psychrolib.GetTDewPointFromHumRatio(Td,w_w_in, P)
    deweffe=(Td-Tw)/(Td-Tdew)
    weteffe=(Td-Tw)/(Td-Twet)
    Q=(1-ext)*m_d*1005*(Td-Tw)

    Coolcapacity.append(Q)
    T_dry.append(Td)
    T_wet.append(Tw)
    T_DP.append(Tdew)
    T_WB.append(Twet)
    deweffectiveness.append(deweffe)
    weteffectiveness.append(weteffe)
    output=[Td,Tw,Q,weteffe,deweffe,Tdew,Twet,h_Primary[0]]
    return output


# In[33]:


def sat_event(A,hw,m_d,m_w,w_in,params):    # identify if the wet stream reaches a sat. state
    h_d = hw[0]
    h_w = hw[1]
    w_w = hw[2]
    
    P0 = params['Patm']
    
    T_d = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(h_d,w_in) # CP.HumidAirProp.HAPropsSI('T','H',h_d,'P',101325,'W',w_in)
    T_w = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(h_w,w_w)
    RH_w = psychrolib.GetRelHumFromHumRatio(T_w,w_w,P0)
    
    return (params['T0']-T_d)


# In[75]:



channel={}
params = {}
channel['Width']=0.5
channel['Depth']=0.005
channel['length']=0.5
total_area=channel['length']*channel['Width']
params['hydraulic diameter']=2*channel['Depth']
A_cr=channel['Width']*channel['Depth']

params['h_f'] = 22146.5
params['Patm'] = 101325
up=2 # Primary inlet velocity
rho_air_p=1.2 # air density

m_d=rho_air_p*A_cr*up #dry channel mass flow rate
ext=0.3
us=ext*up
m_w=ext*m_d
w = 0.007 # inlet humidity ratio
params['T0']=35
T_out=16
T_d_out = T_out
T_w_in = T_out
w_w_in = w


# In[76]:


# CP.HumidAirProp.HAPropsSI('H','T',T_w_in,'P',101325,'R',RH_w_in)
h_w_in = psychrolib.GetMoistAirEnthalpy(T_w_in, w_w_in)
# CP.HumidAirProp.HAPropsSI('H','T',T_d_out,'P',101325,'W',w_w_in)
h_d_out = psychrolib.GetMoistAirEnthalpy(T_d_out, w_w_in)

hw_0 = np.array([h_d_out, h_w_in, w_w_in])
Aspan = np.array([0, total_area])
t_eval_pts=np.linspace(0,total_area,100)
sat_event.terminal = True
solution = sp.integrate.solve_ivp(dhwdx, Aspan, hw_0,   #t_eval=t_eval_pts,
                                  events=[sat_event],args=(m_d, m_w, w_w_in, params))

h_d = solution.y[0, :]
h_w = solution.y[1, :]
w_w = solution.y[2, :]

t_steps = solution.t

P_array = params['Patm']*np.ones(np.size(h_w))
T_d = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(
    h_d, w_w_in*np.ones(np.size(h_w)))
T_w = psychrolib.GetTDryBulbFromEnthalpyAndHumRatio(h_w, w_w)
RH_w = psychrolib.GetRelHumFromHumRatio(T_w, w_w, P_array)
RH_d = psychrolib.GetRelHumFromHumRatio(T_d, w_w_in, P_array)
calculations=results(T_d[-1], T_w[-1], w_w_in, params['Patm'])


# In[77]:


print(t_steps[-1])
print(total_area)
#print(calculations)
print("The cooling capacity is =",m_d*(1-ext)*1005*(T_d[-1])-T_d[0])
print("Number of channel required to generate 1kW is =", 1000/(m_d*(1-ext)*1005*(T_d[-1])-T_d[0]))
print(m_d*(1-ext)*1005*3600*365*(T_d[-1])-T_d[0])


# In[ ]:




