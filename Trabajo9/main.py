# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 12:34:27 2021

@author: myriam
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from newton import newton


plt.close('all')
# definicion de paramtros

eps_0=8.854187817e-12       # Permitividad en espacio libre
eps_s=5.25
eps_inf=2.25
w0=8e14             #(en sec^-1)
wc=8.61e14
mu_0=4*np.pi*1e-7         # Permeabilidad en espacio libre
sigma=4e9             # Conductividad electrica 
c=1/np.sqrt(mu_0*eps_0)
tau_1=12.2e-15   
tau_2=32e-15
t_delay=41.7e-15
T_w=3.5e-15
dto=0.99

# Tamaño de la malla
grid_size=70

# Discretizacion espacial
dx=5e-9

# Paso temporal
dt=1/(c*np.sqrt((1/dx**2)))

# Condicion CFL: c * dt * sqrt((1/dx^2)+(1/dy^2)+(1/dz^2))<=1
courant=0.8
dt=courant*dt


t_steps=25000  # iteraciones
t=(np.linspace(1,t_steps,t_steps)-0.5)*dt


# Valores iniciales para las funciones chi y alpha
# Linealidad tipo Raman
chi_0=2.0      #chi0^3 en (V/m)^-2
alpha0=0.0

gamma_NL=1/(tau_2)+1j/(tau_1)
gamma_L=sigma+1j*np.sqrt(w0**2-sigma**2)

# funcion chi:

def function_chi_3(t):
    return chi_0*(tau_1**2+tau_2**2)/(tau_1*tau_2**2)*np.exp(-t/tau_2)*np.sin(t/tau_1)

def function_chi_1(t):
    return (eps_0-eps_inf)*w0**2/(np.sqrt(w0**2-sigma**2))*np.exp(-sigma*t)*np.sin(np.sqrt(w0**2-sigma**2)*t)

def function_chi_1_v2(t):
    return (dt-t)*(eps_0-eps_inf)*w0**2/(np.sqrt(w0**2-sigma**2))*np.exp(-sigma*t)*np.sin(np.sqrt(w0**2-sigma**2)*t)


def function_chi_3_v2(t):
    return (dt-t)*chi_0*(tau_1**2+tau_2**2)/(tau_1*tau_2**2)*np.exp(-t/tau_2)*np.sin(t/tau_1)

def function_chi_3_v3(t):
    return ((dt-t)**2)*chi_0*(tau_1**2+tau_2**2)/(tau_1*tau_2**2)*np.exp(-t/tau_2)*np.sin(t/tau_1)

# campo incidente
def E_inc(t,E0):
    return E0*np.cos(wc*(t-t_delay))*1/(np.cosh((t-t_delay)/T_w))

# Pulso incidente gaussiano
def gauss(t):
    g=6.2e14
    tdum=4.23e-15
    return np.exp(-(g*(t-tdum))**2)

# Calculo de las susceptibilidades

# Valores iniciales
# Para obtener el valor inicial de psi, hacemos la integral de chi

psi_0_L, err=quad(function_chi_1,0,dt)

psi_1_L, err=quad(function_chi_1_v2,0,dt)
psi_1_L=(1/dt)*psi_1_L

psi_0_NL, err=quad(function_chi_3,0,dt)

psi_1_NL, err=quad(function_chi_3_v2,0,dt)
psi_1_NL=(1/dt)*psi_1_NL

psi_2_NL, err=quad(function_chi_3_v3,0,dt)
psi_2_NL=(1/dt**2)*psi_2_NL


# Inicializacion de funciones PSI
psi_L0=np.zeros((grid_size,1))
psi_L0[0]=psi_0_L

psi_L1=np.zeros((grid_size,1))
psi_L1[0]=psi_1_L

psi_NL0=np.zeros((grid_size,1))
psi_NL0[0]=psi_0_NL

psi_NL1=np.zeros((grid_size,1))
psi_NL1[0]=psi_1_NL

psi_NL2=np.zeros((grid_size,1))
psi_NL2[0]=psi_2_NL
  


#E=E_inc(t,1)
E=gauss(t)


# Inicializacion de vectores de campo electrico, magnetico y coeficientes
e = np.zeros((grid_size+2,1))
h = np.zeros((grid_size+2,1))

a0=np.zeros((grid_size+1,1))
a1=np.zeros((grid_size+1,1))
a2=np.zeros((grid_size+1,1))
a3=np.zeros((grid_size+1,1))


P_L=np.zeros((grid_size+1,1))
P_NL=np.zeros((grid_size+1,1))

cH = dt / mu_0 / dx
cE = dt / eps_0 / dx

m_in=11


for i in range(0, t_steps+1):
    old1=e[1]
    old2=e[grid_size]
    
    for m in range(1,grid_size):
        
        e[m]= e[m]+dto*(h[m]-h[m-1])
        
    e[int(grid_size/2)]=e[int(grid_size/2)]+np.sin(0.03*i*dto)
   
    for m in range(0,grid_size):
        h[m]=h[m] +dto*(e[m+1]-e[m])
       
   
    h[int(grid_size/2)]=h[int(grid_size/2)]+np.sin(0.03*(i+0.5)*dto)
    e[0]=old1
    e[grid_size+1]=old2
    
    for i in range(0,grid_size-1):

        psi_L0[i+1]=psi_L0[i]*np.exp(-gamma_L*dt)
        
        psi_L1[i+1]=psi_L1[i]*np.exp(-gamma_L*dt)
        
        psi_NL0[i+1]=psi_NL0[i]*np.exp(-gamma_NL*dt)
        
        psi_NL1[i+1]=psi_NL1[i]*np.exp(-gamma_NL*dt)
        
        psi_NL2[i+1]=psi_NL2[i]*np.exp(-gamma_NL*dt)

        
    # Campo electrico (en x) el medio no lineal empieza en grid size=12

    for k in range(m_in,grid_size-1):
        
        new1=e[m_in+1]
        new2=e[grid_size-1]
    
        P_L[k+1]=P_L[k]*np.exp(-gamma_L*dt)+ e[k]*np.real(psi_L0[k])-(e[k+1]-e[k])*np.real(psi_L1[k])
        P_NL[k+1]=P_NL[k]*np.exp(-gamma_NL*dt)+ (e[k]**2)*psi_NL0[k]-2*e[k]*(e[k+1]-e[k])*psi_NL1[k]+((e[k+1]-e[k])**2)*psi_NL2[k]
        a0[k+1]=-(dt/dx)*(h[k]-h[k-1])+(sigma*dt/2-eps_0*eps_inf)*e[k]-eps_0*(e[k]*psi_L1[k]+(1-np.exp(-gamma_L*dt))*P_L[k])-eps_0*e[k]*(e[k]**2*alpha0+P_NL[k])
        a1[k+1]=sigma*dt/2+eps_0*eps_inf+eps_0*(psi_L0[k]-psi_L1[k])+ eps_0*((e[k]**2)*psi_NL2[k]+np.exp(-gamma_NL*dt)*P_NL[k])
        a2[k+1]=2*eps_0*e[k]*(psi_NL1[k]-psi_NL2[k])
        a3[k+1]=eps_0*(alpha0+psi_NL0[k]-2*psi_NL1[k]+psi_NL2[k])
    
        f = lambda x: np.real(a0[k])+np.real(a1[k])*x+np.real(a2[k])*x**2+np.real(a3[k])*x**3
        df= lambda x: np.real(a1[k])+2*np.real(a2[k])*x+3*np.real(a3[k])*x**2
           
        e[k]=newton(f,df,e[0],epsilon=10.0e-4,max_iter=6)
        
        
        e[10] += E[k]
        
    e[m_in]=new1
    e[grid_size]=new2
    
        
    #print(e[k])    
    plt.plot(e,color='m')
    plt.pause(0.7)
    plt.clf()


#%% ESTO FUNCIONA
# CONDICIONES ABSORBENTES. NO SE REFLEJA EL CAMPO
import numpy as np
import matplotlib.pyplot as plt
import time


plt.close('all')
nx=1000
Eze=np.zeros((nx+2,1))
Hy=np.zeros((nx+2,1))


dt=0.99

plt.ion()
figure, ax=plt.subplots(figsize=(10,8))

number_of_time_steps=2000
        
for time_step in range(0,number_of_time_steps+1):
    
    viejo1=Eze[1]
    viejo2=Eze[nx]
    for i in range(1,nx):
        #print(i)
        Eze[i]=Eze[i]+dt*(Hy[i]-Hy[i-1])
        
    
    Eze[int(nx/2)]=Eze[int(nx/2)]+np.sin(0.03*time_step*dt)
   
   
   
    
    for i in range(0,nx):
        Hy[i]=Hy[i] +dt*(Eze[i+1]-Eze[i])
       
   
    Hy[int(nx/2)]=Hy[int(nx/2)]+np.sin(0.03*(time_step+0.5)*dt)
    Eze[0]=viejo1
    Eze[nx+1]=viejo2
    
    plt.plot(Eze,color='k')
    plt.ylim(-1,1)
    
    figure.canvas.draw()

    figure.canvas.flush_events()
  
    time.sleep(1e-23)
    plt.clf()
       
#    plt.plot(Hy[1:int(nx/2)],'r')
#    plt.pause(0.002)
#    plt.clf() 
        

  

    
    
    
    
    
    
    