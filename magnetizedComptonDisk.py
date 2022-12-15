#==============================================================#
#  magnetizedDisk.py based on adf5.py				           #
#  Thermal equilibrium curves of accretion flows  		       #
#          magnetic flux formulation			               # 
#							                                   #
#                                 2021.12.28 by RM             #
#                          revisede for python 2021.1.2 by TI  #
#		     implement Compton cooling 2022.12.8 by TI         #
#==============================================================#    

#==============================================================#
import numpy as np                                             
import matplotlib.pyplot as plt                                          
#==============================================================#

#==============================================================#
# Constatnts		     				                       #
#==============================================================#
sqr3= np.sqrt(3.0e0)
#
#I_N = (2^N*N!)/((2N+1)!)
#N = 3 e.g., Oda et al. 2009
#
ai3 = 16.0/35.0
ai4 = 128.0/315.0
ai65= 0.33   # not exact
#
#gas constant
rr  = 8.3145e7
#mean molecular weight
xmu = 0.5e0
#electron scattering opacity
kes = 0.40e0
#radiation constant
aa  = 7.5646e-15
#speed of light
cc  = 2.9979e10
#gravitational constatn
gg  = 6.6725e-8
#electron mass 
me  = 9.1094e-28
#boltzmann constant
kb  = 1.3807e-16
#for Compton cooling
fac = 4e0*kb/(me*cc*cc)
#================================================================#

#===================================================================#
# Calcurate thermal equilibrium solution,
# Q^+_vis = Q^-_rad + Q^-_adv,
# by newton method.
# The basic equations are same as Oda et al. 2009 
# except for the determination of the magnetic flux.
#===================================================================#
def thermal_equil_newton(dotm0, dotm1, sig0, \
		bhm=1e7, r=40e0, ellin=1.7e0, xi=1e0, alpha=0.03,\
		s0=10, ze=0.5, p0=1e17):

    #================================================================# 
    #Physical parameter						                         #
    #================================================================#
    #----------------------------------------------------------------#
    #Schwartzchild radius
    rs  = 3.0e5*bhm
    #----------------------------------------------------------------#
    #Keplerian rotation 
    omk = np.sqrt(0.5e0/r**3)
    #----------------------------------------------------------------#
    #angular momentum
    ell = r*r*omk
    #================================================================#

    #================================================================#
    # initial guess 
    #================================================================#
    num = 2000
    cnt = 0

    sig = sig0
    wt  = dotm0*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
    tem = wt/((ai4/ai3)*(rr/xmu)*sig)

    sig_rt  = np.full(1, sig)
    tem_rt  = np.full(1, tem)
    wt_rt   = np.full(1, wt )
    #=================================================================#
   
    for dotm in np.logspace(np.log10(dotm0), np.log10(dotm1), num):
        wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
        #tem = wt/((ai4/ai3)*(rr/xmu)*sig)
        #tem05 = (wt/((ai4/ai3)*(rr/xmu)*sig))**0.5

        # iteration for newton
        for i in range(1,20): 
	    #Disk height
            hh     = 3.0e0*(np.sqrt(wt/sig)/cc)/omk
 	    #Vertically integrated magnetic pressure
            wb     = (p0**2*s0**(-2.0*ze)/(8.0e0*np.pi*hh*rs))*sig**(2.0*ze)
	    #Electron scattering optical depth
            taues  = 0.5*kes*sig
	    #Absorption optical depth
            tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
	    #Total optical depth
            tau    = tauabs+0.5e0*kes*sig
	    #Effective optical depth
            taueff = np.sqrt(tau*tauabs)
            qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
	    #Radiative cooling rate
            qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
            ter    = (qm*rs/(aa*cc**2))**(0.25)
            f1     = 1.5e0*alpha*wt*omk-(dotm/(r*r*kes))*((wt-wb)/sig)*xi-qm*rs/cc*(1+sig*kes*fac*(tem - ter)) 
            f2     = wt-wb-(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
            dhds   =-0.5e0*hh/sig
            dtdt   =-3.5e0*tauabs/tem
            dtads  = 2.5e0*tauabs/sig
            dtds   = dtads+0.5e0*kes
            dqds   =-qm*(1.5e0*dtds-dtads/(tauabs**2))/qmd
            dqdt   = qm*(4/tem-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
            dwbds  = wb*(2.0e0*ze+0.5e0)/sig
            dtrds  = 0.25*ter/qm*dqds
            dtrdt  = ter/qm*dqdt
            df1ds  = (dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds \
                    -dqds*rs/cc*(1e0+sig*kes*fac*(tem-ter)) - qm*rs/cc*kes*fac*(tem - ter - sig*dtrds)
            df1dt  =-dqdt*rs/cc*(1+sig*kes*fac*(tem-ter)) - qm*rs/cc*sig*kes*fac*(1e0 - dtrdt)
            df2ds  =-(ai4/ai3)*(rr/xmu)*tem-(1.0e0/(4.0e0*cc))*(ai4/ai3)*dqds*hh*rs*(tau+2.0e0/sqr3) \
                    -(qm/(4.0e0*cc))*(ai4/ai3)*(dhds*rs*(tau+2.0e0/sqr3)+hh*rs*dtds)-dwbds
            df2dt  =-(ai4/ai3)*(rr/xmu)*sig-(1.0e0/(4.0e0*cc))*dqdt*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3) \
            	    -(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*dtdt
            dd     = df1ds*df2dt-df1dt*df2ds
            dsig   =-(f1*df2dt-f2*df1dt)/dd 
            dtem   =-(df1ds*f2-df2ds*f1)/dd
            sig    = sig+dsig
            tem    = tem+dtem
            pgas   = (ai4/ai3)*(rr/xmu)*sig*tem
            prad   = (qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)

            # convergence check
            if (abs(dsig/sig) < 1.0e-10):
                sig_rt = np.append(sig_rt, sig)
                tem_rt = np.append(tem_rt, tem)
                wt_rt = np.append(wt_rt, wt)
                cnt = cnt+1
                break
            #else:
                #sig = sig_rt[cnt]
                #tem = tem_rt[cnt]

    sig_rt = np.delete(sig_rt, 0)
    tem_rt = np.delete(tem_rt, 0)
    wt_rt  = np.delete(wt_rt,  0)
    dotm_rt = wt_rt/( (ell-ellin)*(cc*cc/kes)/(r*r*alpha) )
    
    return dotm_rt,sig_rt,tem_rt,wt_rt

def calc():
    #================================================================# 
    #Physical parameter						     #
    #================================================================#
    #black hole mass
    #----------------------------------------------------------------#
    bhm = 1.0e7
    #bhm = 1.0e1
    #----------------------------------------------------------------#
    #Schwartzchild radius
    rs  = 3.0e5*bhm
    #----------------------------------------------------------------#
    # Eddington luminosity/accretion rate
    #----------------------------------------------------------------#
    ledd = 2e0*np.pi*rs*cc**3/kes
    mded = ledd/cc**2
    #----------------------------------------------------------------#
    #radius / rs
    #----------------------------------------------------------------#
    r   = 40.0e0
    #r   = 50.0e0
    #----------------------------------------------------------------#
    #Keplerian rotation 
    #----------------------------------------------------------------#
    omk = np.sqrt(0.5e0/r**3)
    #----------------------------------------------------------------#
    #----------------------------------------------------------------#
    #angular momentum at rin
    #Matsumoto et al. 1984
    #----------------------------------------------------------------#
    ellin = 1.7e0
    #----------------------------------------------------------------#
    # entropy gradient parameter (e.g., Kato et al. 2008)
    # Q^-_adv = \dot{M}/(2\pi r^2)*W/Sigma*xi
    #----------------------------------------------------------------#
    xi  = 1.0
    #----------------------------------------------------------------#
    #alpha viscousity
    #----------------------------------------------------------------#
    #alpha = 0.005
    #alpha = 0.01
    alpha = 0.03
    #alpha = 0.05
    #alpha = 0.10
    #alpha = 0.30
    #alpha = 0.60
    #----------------------------------------------------------------#
    # initial magnetic flux
    # \Phi = \Phi_0 (\Sigma/\Sigma_0)^\zeta
    # Different from Oda et al. 2009, 2012
    #----------------------------------------------------------------#
    #Sigma_0
    #s0  = 1.0
    #s0  = 60
    #s0  = 20
    #s0  = 10
    s0  = 1e2
    #----------------------------------------------------------------#
    #\zeta
    ze  = 0.5
    #ze  = 0.6
    #ze  = 1.0
    #----------------------------------------------------------------#
    #\Phi_0
    p0 = 1e17
    #p0 = 3e16
    #p0 = 8e15
    #p0 = 3e15
    #================================================================#

    #for RIAF
    aa  = -1.5e0*2*np.pi*alpha*(r*rs)**2*omk*cc/rs/(xi*mded)
    bb  = 6.2e20*ai65/(2*ai3**2)*((2*np.pi*r*rs)**2*alpha)/(xi*mded**2)*np.sqrt(xmu/(6*rr))
    tmp1 = aa**2/(4e0*bb)
    tmp2 = np.sqrt(bb*tmp1**3)
    dotm0 = tmp2
    dotm1 = dotm0*1e-2
    sig0  = tmp1
    
    dotm,sig,wt,tem = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)

    return dotm,sig,wt,tem

def plot_theq():
    #================================================================# 
    #Physical parameter						     #
    #================================================================#
    #black hole mass
    #----------------------------------------------------------------#
    bhm = 1.0e7
    #bhm = 1.0e1
    #----------------------------------------------------------------#
    #Schwartzchild radius
    rs  = 3.0e5*bhm
    #----------------------------------------------------------------#
    # Eddington luminosity/accretion rate
    #----------------------------------------------------------------#
    ledd = 2e0*np.pi*rs*cc**3/kes
    mded = ledd/cc**2
    #----------------------------------------------------------------#
    #radius / rs
    #----------------------------------------------------------------#
    r   = 40.0e0
    #r   = 50.0e0
    #----------------------------------------------------------------#
    #Keplerian rotation 
    #----------------------------------------------------------------#
    omk = np.sqrt(0.5e0/r**3)
    #----------------------------------------------------------------#
    #----------------------------------------------------------------#
    #angular momentum at rin
    #Matsumoto et al. 1984
    #----------------------------------------------------------------#
    ellin = 1.7e0
    #----------------------------------------------------------------#
    # entropy gradient parameter (e.g., Kato et al. 2008)
    # Q^-_adv = \dot{M}/(2\pi r^2)*W/Sigma*xi
    #----------------------------------------------------------------#
    xi  = 1.0
    #----------------------------------------------------------------#
    #alpha viscousity
    #----------------------------------------------------------------#
    #alpha = 0.005
    #alpha = 0.01
    alpha = 0.03
    #alpha = 0.05
    #alpha = 0.10
    #alpha = 0.30
    #alpha = 0.60
    #----------------------------------------------------------------#
    # initial magnetic flux
    # \Phi = \Phi_0 (\Sigma/\Sigma_0)^\zeta
    # Different from Oda et al. 2009, 2012
    #----------------------------------------------------------------#
    #Sigma_0
    #s0  = 1.0
    #s0  = 60
    #s0  = 20
    #s0  = 10
    s0  = 1e2
    #----------------------------------------------------------------#
    #\zeta
    ze  = 0.5
    #ze  = 0.6
    #ze  = 1.0
    #----------------------------------------------------------------#
    #\Phi_0
    #p0 = 3e17
    p0 = 1e17
    #p0 = 3e16
    #p0 = 8e15
    #p0 = 3e15
    #================================================================#

    #RIAF
    aa  = -3e0*np.pi*alpha*(r*rs)**2*omk*cc/rs/(xi*mded)
    bb  = 6.2e20*ai65/(2*ai3**2)*((2*np.pi*r*rs)**2*alpha)/(xi*mded**2)*np.sqrt(xmu/(6*rr))
    tmp1 = aa**2/(4e0*bb)
    tmp2 = np.sqrt(bb*tmp1**3)
    
    dotm0 = np.sqrt(bb*(tmp1*1e-1)**3)
    dotm1 = np.sqrt(bb*tmp1**3)
    sig0  = tmp1*1e-2
    print(tmp1,tmp2)
    
    dotm,sig,wt,tem = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)
    plt.plot(sig,dotm,color="k")
    #plt.plot(sig,dotm,color="k",linestyle="dashdot")

    #SLE
    dotm1 = np.sqrt(bb*tmp1**3)
    dotm0 = 1e-3
    sig0  = 1
    print(tmp1,dotm0)
    
    dotm,sig,wt,tem = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)
    plt.plot(sig,dotm,color="k")
    #plt.plot(sig,dotm,color="k",linestyle="dashdot")

    #Standard-slim disk
    dotm0 = 1e-3
    dotm1 = 1e4
    sig0  = 1e3
    
    dotm,sig,wt,tem = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)
    #plt.plot(sig,dotm,color="k")
    plt.plot(sig,tem,color="k")
    #plt.plot(sig,dotm,color="k",linestyle="dashdot")
   
    plt.loglog()

    return 
