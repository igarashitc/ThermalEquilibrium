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
ai7 = 2048.0/6435.0
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

def plot(\
    #0:dotm, 1:wt, 2:tem
    yax = 0, \
    clr = "k", \
    #================================================================# 
    #Physical parameter						                         #
    #================================================================#
    #black hole mass
    #----------------------------------------------------------------#
    bhm = 1.0e7, \
    #bhm = 1.0e1, \
    #----------------------------------------------------------------#
    #radius / rs
    #----------------------------------------------------------------#
    r   = 40.0e0, \
    #r   = 50.0e0, \
    #----------------------------------------------------------------#
    #angular momentum at rin
    #Matsumoto et al. 1984
    #----------------------------------------------------------------#
    ellin = 1.7e0, \
    #----------------------------------------------------------------#
    # entropy gradient parameter (e.g., Kato et al. 2008)
    # Q^-_adv = \dot{M}/(2\pi r^2)*W/Sigma*xi
    #----------------------------------------------------------------#
    xi  = 1.0, \
    #----------------------------------------------------------------#
    #alpha viscousity
    #----------------------------------------------------------------#
    #alpha = 0.005, \
    #alpha = 0.01, \
    alpha = 0.03, \
    #alpha = 0.05, \
    #alpha = 0.10, \
    #alpha = 0.30, \
    #alpha = 0.60, \
    #----------------------------------------------------------------#
    # initial magnetic flux
    # \Phi = \Phi_0 (\Sigma/\Sigma_0)^\zeta
    # Different from Oda et al. 2009, 2012
    #----------------------------------------------------------------#
    #Sigma_0
    #s0  = 1.0, \
    #s0  = 60, \
    #s0  = 20, \
    s0  = 10, \
    #s0  = 1e2, \
    #----------------------------------------------------------------#
    #\zeta
    ze  = 0.5, \
    #ze  = 0.6, \
    #ze  = 1.0, \
    #----------------------------------------------------------------#
    #\Phi_0
    #p0 = 3e17 \
    #p0 = 1e17 \
    #p0 = 3e16 \
    #p0 = 1e16 \
    p0 = 8e15 \
    #p0 = 1e15 \
    #p0 = 3e10 \
    #p0 = 0e0 \
    #----------------------------------------------------------------#
    ):
    #================================================================#

    #================================================================#
    #parameters
    #----------------------------------------------------------------#
    #Schwartzchild radius
    rs  = 3.0e5*bhm
    #----------------------------------------------------------------#
    # Eddington luminosity/accretion rate
    #----------------------------------------------------------------#
    ledd = 2e0*np.pi*rs*cc**3/kes
    mded = ledd/cc**2
    #----------------------------------------------------------------#
    #Keplerian rotation 
    #----------------------------------------------------------------#
    omk = np.sqrt(0.5e0/r**3)
    #================================================================#

    #----------------------------------------------------------------#
    #upper limit for RIAF (Abramowicz et al. 1995)
    aa  = -3e0*np.pi*alpha*(r*rs)**2*omk*cc/rs/(xi*mded)
    bb  = 6.2e20*ai65/(2*ai3**2)*((2*np.pi*r*rs)**2*alpha) \
            /(xi*mded**2)*np.sqrt(xmu/(6*rr))
    tmp1  = aa**2/(4e0*bb)
    #----------------------------------------------------------------#

    #----------------------------------------------------------------#
    #RIAF
    #----------------------------------------------------------------#
    sig0  = tmp1*1e-4
    dotm0 = cc*rs/mded*3e0*np.pi*r**2e0*omk/xi*alpha*sig0*2
    dotm1 = np.sqrt(bb*tmp1**3)*100
    tmp = dotm1
    #plt.scatter(sig0,dotm1)
    
    dotm,sig,tem,wt,bt,qm,qc,qa,qv = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)

    print(sig.shape,"riaf")
    plot_fork(sig,dotm,wt,tem,bt,qm,qc,qa,qv,yax=yax,clr=clr)
    #----------------------------------------------------------------# 
    if (sig.shape[0] != 0):
        sig0  = sig[sig.shape[0]-1]
        dotm0 = dotm[dotm.shape[0]-1]
    dotm1 = dotm0*2
    
    dotm,sig,tem,wt,bt,qm,qc,qa,qv = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)
    
    print(sig.shape,"riaf2")
    plot_fork(sig,dotm,wt,tem,bt,qm,qc,qa,qv,yax=yax,clr=clr)
    #-----------------------------------------------------------------#

    #----------------------------------------------------------------#
    #SLE 
    if (sig.shape[0] != 0):
        sig0  = sig[sig.shape[0]-1]*1.2
        dotm0 = dotm[dotm.shape[0]-1]
    dotm1 = dotm0*1e-14
    #plt.scatter(sig0,dotm0,color="k")
    
    dotm,sig,tem,wt,bt,qm,qc,qa,qv = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)
    
    print(sig.shape,"sle")
    plot_fork(sig,dotm,wt,tem,bt,qm,qc,qa,qv,yax=yax,clr=clr)
    #-----------------------------------------------------------------#

    #----------------------------------------------------------------#
    #Magnetized disk
    if (sig.shape[0] != 0):
        sig0  = sig[sig.shape[0]-1]*1.4
        dotm0 = dotm[dotm.shape[0]-1]*1
    #sig0  = 10
    #dotm0 = 0.01
    dotm1 = dotm0*1e12
    #plt.scatter(sig0,dotm0,color="k")
    
    dotm,sig,tem,wt,bt,qm,qc,qa,qv = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)
    
    print(sig.shape,"md")
    plot_fork(sig,dotm,wt,tem,bt,qm,qc,qa,qv,yax=yax,clr=clr)
    #----------------------------------------------------------------#
    
    #-----------------------------------------------------------------#
    #Standard-slim disk
    if (sig.shape[0] != 0):
        sig0  = sig[sig.shape[0]-1]
        dotm0 = dotm[dotm.shape[0]-1]
    dotm1 = dotm0*1e8
    #plt.scatter(sig0,dotm0)
    
    dotm,sig,tem,wt,bt,qm,qc,qa,qv = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)
    
    print(sig.shape,"sad")
    plot_fork(sig,dotm,wt,tem,bt,qm,qc,qa,qv,yax=yax,clr=clr)
    #-----------------------------------------------------------------#

    #-----------------------------------------------------------------#
    #Slim disk
    #-----------------------------------------------------------------#
    sig0  = 1e6
    #sig0 = sig[sig.shape[0]-1]
    dotm0 = cc*rs/mded*3e0*np.pi*r**2e0*omk/xi*alpha*sig0*1.001
    #dotm0 = (9*kes/(128*cc*ai3)*(rr/xmu)**4*alpha*omk*(rs/cc)**2)**1/3e0*2*np.pi*alpha*r**2/(r*r*omk-ellin)/mded*sig0**(5e0/3e0)
    dotm1 = dotm0*1e-8
    #plt.scatter(sig0,dotm0)
    
    dotm,sig,tem,wt,bt,qm,qc,qa,qv = thermal_equil_newton(dotm0, dotm1, sig0,\
        bhm=bhm, r=r, ellin=ellin, xi=xi, alpha=alpha,\
		s0=s0, ze=ze, p0=p0)
    
    print(sig.shape,"slim")
    plot_fork(sig,dotm,wt,tem,bt,qm,qc,qa,qv,yax=yax,clr=clr)
    #-----------------------------------------------------------------#

    return 

def plot_fork(sig, dotm, wt, tem, bt, qm, qc, qa, qv, yax=0, clr="k"):
    
    if (yax == 0):
        plt.plot(sig,dotm,color=clr)
        plt.loglog()
    elif (yax == 1):
        plt.plot(sig,wt,color=clr)
        plt.loglog()
    elif (yax == 2):
        plt.plot(sig,tem,color=clr)
        plt.loglog()
    elif (yax == 3):
        plt.plot(sig,bt,color=clr)
        plt.loglog()
    elif (yax == 4):
        plt.plot(sig,qm,color=clr,linestyle="dashdot")
        plt.plot(sig,qc,color=clr,linestyle="dotted")
        plt.plot(sig,qa,color=clr,linestyle="dashed")
        plt.plot(sig,qv,color=clr)
        #plt.plot(sig,qv-qm-qc-qa,color=clr)
        plt.xscale("log")
    elif (yax == 5):
        plt.plot(sig,tem,color=clr)
        plt.plot(sig,(qm/(aa*cc))**(0.25),color=clr,linestyle="dashed")
        plt.loglog()
    else:
        print("yax=0:accretion rate, yax=1:vertically integrated pressure, yax=2:temperature")

    return

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
    bt_rt   = np.full(1, wt )
    qm_rt   = np.full(1, wt )
    qc_rt   = np.full(1, wt )
    qa_rt   = np.full(1, wt )
    qv_rt   = np.full(1, wt )
    #=================================================================#
   
    for dotm in np.logspace(np.log10(dotm0), np.log10(dotm1), num):
        #vertically integrated total pressure
        wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)

        # iteration for newton
        for i in range(1,20): 
        #Electron temperature
            teme   = min(tem, 1e9)
	    #Disk height
            hh     = 3.0e0*(np.sqrt(wt/sig)/cc)/omk
	    #Electron scattering optical depth
            taues  = 0.5*kes*sig
	    #Absorption optical depth
            tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3*ai7)*(sig*sig/hh)*tem**(-3.5)
            #tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3*ai7)*(sig*sig/hh)*teme**(-3.5)
	    #Total optical depth
            tau    = tauabs+0.5e0*kes*sig
	    #Effective optical depth
            #taueff = np.sqrt(tau*tauabs)
        #Denominator of radiative cooling rate
            qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
	    #Radiative cooling rate
            qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
            #qm     = 4.0e0*aa*cc*ai3*teme**4/qmd
        #Radiative temperature
            ter    = (qm/(aa*cc))**(0.25)
        #Compton cooling rate
            qc     = fac*qm*kes*sig*(ai4/ai3*(tem - ter))
            #qc     = fac*qm*kes*sig*(ai4/ai3*(teme - ter))
        #Vertically integrated gas pressure
            wg     = (ai4/ai3)*(rr/xmu)*sig*tem
        #Vertically integrated radiation pressure
            wr     = (qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
 	    #Vertically integrated magnetic pressure
            wb     = (p0**2*s0**(-2.0*ze)/(8.0e0*np.pi*hh*rs))*sig**(2.0*ze)
        #f1=Q^+ - Q^-_rad - Q^-_adv
            f1     = 1.5e0*alpha*wt*omk-qm*rs/cc-qc*rs/cc-(dotm/(r*r*kes))*((wt-wb)/sig)*xi
        #f2=W_tot - W_mag - W_gas - W_rad
            f2     = wt-wb-(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        #dH/d\Sigma
            dhds   =-0.5e0*hh/sig
        #d\tau/dT
            dtdt   =-3.5e0*tauabs/tem
            #dtdt   =-3.5e0*tauabs/teme
        #d\tau_abs/d\Sigma
            dtads  = 2.5e0*tauabs/sig
        #d\tau/d\Sigma
            dtds   = dtads+0.5e0*kes
        #dQ^-/d\Sigma
            dqds   =-qm*(1.5e0*dtds-dtads/(tauabs**2e0))/qmd
        #dQ^-/dT
            #dqdt   = qm*(4.e0/tem-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
            dqdt   = qm*(4.e0/teme-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
        #dW_b/d\Sigma
            dwbds  = wb*(2.0e0*ze+0.5e0)/sig
        #dT_r/d\Sigma
            dtrds  = 0.25e0*ter/qm*dqds
        #dT_r/dT
            dtrdt  = 0.25e0*ter/qm*dqdt
        #df1/d\Sigma
            df1ds  =-dqds*rs/cc+(dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds \
                    -fac*((kes*sig*dqds + kes*qm)*(ai4/ai3*tem - ter) - kes*sig*qm*dtrds)*rs/cc
            #df1ds  =-dqds*rs/cc+(dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds \
            #        -fac*((kes*sig*dqds + kes*qm)*(ai4/ai3*teme - ter) - kes*sig*qm*dtrds)*rs/cc
        #df1/dT
            df1dt  =-dqdt*rs/cc \
                    -fac*kes*sig*(dqdt*(ai4/ai3*tem - ter) + qm*(ai4/ai3 - dtrdt))*rs/cc
            #df1dt  =-dqdt*rs/cc \
            #        -fac*kes*sig*(dqdt*(ai4/ai3*teme - ter) + qm*(ai4/ai3 - dtrdt))*rs/cc
        #df2/d\Sigma
            df2ds  =-(ai4/ai3)*(rr/xmu)*tem-(1.0e0/(4.0e0*cc))*(ai4/ai3)*dqds*hh*rs*(tau+2.0e0/sqr3) \
                    -(qm/(4.0e0*cc))*(ai4/ai3)*(dhds*rs*(tau+2.0e0/sqr3)+hh*rs*dtds)-dwbds
        #df2/dT
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

                wb    = (p0**2*s0**(-2.0*ze)/(8.0e0*np.pi*hh*rs))*sig**(2.0*ze)
                bt_rt = np.append(bt_rt, wt/wb - 1.e0)
                
                tauabs = (6.2e20/(2.0e0*aa*cc*rs))*(ai65/(ai3*ai7))*(sig*sig/hh)*tem**(-3.5e0)
                tau    = tauabs+0.5e0*kes*sig
                qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
                qm     = 4.0e0*aa*cc*ai3*tem**4/qmd*rs/cc
                qm_rt  = np.append(qm_rt, qm)

                ter    = (qm/(aa*cc))**(0.25)
                qc     = fac*qm*kes*sig*(ai4/ai3*(tem - ter))
                qc_rt  = np.append(qc_rt, qc)

                qa     = (dotm/(r*r*kes))*((wt-wb)/sig)*xi
                qa_rt  = np.append(qa_rt, qa)

                qv     = 1.5e0*alpha*wt*omk
                qv_rt  = np.append(qv_rt, qv)

                cnt = cnt+1
                break
            #else:
                #sig = sig_rt[cnt]
                #tem = tem_rt[cnt]

    sig_rt = np.delete(sig_rt, 0)
    tem_rt = np.delete(tem_rt, 0)
    wt_rt  = np.delete(wt_rt,  0)
    bt_rt  = np.delete(bt_rt,  0)
    qm_rt  = np.delete(qm_rt,  0)
    qc_rt  = np.delete(qc_rt,  0)
    qa_rt  = np.delete(qa_rt,  0)
    qv_rt  = np.delete(qv_rt,  0)
    dotm_rt = wt_rt/( (ell-ellin)*(cc*cc/kes)/(r*r*alpha) )
    
    return dotm_rt,sig_rt,tem_rt,wt_rt,bt_rt,qm_rt,qc_rt,qa_rt,qv_rt
