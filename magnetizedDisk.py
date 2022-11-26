#!===================================================
#!  magnetizedDisk.py based on adf5.py
#!  Thermal equilibrium curves of accretion flows  
#!          magnetic flux formulation 
#!
#!                                 2021.12.28 by RM
#!                          revisede for python 2021.1.2 by TI
#!===================================================    

#============================#
# Constatnts		     #
#============================#
sqr3= sqrt(3.0e0)
rr  = 8.3145e7
#
#I_N = 2^NN!/(2N+1)!
#N = 3 e.g., Oda et al. 2009
#
ai3 = 16.0/35.0
ai4 = 128.0/315.0
ai65= 0.33   # not exact
#
#mean molecular weight
xmu = 0.5e0
#electron scattering opacity
kes = 0.40e0
#radiation constant
aa  = 7.5646e-15
#speed of light
cc  = 2.9979e10
#=======================#

#=======================# 
#Physical parameter
#=======================#
#black hole mass
bhm = 1.0e7
#bhm = 1.0e1
#-----------------------#
#Schwartzchild radius
rs  = 3.0e5*bhm
#-----------------------#
#radius / rs
r   = 40.0e0
#r   = 50.0e0
#-----------------------#
#Keplerian rotation 
omk = sqrt(0.5e0/r**3)
#-----------------------3
#angular momentum
ell = r*r*omk
#angular momentum at rin
#Matsumoto et al. 1984
ellin = 1.7e0
#-----------------------3
# parameter in advection cooling
xi  = 1.0
#-----------------------3
#alpha viscousity
#alpha = 0.005
#alpha = 0.01
alpha = 0.03
#alpha = 0.05
#alpha = 0.10
#alpha = 0.30
#alpha = 0.60
#-----------------------3
#plasma beta
#bt  = 1000.0
#bt  = 100.0
#bt  = 10.0
#bt  = 1.0
#bt  = 0.5
#bt  = 0.1
#bt1 = 1.0e0+1.0e0/bt

#-----------------------3
# initial magnetic flux
# \Phi = \Phi_0 (\Sigma/\Sigma_0)^\zeta
# Different from Oda et al. 2009, 2012
#-----------------------3
#Sigma_0
#s0  = 1.0
#s0  = 60
#s0  = 20
#s0  = 10
s0  = 1e2
#-----------------------3
#\zeta
ze  = 0.5
#ze  = 0.6
#ze  = 1.0
#-----------------------3
#\Phi_0
p0 = 3e17
#p0 = 3e16
#p0 = 8e15
#p0 = 3e15
#-----------------------3
#========================

dotm=1.0e-3
wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
h0 = sqrt(wt/s0)*(rs/cc)/omk
#!     p0 = sqrt(wt/s0)*sqrt(4.0e0*pi*s0*h0/bt)
#!     write(6,*) 'initial magnetic flux = ',p0

num = 2000

#!     SLE
# Initial guess #
#dotm = 1.0e-1
#tem  = 1.0e8
#sig  = 3.0e0
sig  = 1e1
#sig  = 8
wt   = ((6.2e20*(ai65/(ai3*ai3)))/(9.0e0*alpha*sqrt((ai4/ai3)*rr/xmu)))*sig*sig
tem  = wt/((ai4/ai3)*(rr/xmu)*sig)
dotm = wt/((ell-ellin)*(cc*cc/kes)/(r*r*alpha))
################
# Define array #
sig_sle = np.full(2, sig)
wt_sle  = np.full(2, wt)
tem_sle = np.full(2, tem)
#dotm_sle = dotm*np.logspace(-0.1, -4, num)
dotm_sle = dotm*np.logspace(-8, 0.23, num)
################
#
cnt = 1
print(tem)
#
for j in reversed(range(num)):
#for j in range(1,num-1):
    dotm = dotm_sle[j]
    wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
    for i in range(1,20):
        hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
        wb     = (p0**2*s0**(-2.0*ze)/(8.0e0*pi*hh*rs))*sig**(2.0*ze)
        taues  = 0.5*kes*sig
        tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
        tau    = tauabs+0.5e0*kes*sig
        taueff = sqrt(tau*tauabs)
        qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
        qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
        f1     = 1.5e0*alpha*wt*omk-qm*rs/cc-(dotm/(r*r*kes))*((wt-wb)/sig)*xi
        f2     = wt-wb-(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        dhds   =-0.5e0*hh/sig
        dtdt   =-3.5e0*tauabs/tem
        dtads  = 2.5e0*tauabs/sig
        dtds   = dtads+0.5e0*kes
        dqds   =-qm*(1.5e0*dtds-dtads/(tauabs**2))/qmd
        dqdt   = qm*(4.0e0/tem-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
        dwbds   = wb*(2.0e0*ze+0.5e0)/sig
        df1ds  =-dqds*rs/cc+(dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds
        df1dt  =-dqdt*rs/cc
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
#        write(6,*) dotm,sig,tem
        #if (abs(dsig/sig) < 1.0e-10) write(6,*) sig,dotm,tem,taueff,prad/pgas,wt,dsig/sig,dtem/tem
    if (abs(dsig/sig) < 1.0e-10):
        sig_sle = np.append(sig_sle, sig)
        wt_sle  = np.append(wt_sle , wt )
        tem_sle = np.append(tem_sle, tem)
        cnt     = cnt+1
    else:
        sig = sig_sle[cnt]
        tem = tem_sle[cnt]

#!    SLE2
# Initial guess #
#dotm = 1.0e-1
#tem  = 1.0e8
#sig  = 10.0e0
#sig  = 3.0*10**(-0.1)
#sig  = 3.0*10
sig  = 7
wt   = ((6.2e20*(ai65/(ai3*ai3)))/(9.0e0*alpha*sqrt((ai4/ai3)*rr/xmu)))*sig*sig
tem  = wt/((ai4/ai3)*(rr/xmu)*sig)
dotm = wt/((ell-ellin)*(cc*cc/kes)/(r*r*alpha))
#################
# Define array #
sig_sle2  = np.full(2, sig)
wt_sle2   = np.full(2, wt)
tem_sle2  = np.full(2, tem)
dotm_sle2 = dotm*np.logspace(-2,1,num)
#dotm_sle2 = dotm*np.logspace(-2,4,num)
#################
#
cnt = 1
#
for j in range(1,num):
    dotm = dotm_sle2[j]
    wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
    for i in range(1,20):
        hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
        wb     = (p0**2*s0**(-2.0*ze)/(8.0e0*pi*hh*rs))*sig**(2.0*ze)
        taues  = 0.5*kes*sig
        tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
        tau    = tauabs+0.5e0*kes*sig
        taueff = sqrt(tau*tauabs)
        qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
        qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
        f1     = 1.5e0*alpha*wt*omk-qm*rs/cc-(dotm/(r*r*kes))*((wt-wb)/sig)*xi
        f2     = wt-wb-(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        dhds   =-0.5e0*hh/sig
        dtdt   =-3.5e0*tauabs/tem
        dtads  = 2.5e0*tauabs/sig
        dtds   = dtads+0.5e0*kes
        dqds   =-qm*(1.5e0*dtds-dtads/(tauabs**2))/qmd
        dqdt   = qm*(4.0e0/tem-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
        dwbds   = wb*(2.0e0*ze+0.5e0)/sig
        df1ds  =-dqds*rs/cc+(dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds
        df1dt  =-dqdt*rs/cc
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
#       write(6,*) dotm,sig,tem
    if (abs(dsig/sig) < 1.0e-10): 
        sig_sle2 = np.append(sig_sle2, sig)
        tem_sle2 = np.append(tem_sle2, tem)
        wt_sle2  = np.append(wt_sle2 , wt )
        cnt      = cnt + 1
        #write(6,*) sig,dotm,tem,taueff,prad/pgas,wt,dsig/sig,dtem/tem
    else:
        sig = sig_sle2[cnt]
        tem = tem_sle2[cnt]

#!     RIAF
# Initial guess #
dotm = 1.0e-4
wt   = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
#!!    sig  = (dotm*xi/(r*r*kes*bt1))/(1.5e0*alpha*omk)
#!!    tem  = wt/((ai4/ai3)*(rr/xmu)*sig*bt1)
sig  = wt/(1.5e0*omk*(ell-ellin)*cc*cc/xi)
tem  = wt/((rr/xmu)*sig)
##################
# Define array #
dotm_riaf = dotm*np.logspace(-2,5,num)
sig_riaf = np.full(2, sig)
wt_riaf  = np.full(2, wt)
tem_riaf = np.full(2, tem)
################
#for j in range(1,80):
#
cnt = 1
#
for j in range(num):
    dotm = dotm_riaf[j]
    wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
    for i in range(1,20):
        hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
        wb     = (p0**2*s0**(-2.0*ze)/(8.0e0*pi*hh*rs))*sig**(2.0*ze)
        taues  = 0.5*kes*sig
        tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
        tau    = tauabs+0.5e0*kes*sig
        taueff = sqrt(tau*tauabs)
        qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
        qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
        f1     = 1.5e0*alpha*wt*omk-qm*rs/cc-(dotm/(r*r*kes))*((wt-wb)/sig)*xi
        f2     = wt-wb-(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        dhds   =-0.5e0*hh/sig
        dtdt   =-3.5e0*tauabs/tem
        dtads  = 2.5e0*tauabs/sig
        dtds   = dtads+0.5e0*kes
        dqds   =-qm*(1.5e0*dtds-dtads/(tauabs**2))/qmd
        dqdt   = qm*(4.0e0/tem-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
        dwbds  = wb*(2.0e0*ze+0.5e0)/sig
        df1ds  =-dqds*rs/cc+(dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds
        df1dt  =-dqdt*rs/cc
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
#!       write(6,*) dotm,sig,tem
    if (abs(dsig/sig) < 1.0e-10):
        sig_riaf = np.append(sig_riaf,sig)
        tem_riaf = np.append(tem_riaf,tem)
        wt_riaf  = np.append(wt_riaf ,wt )
        cnt      = cnt + 1
        #write(6,*) sig,dotm,tem,taueff,prad/pgas,wt,dsig/sig,dtem/tem
    else:
        sig = sig_riaf[cnt]
        tem = tem_riaf[cnt]

#!     SADM
# Initial guess #
dotm = 1.0e-2
wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
#!!    tem = (wt/bt1)*1.5*alpha*wt*omk*1.5e0*0.5e0*kes/((rr/xmu)*ai4*4.0e0*aa*rs)
tem = wt*1.5*alpha*wt*omk*1.5e0*0.5e0*kes/((rr/xmu)*ai4*4.0e0*aa*rs)
tem = tem**0.2
#!     sig = (wt/bt1)/((ai4/ai3)*(rr/xmu)*tem)
#sig = wt/((ai4/ai3)*(rr/xmu)*tem)
sig = 100
#!     write(6,*) 'SADM',' sig = ',sig,' tem = ',tem
#!     sig = (wt*wt*wt*32.0e0*aa*ai3*rs/(9.0e0*alpha*omk*kes*(rr/xmu)**4))**0.2
#!     tem = wt/((rr/xmu)*sig)
##################
# Define array #
dotm_sadm = np.logspace(-7,4,num)
#dotm_sadm = np.logspace(-3,4,num)
sig_sadm = np.full(2, sig)
wt_sadm  = np.full(2, wt)
tem_sadm = np.full(2, tem)
################
#
cnt = 1
#
#for j in range(21,80):
for j in range(num):
    dotm = dotm_sadm[j]
    wt   = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
    for i in range(1,20):
        hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
        wb     = (p0**2*s0**(-2.0*ze)/(8.0e0*pi*hh*rs))*sig**(2.0*ze)
        taues  = 0.5*kes*sig
        tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
        tau    = tauabs+0.5e0*kes*sig
        taueff = sqrt(tau*tauabs)
        qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
        qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
        f1     = 1.5e0*alpha*wt*omk-qm*rs/cc-(dotm/(r*r*kes))*((wt-wb)/sig)*xi
        f2     = wt-wb-(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        dhds   =-0.5e0*hh/sig
        dtdt   =-3.5e0*tauabs/tem
        dtads  = 2.5e0*tauabs/sig
        dtds   = dtads+0.5e0*kes
        dqds   =-qm*(1.5e0*dtds-dtads/(tauabs**2))/qmd
        dqdt   = qm*(4.0e0/tem-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
        dwbds  = wb*(2.0e0*ze+0.5e0)/sig
        df1ds  =-dqds*rs/cc+(dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds
        df1dt  =-dqdt*rs/cc
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
#!        write(6,*) dotm,sig,tem
    if (abs(dsig/sig) < 1.0e-10):
        sig_sadm = np.append(sig_sadm, sig)
        tem_sadm = np.append(tem_sadm, tem)
        wt_sadm = np.append(wt_sadm, wt)
        cnt = cnt + 1
        #write(6,*) sig,dotm,tem,taueff,prad/pgas,wt,dsig/sig,dtem/tem
    else:
        sig = sig_sadm[cnt]
        tem = tem_sadm[cnt]

#!     SADM2
# Initial guess #
dotm = 1.0e-2
wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
#!     tem = (wt/bt1)*1.5*alpha*wt*omk*1.5e0*0.5e0*kes/((rr/xmu)*ai4*4.0e0*aa*rs)
tem = wt*1.5*alpha*wt*omk*1.5e0*0.5e0*kes/((rr/xmu)*ai4*4.0e0*aa*rs)
tem = tem**0.2
#!     sig = (wt/bt1)/((ai4/ai3)*(rr/xmu)*tem)
sig = wt/((ai4/ai3)*(rr/xmu)*tem)
#!     write(6,*) 'SADM',' sig = ',sig,' tem = ',tem
##################
# Define array #
dotm_sadm2 = np.logspace(6, -3, num)
sig_sadm2 = np.full(2, sig)
tem_sadm2 = np.full(2, tem)
wt_sadm2 = np.full(2, wt)
################
#
cnt = 1
#
for j in range(num):
    dotm=dotm_sadm2[j]
    wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
    for i in range(1,20): 
        hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
        wb     = (p0**2*s0**(-2.0*ze)/(8.0e0*pi*hh*rs))*sig**(2.0*ze)
        taues  = 0.5*kes*sig
        tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
        tau    = tauabs+0.5e0*kes*sig
        taueff = sqrt(tau*tauabs)
        qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
        qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
        f1     = 1.5e0*alpha*wt*omk-qm*rs/cc-(dotm/(r*r*kes))*((wt-wb)/sig)*xi
        f2     = wt-wb-(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        dhds   =-0.5e0*hh/sig
        dtdt   =-3.5e0*tauabs/tem
        dtads  = 2.5e0*tauabs/sig
        dtds   = dtads+0.5e0*kes
        dqds   =-qm*(1.5e0*dtds-dtads/(tauabs**2))/qmd
        dqdt   = qm*(4.0e0/tem-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
        dwbds  = wb*(2.0e0*ze+0.5e0)/sig
        df1ds  =-dqds*rs/cc+(dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds
        df1dt  =-dqdt*rs/cc
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
#!        write(6,*) dotm,sig,tem
    if (abs(dsig/sig) < 1.0e-10):
        sig_sadm2 = np.append(sig_sadm2, sig)
        tem_sadm2 = np.append(tem_sadm2, tem)
        wt_sadm2 = np.append(wt_sadm2, wt)
        cnt = cnt+1
        #write(6,*) sig,dotm,tem,taueff,prad/pgas,wt,dsig/sig,dtem/tem
    else:
        sig = sig_sadm2[cnt]
        tem = tem_sadm2[cnt]

def calc_newton():
    dotm_rt = np.logspace(6, -3, num)
    sig_rt = np.full(2, sig)
    tem_rt = np.full(2, tem)
    wt_rt = np.full(2, wt)
    for j in range(num):
	dotm=dotm_sadm2[j]
	wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
	for i in range(1,20): 
	    hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
	    wb     = (p0**2*s0**(-2.0*ze)/(8.0e0*pi*hh*rs))*sig**(2.0*ze)
	    taues  = 0.5*kes*sig
	    tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
	    tau    = tauabs+0.5e0*kes*sig
	    taueff = sqrt(tau*tauabs)
	    qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
	    qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
	    f1     = 1.5e0*alpha*wt*omk-qm*rs/cc-(dotm/(r*r*kes))*((wt-wb)/sig)*xi
	    f2     = wt-wb-(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
	    dhds   =-0.5e0*hh/sig
	    dtdt   =-3.5e0*tauabs/tem
	    dtads  = 2.5e0*tauabs/sig
	    dtds   = dtads+0.5e0*kes
	    dqds   =-qm*(1.5e0*dtds-dtads/(tauabs**2))/qmd
	    dqdt   = qm*(4.0e0/tem-(1.5e0*dtdt-dtdt/(tauabs**2))/qmd)
	    dwbds  = wb*(2.0e0*ze+0.5e0)/sig
	    df1ds  =-dqds*rs/cc+(dotm/(r*r*kes))*((wt-wb)/sig**2)*xi+(dotm/(r*r*kes*sig))*xi*dwbds
	    df1dt  =-dqdt*rs/cc
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
	if (abs(dsig/sig) < 1.0e-10):
	    sig_rt = np.append(sig_rt, sig)
	    tem_rt = np.append(tem_rt, tem)
	    wt_rt = np.append(wt_rt, wt)
	    cnt = cnt+1
	else:
	    sig = sig_rt[cnt]
	    tem = tem_rt[cnt]

    return sig_rt,tem_rt,wt_rt

	

#!    isotemperature 
nut = 7
tem_iso = np.zeros(nut)
dotm_iso = np.zeros([nut, num])
wt_iso = np.zeros([nut, num])
sig_iso = np.zeros([nut, num])
sig_iso[:,:] = np.logspace(-1, 5, num)
for ib in range(0,nut):
    tem  = 10**(5.0+ib)
    tem_iso[ib] = tem
    dotm = 1.0e-4
    wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
    #for iso in range(10,70):
    for iso in range(1,num):
        sig = sig_iso[ib,iso] 
        for i in range(1,20):
            hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
            taues  = 0.5*kes*sig
            tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
            tau    = tauabs+0.5e0*kes*sig
            qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
            qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
            f2     = wt-p0**2*(sig/s0)**(2*ze)/(8.0e0*pi*hh*rs) \
                    -(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
            dhdw   = hh/(2.0e0*wt)
            dtadw  =-tauabs/(2.0e0*wt)
            dtdw   = dtadw
            dqdw   =-qm*(1.5e0*dtdw-dtadw/(tauabs*tauabs))/(1.5e0*tau+sqr3+1.0e0/tauabs)
            dfdw   = 1.0e0+p0*p0*(sig/s0)**(2.0*ze)*dhdw/(8.0e0*pi*hh*hh*rs)  \
             -(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)*(dqdw/qm+dhdw/hh+dtdw/(tau+2.0e0/sqr3))
            dw = -f2/dfdw 
            wt = wt+dw     
#!          write(6,*) dotm,sig,tem,wt
        
        dotm_iso[ib,iso] = dotm
        wt_iso[ib,iso] = wt
       # write(68,*) sig,wt,dw/wt

#!    tau_eff=1
num = 50
dotm = 1.0e-3
wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
sig_taueff = np.logspace(0, 5, num)
wt_taueff = np.full(num, wt)
#for iso in range(20+1,70):
for iso in range(1, num):
    sig = sig_taueff[iso]
    wt  = wt_taueff[iso-1]
    for i in range(1,20):
        hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
        taues  = 0.5e0*kes*sig
        tau    = 0.5e0*kes*sig
        tauabs = 1.0e0/tau
        tem    = (6.2e20*(ai65/(ai3*ai3*ai3))*0.5e0*kes*omk/(24.0e0*aa*rs))**(2.0/7.0) \
                 *sig*wt**(-1.0/7.0)
        qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
        qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
        f2     = wt-p0**2*(sig/s0)**(2*ze)/(8.0e0*pi*hh*rs) \
                 -(ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        dhdw   = hh/(2.0e0*wt)
        dtedw  =-tem/(7.0e0*wt) 
        dqdw   = qm*4.0e0*dtedw/tem
        df2dw  = 1.0e0+p0*p0*(sig/s0)**(2.0*ze)*dhdw/(8.0e0*pi*hh*hh*rs)  \
          -(ai4/ai3)*(rr/xmu)*sig*dtedw \
          -(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)*(dqdw/qm+dhdw/hh)
        dw = -f2/df2dw
        wt = wt+dw
    wt_taueff[iso] = wt
#:!          write(6,*) dotm,sig,tem,wt
    #write(69,*) sig,wt,dw/wt

#!    Pgas=Prad1
dotm = 1
wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
wt = 1.0e22
sig_pgpr = np.logspace(6, 2, num)
wt_pgpr = np.full(num, wt)
for iso in range(0+1,40):
    sig = 10**(6.0-iso*0.1)
    sig = sig_pgpr[iso]
    wt = wt_pgpr[iso-1]
    for i in range(1,20):
        hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
        taues  = 0.5e0*kes*sig
        tem    = (wt-p0*p0*((sig/s0)**(2.0*ze))/(8.0e0*pi*hh*rs))/(2.0e0*(ai4/ai3)*(rr/xmu)*sig)
        tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
        tau    = tauabs+0.5e0*kes*sig
        qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
        qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
        f      = (ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        dhdw   = hh/(2.0e0*wt)
        dtedw  = (1.0e0+p0*p0*((sig/s0)**(2.0*ze)*dhdw)/(8.0e0*pi*hh*hh*rs)) \
                /(2.0e0*(ai4/ai3)*(rr/xmu)*sig)
        dtadw  = tauabs*(-dhdw/hh-3.5e0*dtedw/tem)
        dqdw   = qm*(4.0e0*dtedw/tem-(1.5e0*dtadw-dtadw/(tauabs**2))/qmd)
        dfdw   = (ai4/ai3)*(rr/xmu)*sig*dtedw-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3) \
                 *(dqdw/qm+dhdw/hh+dtadw/(tau+2.0e0/sqr3))
        dw     =-f/dfdw
        wt     = wt+dw

    wt_pgpr[iso] = wt
#!        write(6,*) sig,wt,tem
   # write(70,*) sig,wt,dw/wt

#!    Pgas=Prad2
dotm = 1.0e-2
wt = dotm*(ell-ellin)*(cc*cc/kes)/(r*r*alpha)
wt = 1.0e26
sig_pgpr2 = np.logspace(3, 1, num)
wt_pgpr2 = np.full(num, wt)
#for j in range(0+1,40):
for j in range(1,num):
    sig = sig_pgpr2[j]
    wt = wt_pgpr2[j-1]
    for i in range(1,20):
        hh     = 3.0e0*(sqrt(wt/sig)/cc)/omk
        taues  = 0.5e0*kes*sig
        tem    = (wt-p0*p0*((sig/s0)**(2.0*ze))/(8.0e0*pi*hh*rs))/(2.0e0*(ai4/ai3)*(rr/xmu)*sig)
        tauabs = (6.2e20/(8.0e0*aa*cc*rs))*(ai65/ai3**3)*(sig*sig/hh)*tem**(-3.5)
        tau    = tauabs+0.5e0*kes*sig
        qmd    = 1.5e0*tau+sqr3+1.0e0/tauabs
        qm     = 4.0e0*aa*cc*ai3*tem**4/qmd
        f      = (ai4/ai3)*(rr/xmu)*sig*tem-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3)
        dhdw   = hh/(2.0e0*wt)
        dtedw  = (1.0e0+p0*p0*((sig/s0)**(2.0*ze)*dhdw)/(8.0e0*pi*hh*hh*rs)) \
                /(2.0e0*(ai4/ai3)*(rr/xmu)*sig)
        dtadw  = tauabs*(-dhdw/hh-3.5e0*dtedw/tem)
        dqdw   = qm*(4.0e0*dtedw/tem-(1.5e0*dtadw-dtadw/(tauabs**2))/qmd)
        dfdw   = (ai4/ai3)*(rr/xmu)*sig*dtedw-(qm/(4.0e0*cc))*(ai4/ai3)*hh*rs*(tau+2.0e0/sqr3) \
                 *(dqdw/qm+dhdw/hh+dtadw/(tau+2.0e0/sqr3))
        dw     =-f/dfdw
        wt     = wt+dw

    wt_pgpr2[j] = wt
#!        write(6,*) sig,wt,tem
#      write(70,*) sig,wt,dw/wt

#!    Pmag=Pgas+Prad
sig_pmpgpr = np.logspace(-2, 5, num)
wt_pmpgpr = (p0*p0*cc*omk/(12.0e0*pi*rs))**(2.0/3.0)*((sig_pmpgpr/s0)**(4.0*ze/3.0))*sig_pmpgpr**(1.0/3.0)
#for i in range(0,70):
#    sig = 10**(-2.0+0.1*i)
#    wt  = (p0*p0*cc*omk/(12.0e0*pi*rs))**(2.0/3.0)*((sig/s0)**(4.0*ze/3.0))*sig**(1.0/3.0)
    #write(71,*) sig,wt

