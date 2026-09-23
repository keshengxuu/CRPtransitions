import matplotlib
# matplotlib.use('Agg')
import os
import numpy as np
np.random.seed(11)
import matplotlib.pyplot as plt
from scipy import signal
# import Wavelets
from NetworkarchitectureB import Networkmodel
# from mpi4py import MPI
import csv

rank = 0
threads = 1

def spikeDetection(t, V, spikeThreshold):
    tSpikes = []
    v = np.asarray(V)
    nSteps = len(V)

    for i in range(1, nSteps):
        if (V[i - 1] <= spikeThreshold) & (V[i] > spikeThreshold):
            ts = ((i - 1) * dt * (V[i - 1] - spikeThreshold) +
                  i * dt * (spikeThreshold - V[i])) / (V[i - 1] - V[i])
            tSpikes.append(ts)
    return tSpikes
def spikeToFile(t_spikes, fileName):
    with open(fileName, "w") as f:
        for neuron_spikes in t_spikes:
            line = ",".join(f"{ts:.3f}" for ts in neuron_spikes)
            f.write(line + "\n")

def expnorm(tau1, tau2):
    if tau1 > tau2:
        t2 = tau2;
        t1 = tau1
    else:
        t2 = tau1;
        t1 = tau2
    tpeak = t1 * t2 / (t1 - t2) * np.log(t1 / t2)
    return (np.exp(-tpeak / t1) - np.exp(-tpeak / t2)) / (1 / t2 - 1 / t1)


def ISI_Phase(Num_Neurs, TIME, spikes):
    Phase = np.zeros((TIME.size, Num_Neurs), dtype=np.float32)
    for ci in range(Num_Neurs):  # Num_Neurs is total number of neurons
        mth = 0  # the flag of mth spikes of neuron i at time t
        nt = 0
        for t in TIME:
            # if np.array(spikes[ci]).size ==0:
            if np.sum(spikes[ci]) < 2:
                Phase[nt, ci] = 0
            elif (mth + 1) < np.array(spikes[ci]).size:
                Phase[nt, ci] = 2.0 * np.pi * (t - spikes[ci][mth]) / (spikes[ci][mth + 1] - spikes[ci][mth])
                nt = nt + 1
                if t > spikes[ci][mth + 1]:
                    mth = mth + 1
    return Phase


def HR_network(X, i):
    global firing
    x, y, z,cl_in, sex, sey, six, siy, sexe, seye = X  # agregué variable 's'
    E_cl = -0.62 * np.log(3.01 / cl_in)
    Icl = siy * (x - E_cl)
    ISyn = (sey + seye) * (x - VsynE) + siy * (x - E_cl)
    firingExt = np.random.binomial(1, iRate * dt, size=N)
    if any(i > delay_dt):
        firing = (V_t[i - delay_dt, range(N)] > theta) * (V_t[i - delay_dt - 1, range(N)] < theta)

    return np.array([y - a0 * x ** 3 + b0 * x ** 2 + i_ext0 - z - ISyn,
                     c0 - d0 * x ** 2 - y,
                     r0 * (s0 * (x - k0) - z),
                     W * scaling * Icl/ (V_d * F_r) + (cl_in_eq_scale - cl_in) / tau_Cl,
                     -sex * (1 / tau1E + 1 / tau2E) - sey / (tau1E * tau2E) + np.dot(CMeMatrix, firing[0:Ne]) + np.dot(
                         CMieMatrix, firing[0:Ne]),
                     sex,
                     -six * (1 / tau1I + 1 / tau2I) - siy / (tau1I * tau2I) + np.dot(CMiMatrix, firing[Ne:]) + np.dot(
                         CMeiMatrix, firing[Ne:]),
                     six,
                     -sexe * (1 / tau1E + 1 / tau2E) - seye / (tau1E * tau2E) + firingExt * GsynExt,
                     sexe])

scaling = 2000
W = 1
tau_Cl = 500   #5s

equil = 400  # 400
Trun =3600 # 2000
Total = Trun + equil  # ms

V_d = 0.24
F_r = 9.649 * 10000

cl_in_eq = 6
scale = 10
cl_in_eq_scale =  cl_in_eq /scale
I_scale = 1

dt = 0.01  # ms

a0 = 1.0
b0 = 3.0
c0 = 1.0
d0 = 5.0
s0 = 4.0
r0 = 0.006
k0 = -1.56
i_ext0 = 1.5 # chaos3.0

# Synaptic parameters
mGsynE = 1.0;
mGsynI = 40;
mGsynExt = 0.6  # mean
sGsynE = 0.2;
sGsynI = 2;
sGsynExt = 0.2
VsynE = 0;
# Ecl = -1.0  # reversal potential
tau1E = 3;
tau2E = 1
tau1I = 4;
tau2I = 1
Pe = 0.3;
Pi = 0.3
iRate = 6  # 10*Pi
P_SW = 0.01

factE = 1000 * dt * expnorm(tau1E, tau2E)
factI = 1000 * dt * expnorm(tau1I, tau2I)
W_gap = 0  # 0.001
W_gapi = 0.001  # 0.05

mdelay = 1.5;
sdelay = 0.1  # ms

theta = 0

Ne = 1000  # Numero de neuronas excitatorias
Ni = 250  # Numero de neuronas inhibitorias
N = Ne + Ni
# CM=np.zeros((N,N))
GsynE = np.random.normal(mGsynE, sGsynE, size=(N, Ne)) / factE
GsynExt = np.random.normal(mGsynExt, sGsynExt, size=N)
GsynExt = GsynExt * (GsynExt > 0) / factE
GsynI = np.random.normal(mGsynI, sGsynI, size=(N, Ni)) / factI
delay = np.random.normal(mdelay, sdelay, size=N)

firing = np.zeros(N)

delay_dt = (delay / dt).astype(int)
equil_dt = int(equil / dt)
Time = np.arange(0, Total, dt)
nsteps = len(Time)
Time2 = Time[equil_dt:]

jee = 3.0
jei = 0.4
jie = 0.02
jii = 0.04

isim = 0


idx1 = 49
dtype = np.float32
V_t = np.zeros((nsteps, N), dtype=np.float32)
Y_t = np.zeros((nsteps, N), dtype=np.float32)
Z_t = np.zeros((nsteps, N), dtype=np.float32)
ISyn_t = np.zeros((nsteps, N), dtype=np.float32)
ISyn_It = np.zeros((nsteps, N), dtype=np.float32)
x_Ecl = np.zeros((nsteps, N), dtype=np.float32)
Ecl_t = np.zeros((nsteps, N), dtype=np.float32)
Cl_t = np.zeros((nsteps, N), dtype=np.float32)
siy_t = np.zeros((nsteps, N), dtype=np.float32)
# g_cl = np.zeros((nsteps, N), dtype=np.float32)
# Icl = np.zeros((nsteps, N), dtype=np.float32)
if isim % threads == rank:

    net = Networkmodel(Ne1=Ne,  # The numbers of excitatroy ensemble
                       Ni1=Ni,  # The numbers of inhibitory ensemble
                       cp=P_SW,  # the probabilty to add new links
                       W_gap=0,  # the weight of gap junction for excitatoty population.
                       W_Chem1=jee,  # the weight of  itself chemical synapse in excitatoty population.
                       W_ei1=jei,  # the weight of inhibitory synapse currents from inhibitory to excitatory population.
                       W_ii1=jii,  # the weight of  itself inhibitory currents in inhibitory population
                       W_ie1=jie,
                       # the weight of excitatory currents in inhibitory population from excitatory population
                       )

    wEE_gap, wEE_Chem, wEI_chem, wII_chem, wIE_chem = net.SmallWorld(deg=10, opt=2)

    # the self-excitatory connections matrix  of the excitatory populations
    CMeMatrix = np.concatenate((wEE_Chem, np.zeros((Ni, Ne))), axis=0) * GsynE
    # the inhiboptry connections from inhibitory populations to excitatory populatuions
    CMeiMatrix = np.concatenate((wEI_chem, np.zeros((Ni, Ni))), axis=0)
    # the itself-inhibitory connections matrix of the inhibitory populations
    CMiMatrix = np.concatenate((np.zeros((Ne, Ni)), wII_chem), axis=0) * GsynI
    # the excitatory connection matrix from excitatory populations to inhibitory populations
    CMieMatrix = np.concatenate((np.zeros((Ne, Ne)), wIE_chem), axis=0)
    # the gap junction matrix
    CM0 = np.concatenate((wEE_gap, np.zeros((Ni, Ne))), axis=0)
    CMgapMat = np.concatenate((CM0, np.zeros((N, Ni))), axis=1)

    x = np.zeros((nsteps, N))
    ISyn_t = np.zeros((nsteps, N))
    V_t = np.zeros((nsteps, N))
    Y_t = np.zeros((nsteps, N))
    Z_t = np.zeros((nsteps, N))

    x_init = np.random.uniform(-2, 1, size=N)  # -70.0 * np.ones(N) # -70 is the one used in brian simulation
    y = np.random.uniform(-2, 1, size=N)
    z = np.random.uniform(-3, -2, size=N)
    sex = np.zeros_like(x_init)
    sey = np.zeros_like(x_init)
    six = np.zeros_like(x_init)
    siy = np.zeros_like(x_init)
    sexe = np.zeros_like(x_init)
    seye = np.zeros_like(x_init)

    cl0 = 0.6 * np.ones_like(x_init)  # mM

    X = (x_init, y, z,cl0, sex, sey, six, siy, sexe, seye)

    for i in range(nsteps):

        X += dt * HR_network(X, i)
        V_t[i] = X[0]
        sey = X[5]
        seye = X[-1]
        siy = X[-3]
        Y_t[i] = X[1]  # Y对应恢复变量y
        Z_t[i] = X[2]  # Z对应慢变电流z
        # x,y,z,sex,sey,six,siy,sexe,seye=X
        cl_now = np.maximum(X[3], 1e-6)
        Cl_t[i, :] = cl_now
        E_cl_now = -0.62 * np.log(3.01 / cl_now)
        Ecl_t[i, :] = E_cl_now

        ISyn_t[i, :] = (X[5] + X[9]) * (X[0] - VsynE) + X[7] * (X[0] - E_cl_now)
        ISyn_It[i, :] =X[7] * (X[0] - E_cl_now)
        x_Ecl[i, :]= (X[0] - E_cl_now)
        siy_t[i, :] = X[7]
        # g_cl[i, :] = X[7]
        # Icl[i, :] = X[0] - E_cl_now
    V_w = V_t[:, :]
    ISyn_w = ISyn_t[:, :]
    Ecl_w = Ecl_t[:, :]
    Cl_w = Cl_t[:, :]
    Icl_w = ISyn_It[:, :]
    siy_w = siy_t[:, :]
    x_Ecl_w = x_Ecl[:, :]
    # mean_siy_E = np.mean(siy_w[:, 0:Ne], axis=1)
    # mean_Icl_E = np.mean(Icl_w[:, 0:Ne], axis=1)
    # mean_xEcl_E = np.mean(x_Ecl_w[:, 0:Ne], axis=1)
    # mean_Ecl_E = np.mean(Ecl_w[:, 0:Ne], axis=1)


spikes = []
for i in range(N):
    ts_e = spikeDetection(Time, V_w[:,i], 0)
    spikes.append(ts_e)

np.savetxt(f"W{W}V.txt", V_w[:, idx1], fmt="%.6f")
np.savetxt(f"W{W}Ecl.txt", Ecl_w[:, idx1], fmt="%.6f")
np.savetxt(f"W{W}cl.txt", Cl_w[:, idx1], fmt="%.6f")
np.savetxt(f"W{W}Icl.txt", Icl_w[:, idx1], fmt="%.6f")
np.savetxt(f"W{W}Isyn.txt", ISyn_w[:, idx1], fmt="%.6f")
np.savetxt(f"W{W}siy.txt", siy_w[:, idx1], fmt="%.6f")
np.savetxt(f"W{W}x_Ecl.txt", x_Ecl_w[:, idx1], fmt="%.6f")

# np.savetxt(f"W{W}mean_siy_E.txt", mean_siy_E[:], fmt="%.6f")
# np.savetxt(f"W{W}mean_Icl_E.txt", mean_Icl_E[:], fmt="%.6f")
# np.savetxt(f"W{W}mean_x_Ecl_E.txt", mean_xEcl_E[:], fmt="%.6f")
# np.savetxt(f"W{W}mean_Ecl_E.txt", mean_Ecl_E[:], fmt="%.6f")



# np.savetxt(f"scaling{scaling}Ecl.txt", Ecl_w[:, idx1], fmt="%.6f")
spikeToFile(spikes, f"W{W}spikes.txt")

plt.figure(figsize=(10, 6))
plt.plot(Time, V_w[:, idx1], label=f"x neuron {idx1+1}")
plt.legend()
plt.title("HR-V")
plt.savefig(f"W{W}V.png", dpi=300)