import pandas as pd
import cross_flow as cf
import numpy as np


def cross_flow():

    parameters = list(pd.read_csv('input_coker.txt', header=None)[0])
    columns = ('CO2', 'N2', 'O2', 'Ar', 'SO2', 'H2O')

    A = parameters[0]
    dA = 500
    N = int(A / dA)
    y0 = [parameters[i] for i in range(14, 20)]
    x0 = [parameters[i] for i in range(5, 11)]
    M = len(x0)
    mi = [0] * M
    pV = parameters[11]
    pM = [0] * (N+1)
    pM[0] = parameters[2]

    y = pd.DataFrame(index=range(N,-1,-1), columns=columns)
    x = pd.DataFrame(index=range(N+1), columns=columns)
    Q = pd.DataFrame(index=range(N), columns=columns)
    m_track = pd.DataFrame(index=range(N), columns=columns)

    y.iloc[0, :] = y0
    x.iloc[0, :] = x0
    L = [0] * (N+1)
    L[0] = parameters[4]
    V = [0] * (N+1)
    V[0] = parameters[13]

    for k in range(N):
        Q = cf.permeabilityCalc(N, k, pM, x, y, pV, M, Q, parameters)

        mi = cf.permeationCalc(mi, M, Q, dA, pM, pV, y, x, k)
        m_track.iloc[k, :] = mi[:]
        L, V, x, y = cf.calculatedVals(mi, L, V, x, y, k, M)
        delt_P = 0.0689476 / N
        pM[k+1] = pM[k] - delt_P

    return V, L, y, x, pM, pV, M, N, Q, dA

#! ----------------------------------------------------------------------------------------

def permeabilityCalc(N, k, pM, x, y, pV, M, Q, parameters):
    Q_gpu = [0] * M
    pwsatP = (0.61078*np.exp((17.27*(parameters[12]-273.15) / (parameters[12] - 273.15 + 237.3)))) / 100
    pwsatF = (0.61078*np.exp((17.27*(parameters[3]-273.15) / (parameters[3] - 273.15 + 237.3)))) / 100
    CONV = 0.1205/3600
    s = 0.1

    aw = ((pM[k] * x.iloc[k, -1]) * 0.6) / pwsatF + ((pV * y.iloc[k, -1]) * 0.4) / pwsatP
    aw = 1
    if aw > 1:
        aw = 1

    aw = aw*100

    Q_gpu[0] = 0.0047 * np.exp(0.0933*aw) / s
    Q_gpu[1] = 0.0014 * np.exp(.0637*aw) / s
    Q_gpu[2] = 10#10
    Q_gpu[3] = 10#10
    Q_gpu[4] = 1000
    Q_gpu[5] = 1000

    for i in range(0, 6):
        Q.iloc[k, i] = Q_gpu[i] * CONV
    return Q

#! ----------------------------------------------------------------------------


def permeationCalc(mi, M, Q, dA, pM, pV, y, x, k):
    for i in range(M):
        mi[i] = Q.iloc[k, i] * dA * (pM[k] * x.iloc[k, i] - y.iloc[k, i] * pV)

    return mi

#! ----------------------------------------------------------------------------


def calculatedVals(mi, L, V, x, y, k, M):
    Total = sum(mi)
    L[k+1] = L[k] - Total
    V[k+1] = V[k] + Total

    for i in range(M):
        x.iloc[k+1, i] = (L[k] * x.iloc[k, i] - mi[i]) / L[k+1]
        y.iloc[k+1, i] = (V[k] * y.iloc[k, i] + mi[i]) / V[k+1]

    for i in range(M):
        sumX = sum(x.iloc[k+1, :])
        sumY = sum(y.iloc[k+1, :])

        x.iloc[k+1, i] = x.iloc[k+1, i] / sumX
        y.iloc[k+1, i] = y.iloc[k+1, i] / sumY

        if x.iloc[k+1, i] < 0:
            x.iloc[k+1, i] = 0
        elif x.iloc[k+1, i] > 1:
            x.iloc[k+1, i] = 1
        else:
            pass
        if y.iloc[k+1, i] < 0:
            y.iloc[k+1, i] = 0
        elif y.iloc[k+1, i] > 1:
            y.iloc[k+1, i] = 1
        else:
            pass

    return L, V, x, y

