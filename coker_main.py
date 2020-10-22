"""
Variable definition:
* If USER means inital/inlet value is users input. *

V - list - Flowrate of sweep/permeate stream at each stage [mol/s]      *USER* 
y -  dataframe - composition of sweep/permeate at each satge            *USER* 
pV - float - pressure of sweep/permeate assumed constnat                *USER*

L - list - Flowrate of retenate/residue stream at each stage  [mol/s]   *USER* 
x - list - composition of retenate/residue at each stage (inlet)        *USER* 
pM - list - pressure of sweep/permeate at each stage [bar]              *USER* 

M - int - # of components in this file 2. 
N - int - # of stages
dA - int - area increment 
Q - dataframe - Permeance of components [mol / s m^2 bar]

----------------------------------------------------------------------------------------------------

Program runs a cross-flow module, inital values are given in a text file stored in the same directory.
This provides an inital estimate for the calculation of coefficients used in counter-flow calculations.
After counter-flow iteration coefficients are recalculated. This repeats until change in final flowrates
is within tolerance. 

----------------------------------------------------------------------------------------------------

Issue at the moment is that model struggles to converge. For values that it does converge for flowrates are
not close to intial estimates and often don't make physical sense. At the moment 2 component system is not converging
however multi-comp is. Issue with multicomp is that flow rates are 2 order of magnitude less than they should be,
also issue such as having too much of certain components depending on composition. 

#? Comments marked so are possible implementations. 
#! Comments marked so are explanations. 
"""
import pandas as pd
import numpy as np
import cross_flow as cf

columns = ('CO2', 'N2', 'O2', 'Ar', 'SO2', 'H2O')


V, L, y, x, pM, pV, M, N, Q, dA = cf.cross_flow()

#! In Coker et al paper membrane area is 226m^2, area being used for current conditions is N * dA

#? area_ratio = N * dA / 226
#? no_fibres = area_ratio * 3E5
#? L = [L[i] / no_fibres for i in range(N+1)]
#? V = [V[i] / area_ratio for i in range(N+1)]

L[:] = L[::-1]
pM[:] = pM[::-1]

#! L/pM[0] is the retenate exit L/pM[-1] is the entrance. 
#! V[0] is the sweep entrance V[-1] is the exit.

B = pd.DataFrame(index=range(N), columns=columns)
C = pd.DataFrame(index=range(N), columns=columns)
D = pd.DataFrame(index=range(N), columns=columns)
sol = pd.DataFrame(0, index=range(N), columns=columns)
vals = pd.DataFrame(index=range(N), columns=columns)
A = pd.DataFrame(0, index=range(N), columns=range(N))

test1 = 1
test2 = 1
tol = .00001

while test1 > tol or test2 > tol:
    Ltest = L[0]
    Vtest = V[-1]
    for k in range(N):
        for i in range(M):
            if k == 0:
                B.iloc[k, i] = (- V[k] / (pV * dA * Q.iloc[k, i])) #?  * (1 + (Q.iloc[k, i] * dA * pM[k] / L[k]))
                C.iloc[k, i] = 1 + (V[k] / (pV * dA * Q.iloc[k, i])) + (V[k+1] / (pV * dA * Q.iloc[k+1, i])) * (1 + (Q.iloc[k+1, i] * dA * pM[k]) / L[k])
                D.iloc[k, i] = -V[k+1] / (pV * dA * Q.iloc[k, i]) - 1
                sol.iloc[k, i] = - B.iloc[k, i]

            elif N-1 > k > 0:
                B.iloc[k, i] = (- V[k] / (pV * dA * Q.iloc[k, i])) * (1 + (Q.iloc[k, i] * dA * pM[k-1] / L[k-1]))
                C.iloc[k, i] = 1 + (V[k] / (pV * dA * Q.iloc[k-1, i])) + (V[k+1] / (pV * dA * Q.iloc[k+1, i])) * (1 + (Q.iloc[k+1, i] * dA * pM[k]) / L[k])
                D.iloc[k, i] = -V[k+1] / (pV * dA * Q.iloc[k, i]) - 1

            elif k == N-1:
                B.iloc[k, i] = (- V[k] / (pV * dA * Q.iloc[k, i])) * (1 + (Q.iloc[k, i] * dA * pM[k-1] / L[k-1]))
                C.iloc[k, i] = 1 + (V[k] / (pV * dA * Q.iloc[k-1, i])) + (V[k+1] / (pV * dA * Q.iloc[k, i])) * (1 + (Q.iloc[k, i] * dA * pM[k]) / L[k])
                D.iloc[k, i] = -V[k+1] / (pV * dA * Q.iloc[k, i]) - 1
                sol.iloc[k, i] = - D.iloc[k, i]

    for i in range(M):
        for k in range(N):
            if k == 0:
                A.iloc[k, k:2] = [C.iloc[k, i], D.iloc[k, i]]
            elif 0 < k < N-1:
                A.iloc[k, k-1:k+2] = [B.iloc[k, i], C.iloc[k, i], D.iloc[k, i]]
            else:
                A.iloc[k, k-1:k+1] = [B.iloc[k, i], C.iloc[k, i]]

        vals.iloc[:, i] = np.linalg.solve(A, sol.iloc[:, i])

    for k in range(N):
        L[k] = sum(vals.iloc[k, :])

    for k in range(N):
        V[k+1] = V[k] + L[k+1] - L[k]

    test1 = abs((Ltest - L[0]) / L[0])
    test2 = abs((Vtest - V[-1]) / V[-1])
    print(test1,test2)

print('Done', L[0], V[-1])
