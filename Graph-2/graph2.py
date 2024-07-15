# Plotting graphs -
# a) Number of Organizations (Heterogeneous and Homogeneous) VS Number of Training Rounds w.r.t. Old Model
# a) Number of Organizations (Heterogeneous and Homogeneous) VS Number of Training Rounds w.r.t. New Model

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

# Read parameters from Excel file
df = pd.read_excel('parameters2.xlsx')

# Read scalar parameters
K = df['Value'][df['Parameter'] == 'K'].values[0]
T = df['Value'][df['Parameter'] == 'T'].values[0]
TUL = df['Value'][df['Parameter'] == 'TUL'].values[0]
TDL = df['Value'][df['Parameter'] == 'TDL'].values[0]
e0 = df['Value'][df['Parameter'] == 'e0'].values[0]
e1 = df['Value'][df['Parameter'] == 'e1'].values[0]
uploadCost = df['Value'][df['Parameter'] == 'uploadCost'].values[0]
downloadCost = df['Value'][df['Parameter'] == 'downloadCost'].values[0]
investmentCost = df['Value'][df['Parameter'] == 'investmentCost'].values[0]
minR = df['Value'][df['Parameter'] == 'minR'].values[0]
maxR = df['Value'][df['Parameter'] == 'maxR'].values[0]
penalty = df['Value'][df['Parameter'] == 'penalty'].values[0]
stepsize = df['Value'][df['Parameter'] == 'stepsize'].values[0]
threshold = df['Value'][df['Parameter'] == 'threshold'].values[0]
lamda = df['Value'][df['Parameter'] == 'lamda'].values[0]

# Read array parameters
S_arrays = ast.literal_eval(df['Value'][df['Parameter'] == 'S_arrays'].values[0])
D_arrays = ast.literal_eval(df['Value'][df['Parameter'] == 'D_arrays'].values[0])
num_orgs = ast.literal_eval(df['Value'][df['Parameter'] == 'num_orgs'].values[0])
homo_unit_revenues = ast.literal_eval(df['Value'][df['Parameter'] == 'homo_unit_revenues'].values[0])
hetero_unit_revenues = ast.literal_eval(df['Value'][df['Parameter'] == 'hetero_unit_revenues'].values[0])
R_vectors = ast.literal_eval(df['Value'][df['Parameter'] == 'R_vectors'].values[0])
Pi_vectors = ast.literal_eval(df['Value'][df['Parameter'] == 'Pi_vectors'].values[0])

# Convert lists to numpy arrays where necessary
S_arrays = [np.array(arr, dtype=float) for arr in S_arrays]
D_arrays = [np.array(arr, dtype=float) for arr in D_arrays]
homo_unit_revenues = [np.array(arr, dtype=float) for arr in homo_unit_revenues]
hetero_unit_revenues = [np.array(arr, dtype=float) for arr in hetero_unit_revenues]
R_vectors = [np.array(arr, dtype=float) for arr in R_vectors]
Pi_vectors = [np.array(arr, dtype=float) for arr in Pi_vectors]

# Create convergeFlags_arrays
convergeFlags_arrays = [np.array([False] * len(arr), dtype=bool) for arr in S_arrays]

def run_old_model(UnitRevenue, RVector, PiVector, S, D, convergeFlags):
    num_organizations = len(RVector)
    rounds = []
    payoff_history = [[] for _ in range(num_organizations)]
    rF_history = [[] for _ in range(num_organizations)]
    pi_history = [[] for _ in range(num_organizations)]

    def getUtility(rF, organisation):
        er0 = e0 / e1
        erF = e0 / (e1 + (K * rF))
        return UnitRevenue[organisation] * (er0 - erF)

    def getCost(rF, organisation):
        return getCommunicationCost(rF) + getInvestmentCost(rF, organisation) + getOperatingCost(rF, organisation)

    def getCommunicationCost(rF):
        return (uploadCost + downloadCost) * rF

    def getInvestmentCost(rF, organisation):
        return investmentCost * getFNought(rF, organisation)

    def getOperatingCost(rF, organisation):
        return 0.00004833 * getMaxTime(rF, organisation) * getFNought(rF, organisation)**2 * S[organisation] * D[organisation] * K * rF

    def getR(organisation):
        maxPayOff = float('-inf')
        maxPayAt = 0.0
        for rF in np.arange(minR, maxR + 1):
            payOff = getUtility(rF, organisation) - getCost(rF, organisation) - penalty * getPenalityTerm() + PiVector[organisation]
            if payOff > maxPayOff:
                maxPayOff = payOff
                maxPayAt = rF
        return maxPayAt

    def getNewR(rComputed, rOld):
        return rOld + penalty * (rComputed - rOld)

    def getNewPi(oldPi, n):
        i = n - 2
        j = n - 1
        if n == 0:
            i = len(RVector) - 2
            j = len(RVector) - 1
        elif n == 1:
            i = len(RVector) - 1
        return oldPi + penalty * stepsize * (RVector[i] - RVector[j])

    def getRAverage(rF, organisation):
        RvectorTemp = np.copy(RVector)
        RvectorTemp[organisation] = rF
        return np.mean(RvectorTemp)

    def getFNought(rF, organisation):
        return S[organisation] * D[organisation] * K / ((T / getRAverage(rF, organisation)) - TUL - TDL)

    def getMaxTime(rF, organisation):
        return ((S[organisation] * D[organisation] * K) / getFNought(rF, organisation)) + TUL + TDL

    def getPenalityTerm():
        sum_penality = 0.0
        for i in range(len(RVector)):
            j = (i - 2) % len(RVector)
            k = (i - 1) % len(RVector)
            temp = RVector[j] - RVector[k]
            sum_penality += temp * temp
        return sum_penality

    j = 0
    while True:
        globalConv = True
        rounds.append(j)
        for i in range(len(RVector)):
            r = getR(i)
            newR = getNewR(r, RVector[i])
            newPi = getNewPi(PiVector[i], i)
            if abs(newR - r) > threshold:
                RVector[i] = newR
                PiVector[i] = newPi
            else:
                convergeFlags[i] = True

            payoff = getUtility(RVector[i], i) - getCost(RVector[i], i) - penalty * getPenalityTerm() + PiVector[i]
            payoff_history[i].append(payoff)
            rF_history[i].append(RVector[i])
            pi_history[i].append(PiVector[i])

        j += 1
        for flag in convergeFlags:
            if not flag:
                globalConv = False
                break
        if globalConv:
            break

    return rounds, payoff_history, rF_history, pi_history

def run_new_model(UnitRevenue, RVector, PiVector, S, D, convergeFlags):
    num_organizations = len(RVector)
    rounds = []
    payoff_history = [[] for _ in range(num_organizations)]
    rF_history = [[] for _ in range(num_organizations)]
    pi_history = [[] for _ in range(num_organizations)]

    def getUtility(rF, organisation):
        er0 = e0 / e1
        erF = e0 / (e1 + (K * rF))
        return UnitRevenue[organisation] * (er0 - erF)

    def getCost(rF, organisation):
        return getCommunicationCost(rF) + getInvestmentCost(rF, organisation) + getOperatingCost(rF, organisation)

    def getCommunicationCost(rF):
        return (uploadCost + downloadCost) * rF

    def getInvestmentCost(rF, organisation):
        return investmentCost * getFNought(rF, organisation)

    def getOperatingCost(rF, organisation):
        return 0.00004833 * getMaxTime(rF, organisation) * getFNought(rF, organisation)**2 * S[organisation] * D[organisation] * K * rF

    def getR(organisation):
        maxPayOff = float('-inf')
        maxPayAt = 0.0
        for rF in np.arange(minR, maxR + 1):
            payOff = getUtility(rF, organisation) - getCost(rF, organisation) - penalty * getPenalityTerm() + PiVector[organisation]
            if payOff > maxPayOff:
                maxPayOff = payOff
                maxPayAt = rF
        return maxPayAt

    def getNewR(rComputed, rOld):
        return rOld + penalty * (rComputed - rOld)

    def getNewModelPi(r, n):
        pi = 0.0
        for i in range(len(RVector)):
            if i == n:
                continue
            pi += (lamda * getInvestmentCost(r, n)) - (lamda * getInvestmentCost(RVector[i], i))
        return pi

    def getRAverage(rF, organisation):
        RvectorTemp = np.copy(RVector)
        RvectorTemp[organisation] = rF
        return np.mean(RvectorTemp)

    def getFNought(rF, organisation):
        return S[organisation] * D[organisation] * K / ((T / getRAverage(rF, organisation)) - TUL - TDL)

    def getMaxTime(rF, organisation):
        return ((S[organisation] * D[organisation] * K) / getFNought(rF, organisation)) + TUL + TDL

    def getPenalityTerm():
        sum_penality = 0.0
        for i in range(len(RVector)):
            j = (i - 2) % len(RVector)
            k = (i - 1) % len(RVector)
            temp = RVector[j] - RVector[k]
            sum_penality += temp * temp
        return sum_penality

    j = 0
    while True:
        globalConv = True
        rounds.append(j)
        for i in range(len(RVector)):
            r = getR(i)
            newR = getNewR(r, RVector[i])
            newPi = getNewModelPi(r, i)
            if abs(newR - r) > threshold:
                RVector[i] = newR
                PiVector[i] = newPi
            else:
                convergeFlags[i] = True

            payoff = getUtility(RVector[i], i) - getCost(RVector[i], i) - penalty * getPenalityTerm() + PiVector[i]
            payoff_history[i].append(payoff)
            rF_history[i].append(RVector[i])
            pi_history[i].append(PiVector[i])

        j += 1
        for flag in convergeFlags:
            if not flag:
                globalConv = False
                break
        if globalConv:
            break

    return rounds, payoff_history, rF_history, pi_history

# Run models and collect results
old_homo_rounds = []
old_hetero_rounds = []
new_homo_rounds = []
new_hetero_rounds = []

for i in range(len(num_orgs)):
    old_homo, _, _, _ = run_old_model(homo_unit_revenues[i], R_vectors[i], Pi_vectors[i], S_arrays[i], D_arrays[i], convergeFlags_arrays[i].copy())
    old_hetero, _, _, _ = run_old_model(hetero_unit_revenues[i], R_vectors[i], Pi_vectors[i], S_arrays[i], D_arrays[i], convergeFlags_arrays[i].copy())
    new_homo, _, _, _ = run_new_model(homo_unit_revenues[i], R_vectors[i], Pi_vectors[i], S_arrays[i], D_arrays[i], convergeFlags_arrays[i].copy())
    new_hetero, _, _, _ = run_new_model(hetero_unit_revenues[i], R_vectors[i], Pi_vectors[i], S_arrays[i], D_arrays[i], convergeFlags_arrays[i].copy())

    old_homo_rounds.append(len(old_homo))
    old_hetero_rounds.append(len(old_hetero))
    new_homo_rounds.append(len(new_homo))
    new_hetero_rounds.append(len(new_hetero))

# Plot histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Old model histogram
x = np.arange(len(num_orgs))
width = 0.35

ax1.bar(x - width/2, old_homo_rounds, width, label='Homogeneous', color='blue')
ax1.bar(x + width/2, old_hetero_rounds, width, label='Heterogeneous', color='red')
ax1.set_xlabel('Number of Organizations')
ax1.set_ylabel('Number of Training Rounds')
ax1.set_title('Old Model')
ax1.set_xticks(x)
ax1.set_xticklabels(num_orgs)
ax1.legend()

# New model histogram
ax2.bar(x - width/2, new_homo_rounds, width, label='Homogeneous', color='blue')
ax2.bar(x + width/2, new_hetero_rounds, width, label='Heterogeneous', color='red')
ax2.set_xlabel('Number of Organizations')
ax2.set_ylabel('Number of Training Rounds')
ax2.set_title('New Model')
ax2.set_xticks(x)
ax2.set_xticklabels(num_orgs)
ax2.legend()

plt.tight_layout()
plt.show()