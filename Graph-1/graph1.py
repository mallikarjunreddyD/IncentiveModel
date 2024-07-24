# To install packages: python3 -m pip install pandas openpyxl matplotlib numpy
# Plotting graphs -
# a) Number of Training Rounds vs Valuation Difference
# b) Max Payoff(Social Welfare) vs Valuation Difference

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to read parameters from Excel
def read_parameters_from_excel(file_path):
    df = pd.read_excel(file_path)
    
    params = {
        'RVector': np.array(df['RVector'].dropna()),
        'PiVector': np.array(df['PiVector'].dropna()),
        'S': np.array(df['S'].dropna()),
        'D': np.array(df['D'].dropna()),
        'K': df['K'].iloc[0],
        'T': df['T'].iloc[0],
        'TUL': df['TUL'].iloc[0],
        'TDL': df['TDL'].iloc[0],
        'e0': df['e0'].iloc[0],
        'e1': df['e1'].iloc[0],
        'uploadCost': df['uploadCost'].iloc[0],
        'downloadCost': df['downloadCost'].iloc[0],
        'investmentCost': df['investmentCost'].iloc[0],
        'minR': df['minR'].iloc[0],
        'maxR': df['maxR'].iloc[0],
        'penalty': df['penalty'].iloc[0],
        'stepsize': df['stepsize'].iloc[0],
        'threshold': df['threshold'].iloc[0],
        'lamda': df['lamda'].iloc[0],
        'UnitRevenue1': np.array(df['UnitRevenue1'].dropna()),
        'UnitRevenue2': np.array(df['UnitRevenue2'].dropna()),
        'UnitRevenue3': np.array(df['UnitRevenue3'].dropna()),
        'UnitRevenue4': np.array(df['UnitRevenue4'].dropna())
    }
    
    return params

# Function to run the old model
def run_old_model(UnitRevenue, params):
    RVector = params['RVector'].copy()
    PiVector = params['PiVector'].copy()
    convergeFlags = np.array([False] * len(RVector), dtype=bool)

    S = params['S']
    D = params['D']
    K = params['K']
    T = params['T']
    TUL = params['TUL']
    TDL = params['TDL']
    e0 = params['e0']
    e1 = params['e1']
    uploadCost = params['uploadCost']
    downloadCost = params['downloadCost']
    investmentCost = params['investmentCost']
    minR = params['minR']
    maxR = params['maxR']
    penalty = params['penalty']
    stepsize = params['stepsize']
    threshold = params['threshold']
    
    rounds = []
    payoff_history = [[] for _ in range(6)]
    rF_history = [[] for _ in range(6)]
    pi_history = [[] for _ in range(6)]

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
            r = getR(i)                       # Step 6
            newR = getNewR(r, RVector[i])     # Step 7
            newPi = getNewPi(PiVector[i], i)  # Step 8   # For old-model
            if abs(newR - r) > threshold:     # Step 9
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

# Function to run the new model
def run_new_model(UnitRevenue, params):
    RVector = params['RVector'].copy()
    PiVector = params['PiVector'].copy()
    convergeFlags = np.array([False] * len(RVector), dtype=bool)

    S = params['S']
    D = params['D']
    K = params['K']
    T = params['T']
    TUL = params['TUL']
    TDL = params['TDL']
    e0 = params['e0']
    e1 = params['e1']
    uploadCost = params['uploadCost']
    downloadCost = params['downloadCost']
    investmentCost = params['investmentCost']
    minR = params['minR']
    maxR = params['maxR']
    penalty = params['penalty']
    stepsize = params['stepsize']
    threshold = params['threshold']
    lamda = params['lamda']

    rounds = []
    payoff_history = [[] for _ in range(6)]
    rF_history = [[] for _ in range(6)]
    pi_history = [[] for _ in range(6)]

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
            # payOff = getUtility(rF, organisation) - getCost(rF, organisation) - penalty * getPenalityTerm() + PiVector[organisation]
            payOff = getUtility(rF, organisation) - getCost(rF, organisation) - penalty * getPenalityTerm() 
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
            newPi = getNewModelPi(r, i)    # For new-model
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

# Main execution
file_path = 'parameters.xlsx'  
params = read_parameters_from_excel(file_path)

UnitRevenues = [params['UnitRevenue1'], params['UnitRevenue2'], params['UnitRevenue3'], params['UnitRevenue4']]
valuation_differences = [2, 4, 8, 16]

old_model_rounds = []
new_model_rounds = []
old_model_max_payoff = []
new_model_max_payoff = []

for UnitRevenue in UnitRevenues:
    old_rounds, old_payoff, _, _ = run_old_model(UnitRevenue, params)
    new_rounds, new_payoff, _, _ = run_new_model(UnitRevenue, params)

    old_model_rounds.append(len(old_rounds))
    new_model_rounds.append(len(new_rounds))

    old_model_max_payoff.append(max([max(p) for p in old_payoff]))
    new_model_max_payoff.append(max([max(p) for p in new_payoff]))

# Plotting the results
plt.figure(figsize=(12, 5))

# First graph: Valuation Difference vs Number of Training Rounds
plt.subplot(1, 2, 1)
plt.plot(valuation_differences, old_model_rounds, marker='o', label='Old Model')
plt.plot(valuation_differences, new_model_rounds, marker='x', label='New Model')
plt.xlabel('Valuation Difference')
plt.ylabel('Number of Training Rounds')
plt.title('Number of Training Rounds vs Valuation Difference')
plt.legend()

# Second graph: Valuation Difference vs Max Payoff
plt.subplot(1, 2, 2)
plt.plot(valuation_differences, old_model_max_payoff, marker='o', label='Old Model')
plt.plot(valuation_differences, new_model_max_payoff, marker='x', label='New Model')
plt.xlabel('Valuation Difference')
plt.ylabel('Max Payoff')
plt.title('Max Payoff vs Valuation Difference')
plt.legend()

plt.tight_layout()
plt.show()