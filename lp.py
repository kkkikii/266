# Import Python packages- Numpy, matplotlib, and cvxpy
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import pandas as pd
import random
import os

os.chdir("data")

# load flow data
Q_1 = pd.read_csv("inflow1.tsv", sep='\t', header=35, skiprows=[36])
Q_2 = pd.read_csv("inflow2.tsv", sep='\t', header=35, skiprows=[36])
Q_3 = pd.read_csv("inflow3.tsv", sep='\t', header=35, skiprows=[36])
Q_4 = pd.read_csv("inflow4.tsv", sep='\t', header=35, skiprows=[36])
Q_5 = pd.read_csv("inflow5.tsv", sep='\t', header=35, skiprows=[36])
Q_6 = pd.read_csv("inflow6.tsv", sep='\t', header=35, skiprows=[36])
Q_7 = pd.read_csv("inflow7.tsv", sep='\t', header=35, skiprows=[36])
Q_8 = pd.read_csv("inflow8.tsv", sep='\t', header=35, skiprows=[36])
Q_9 = pd.read_csv("inflow9.tsv", sep='\t', header=35, skiprows=[36])
Q_10 = pd.read_csv("inflow10.tsv", sep='\t', header=35, skiprows=[36])

Q_out = Q_1.mean_va - Q_2.mean_va - Q_3.mean_va - Q_4.mean_va - Q_5.mean_va - Q_6.mean_va - Q_7.mean_va # cfs
Q_in = Q_10.mean_va # cfs

# load irrigation data
wa_irrigation = pd.read_csv("water_use_wa.tsv", sep='\t', header=51, skiprows=1)
or_irrigation = pd.read_csv("water_use_or.tsv", sep='\t', header=51, skiprows=1)

wa_counties = ["Whitman", "Adams", "Walla Walla",
               "Franklin", "Garfield", "Columbia",
               "Yakima", "Benton", "Klickitat", "Lewis",
               "Skamania", "Cowlitz", "Clark", "Wahkiakum"]
or_counties = ["Umatilla", "Morrow", "Gilliam",
               "Sherman", "Hood River", "Wasco",
               "Multnomah", "Clackamas", "Jefferson",
               "Crook", "Wheeler", "Deschutes", "Grant",
               "Lane", "Linn", "Washington", "Columbia"]

col_name = "Irrigation, Total total self-supplied withdrawals, fresh, in Mgal/d"
wa_irrigation[col_name] = pd.to_numeric(wa_irrigation[col_name], errors="coerce")
or_irrigation[col_name] = pd.to_numeric(or_irrigation[col_name], errors="coerce")
total_irrigation = 0
for county in wa_counties:
    val = wa_irrigation[wa_irrigation["county_nm"] == county + " County"][col_name].values[0]
    if not pd.isnull(val):
        total_irrigation += val

for county in or_counties:
    val = or_irrigation[or_irrigation["county_nm"] == county + " County"][col_name].values[0]
    if not pd.isnull(val):
        total_irrigation += val

# print("Total irrigation is " + str(total_irrigation) + " Mgal/d")

mgal_p_day_to_cfs = 1.5472
total_irrigation *= mgal_p_day_to_cfs

# print("Total irrigation is " + str(total_irrigation) + " cfs")


# Make LP function
def min_objective(Q_U, Q_L, K_U, K_L, D, Rem_U, Rem_L, P_U, P_L, max_power_U, max_power_L, cost_electricity, fish_max_flow, C_Rem_U, C_Rem_L, C_Rep_U, C_Rep_L, w_1, w_2, w_3, T):

    # Make variables
    S_U = cp.Variable(T+1) # variables for storage in upper reservoirs
    S_L = cp.Variable(T+1) # variables for storage in lower reservoirs
    R_U = cp.Variable(T) # variable for release from upper reservoirs
    R_L = cp.Variable(T) # variable for release from lower reservoirs

    if Rem_U:
        K_U = 0
    if Rem_L:
        K_L = 0

    # Add constraints
    constraints = [
        S_U[0] == K_U / 2,
        S_L[0] == K_L / 2,
        S_L <= K_L,
        S_U <= K_U,
        S_U >= 0, S_L >= 0, R_U >= 0, R_L >= 0,
        S_L[T] == K_L / 2,
        S_U[T] == K_U / 2
    ]

    # ecology = 0
    ecology = 10000 * T * (1 - Rem_U) + 10000 * T * (1 - Rem_L)
    irrigation = 0
    cost = (
            (Rem_U * C_Rem_U + (1 - Rem_U) * C_Rep_U)
            + (Rem_L * C_Rem_L + (1 - Rem_L) * C_Rep_L)
    )

    for t in range(T-1):
        constraints.append(S_U[t] == S_U[t+1] + Q_U[t+1] - R_U[t+1])
        constraints.append(S_L[t] == S_L[t+1] + Q_L[t+1] + R_U[t+1] - R_L[t+1])

        # Make objective
        eco_val = - cp.min(cp.hstack([R_L[t], fish_max_flow])) - cp.min(cp.hstack([R_U[t], fish_max_flow]))
        ecology += 0 if (4 > t % 12) or (t % 12 > 9) else eco_val
        power_U = P_U * R_U[t] * (1 - Rem_U)
        power_L = P_L * R_L[t] * (1 - Rem_L)
        power = cp.min(cp.hstack([power_L, max_power_L])) + cp.min(cp.hstack([power_U, max_power_U]))
        cost -= power * cost_electricity
        irrigation += cp.max(cp.hstack([D[t] - R_L[t], 0]))

    obj = cp.Minimize(w_1 * ecology + w_2 * cost + w_3 * irrigation)

    # Solve the LP
    prob = cp.Problem(obj, constraints)
    prob.solve(solver="GLPK_MI")

    return {
        "prob": prob,
        "obj": obj,
        "S_U": S_U,
        "S_L": S_L,
        "R_U": R_U,
        "R_L": R_L,
        "ecology": ecology.value,
        "cost": cost.value,
        "irrigation": irrigation.value
    }

# conversion factors
days_per_month = 30
hours_per_month = 24 * days_per_month
seconds_per_month = hours_per_month * 60 * 60
cubic_ft_to_taf = 2.29569 * 10e-8

# initialize variables
T = 132
C_Rem_U = 504.7 # million USD
C_Rem_L = 575.3 # million USD
C_Rep_U = 58.6 # million USD
C_Rep_L = 51.4 # million USD
cost_electricity = 0.0875 * 1000 / 10e6 # million USD / MWh
P_U = 385.99 # efficiency* hydraulic head* density* g when Q=TAF and hp=MWh
P_L = 330.69 # efficiency* hydraulic head* density* g when Q=TAF and hp=MWh
K_U = 45694335100 # cubic feet
K_L = 1955839510 # cubic feet
max_capacity_U = (1620) * hours_per_month # MWh
max_capacity_L = (1413) * hours_per_month # MWh
fish_max_flow = 150000 * seconds_per_month * cubic_ft_to_taf # taf

irr_arr = np.ones(T) * total_irrigation * seconds_per_month * cubic_ft_to_taf
for i in range(T):
    if (i+1) % 12 > 3: # if it is NOT December, January, February, or March
        irr_arr[i] = irr_arr[i] * 1.5
    else: # else it is December, January, February, or March
        irr_arr[i] = 0

# change directory to write results
os.chdir("results")

Rem_U = [0.0, 1.0, 0.0, 1.0]
Rem_L = [0.0, 0.0, 1.0, 1.0]
result = {}
i = 0
for w1 in [10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 0.2, 0.3, 0.4, 0.5]:
    for w2 in np.arange(0.0, 1 - w1, 0.05):
        w3 = 1.0 - w1 - w2
        for j in range(len(Rem_U)):
            try:
                temp = min_objective(
                    Q_in * seconds_per_month * cubic_ft_to_taf,
                    (Q_8.mean_va + Q_9.mean_va) * seconds_per_month * cubic_ft_to_taf,
                    K_U * cubic_ft_to_taf,
                    K_L * cubic_ft_to_taf,
                    irr_arr,
                    Rem_U[j],
                    Rem_L[j],
                    P_U,
                    P_L,
                    max_capacity_U,
                    max_capacity_L,
                    cost_electricity,
                    fish_max_flow,
                    C_Rem_U,
                    C_Rem_L,
                    C_Rep_U,
                    C_Rep_L,
                    w1,
                    w2,
                    w3,
                    T
                )

                name = "w1=" + str(round(w1, 6)) + "_w2=" + str(round(w2, 2)) + "_w3=" + str(round(w3, 2)) + "_RemU=" + str(int(Rem_U[j])) + "_RemL=" + str(int(Rem_L[j]))
                result[name] = temp
                temp_df = pd.DataFrame({"S_U": temp["S_U"].value[1:], "S_L": temp["S_L"].value[1:], "R_L": temp["R_L"].value, "R_U": temp["R_U"].value})
                temp_df.to_csv(name + ".csv")
                i += 1
            except:
                i += 1
                print("Failed on iteration " + str(i))
                print(name)

            if not (i % 10):
                print("Completed iteration " + str(i) + " of 680")

print("Exited loop!")

# plot pareto frontier
# from https://stackoverflow.com/questions/37000488/how-to-plot-multi-objectives-pareto-frontier-with-deap-in-python

def simple_cull(inputPoints, dominates):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def dominates(row, candidateRow):
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row)

inputPoints = [[entry["ecology"], entry["irrigation"], entry["cost"]] for _, entry in result.items()]
paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
dp = np.array(list(dominatedPoints))
pp = np.array(list(paretoPoints))
ax.scatter(dp[:,0],dp[:,1],dp[:,2])
ax.scatter(pp[:,0],pp[:,1],pp[:,2],color='red')
ax.set_xlabel("ecology")
ax.set_ylabel("irrigation")
ax.set_zlabel("cost")

triang = mtri.Triangulation(pp[:,0],pp[:,1])
ax.plot_trisurf(triang,pp[:,2],color='red')
plt.savefig("3Dfrontier.png")
print("3D pareto frontier complete!")
