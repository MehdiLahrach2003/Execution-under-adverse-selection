# Exercice 1 : 

prices = [100, 102, 101, 105, 107]
rendement = []
for i in range(1, len(prices)): 
    rendement.append(prices[i]/prices[i-1] - 1)
print(rendement)

# Exercice 2 : 

returns = [0.02, -0.01, 0.03, 0.015, -0.005]
# On commence par calculer la somme des rendements : 
S = 0
for i in range(len(returns)): 
    S += returns[i]
# On calcule maintenant le rendement moyen 
print(S/len(returns))

# Exercice 3 : 

returns = [0.02, -0.01, 0.03, 0.00, -0.005, 0.015]
S1, S2, S3 = 0, 0, 0
for i in range(len(returns)) : 
    if returns[i] > 0 :
        S1 += 1  
    elif returns[i] < 0 :  
        S2 += 1
    else : 
        S3 += 1
    
print("positifs = ", S1)
print("negatifs = ", S2)  
print("nuls = ", S3)

# Exercice 4 : 

returns = [0.10, -0.05, 0.02]
P = 1
for i in range(len(returns)): 
    P *= 1 + returns[i]
print(P-1)

# Exercice 5 : 

weights = [0.50, 0.30, 0.20]
returns = [0.04, 0.02, -0.01]
assets = ["Actions", "Obligations", "Private Equity"]

S = 0
for i in range(len(weights)): 
    S += weights[i]*returns[i]
print(S)

# Exercice 6 : 

assets = ["Actions", "Obligations", "Private Equity", "Monétaire"]
weights = [0.40, 0.35, 0.15, 0.10]
returns = [0.05, 0.015, -0.02, 0.005]

S = 0
for i in range(len(assets)): 
    r = weights[i]*returns[i]
    print(assets[i], ":", r)
    S += r 
print("La performance totale du portefeuille est : ", S)

# Exercice 7 :

assets = ["Actions", "Obligations", "Private Equity", "ESG"]
returns = [0.06, 0.015, -0.03, 0.025]

max_returns = returns[0]
min_returns = returns[0]

for i in range(len(returns)) : 
    if returns[i] > max_returns : 
        max_returns = returns[i]
    if returns[i] < min_returns :
        min_returns = returns[i]
        
for j in range(len(returns)): 
    if returns[j] == max_returns : 
        print("Meilleur actif : ", assets[j], ", rendement = ", max_returns)
    if returns[j] == min_returns : 
        print("Pire actif : ", assets[j], ", rendement = ", min_returns) 
        
# Exercice 8 :

assets = ["Action A", "Obligation B", "ETF C"]
quantities = [100, 50, 200]
prices = [25, 98, 12]

S = 0
for i in range(len(assets)): 
    print(assets[i], ":", quantities[i]*prices[i])
    S += quantities[i]*prices[i]
print("Valeur totale :", S)

# Exercice 9 :

assets = ["Action A", "Obligation B", "ETF C"]
values = [2500, 4900, 2400]

S = 0
for i in range(len(assets)): 
    S += values[i]

for j in range(len(assets)):
    print(assets[j], ":", values[j]/S)
    
# Exercice 10 :

portfolio_value = 10000
current_weights = [0.50, 0.30, 0.20]
target_weights = [0.40, 0.40, 0.20]
assets = ["Actions", "Obligations", "Monétaire"]

for i in range(len(assets)): 
    val_actuelle = portfolio_value * current_weights[i]
    val_cible = portfolio_value * target_weights[i]
    ecart = val_cible - val_actuelle
    if ecart > 0 : 
        print(assets[i], ": acheter ", ecart)
    elif ecart < 0 :  
        print(assets[i], ": vendre ", abs(ecart))
    else : 
        print(assets[i], ": ne rien faire")
        
# Exercice 11 :

assets = ["Actions", "Obligations", "Private Equity", "ESG"]
returns = [0.02, -0.01, -0.08, -0.03]
threshold = -0.05

for i in range(len(returns)): 
    if returns[i] < -0.05 : 
        print("Alerte :", assets[i], "a une perte de", returns[i])

# Exercice 12 :

portfolio_values = [100, 105, 103, 110, 104, 98, 102]

max_historique = portfolio_values[0]
max_drawdown = 0 

for i in portfolio_values : 
    drawdown = (i-max_historique) / max_historique
    if i > max_historique : 
        max_historique = i 
    if drawdown < max_drawdown:
        max_drawdown = drawdown
print(max_drawdown)

# Exercice 13 :

import statistics
returns = [0.02, -0.01, 0.03, 0.00, -0.02]
print(statistics.stdev(returns))

# Exercice 14 :

returns = [0.02, -0.01, 0.03, 0.00, -0.02]
risk_free_rate = 0.005

S = 0
for i in range(len(returns)) : 
    S += returns[i]

moyenne = S/len(returns)

print("Le sharpe ratio vaut :", (moyenne - risk_free_rate)/statistics.stdev(returns))

# Exercice 15 :

portfolio_returns = [0.02, -0.01, 0.03, 0.01]
benchmark_returns = [0.015, -0.005, 0.025, 0.02]

surperformances = []
for i in range(len(portfolio_returns)) : 
    surperformances.append(portfolio_returns[i] - benchmark_returns[i])

S = 0
for j in range(len(surperformances)) : 
    S += surperformances[j]
m = S/len(surperformances)

print(surperformances)
print(m)

# Exercice 16 :

portfolio_returns = [0.02, -0.01, 0.03, 0.01, 0.00]
benchmark_returns = [0.015, -0.005, 0.025, 0.02, -0.01]

compteur = 0 
for i in range(len(portfolio_returns)) : 
    if portfolio_returns[i] - benchmark_returns[i] > 0 : 
        compteur += 1
print("Nombre de périodes de surperformance :", compteur)

# Exercice 17 :

companies = ["Entreprise A", "Entreprise B", "Entreprise C"]
environment = [80, 60, 90]
social = [70, 75, 85]
governance = [60, 80, 70]

for i in range(len(companies)) : 
    print(companies[i], ":", 0.4*environment[i] + 0.3*social[i] + 0.3*governance[i])
    
# Exercice 18 :

companies = ["Entreprise A", "Entreprise B", "Entreprise C", "Entreprise D"]
esg_scores = [71.0, 70.5, 82.5, 55.0]
threshold = 70

for i in range(len(esg_scores)): 
    if esg_scores[i] >= 70 : 
        print(companies[i])
        
# Exercice 19 :

bonds = ["Obligation A", "Obligation B", "Obligation C"]
durations = [3, 5, 8]
rate_change = 0.01

for i in range(len(bonds)) : 
    print(bonds[i], ":", -durations[i]*rate_change)
    
# Exercice 20 :

bonds = ["Obligation A", "Obligation B", "Obligation C"]
weights = [0.40, 0.35, 0.25]
durations = [3, 5, 8]

S = 0
for i in range(len(bonds)) : 
    S += weights[i]*durations[i]
print(S)

# Exercice 21 :

assets = ["Actions", "Obligations", "Private Equity", "Monétaire"]
returns = [0.08, 0.015, -0.04, 0.002]

for i in range(len(assets)) : 
    if returns[i] >= 0.05 : 
        print(assets[i], ": forte performance")
    elif returns[i] >= 0 and returns[i] < 0.05 : 
        print(assets[i], ": performance positive modérée")
    else : 
        print(assets[i], ": performance négative")
        
# Exercice 22 :

returns = [0.02, None, -0.01, 0.03, None, 0.00]
nv_liste = []
for i in returns : 
    if i != None : 
        nv_liste.append(i)
print(nv_liste)

# Exercice 23 :

returns = [0.02, None, -0.01, 0.03, None, 0.00]

print([0 if x is None else x for x in returns])

# Exercice 24 :

assets = ["Actions", "Obligations", "Private Equity"]
weights = [0.50, 0.30, 0.20]
returns = [0.04, 0.02, -0.01]

for i in range(len(assets)) : 
    print("La classe", assets[i], "représente", weights[i]*100, "% du portefeuille et a réalisé une performance de", returns[i]*100, "% .")
    
# Exercice 25 :

assets = ["Actions", "Obligations", "Private Equity", "ESG"]
weights = [0.35, 0.40, 0.15, 0.10]
returns = [0.06, 0.015, -0.02, 0.03]
benchmark_returns = [0.05, 0.01, -0.01, 0.025]

contrib = []
S1, S2 = 0, 0
for i in range(len(assets)) : 
    S1 += weights[i]*returns[i]
    S2 += weights[i]*benchmark_returns[i]
    print(assets[i], ":", weights[i]*returns[i])
    contrib.append(weights[i]*returns[i])

print("Performance portefeuille :", S1)  
print("Performance benchmark :", S2)  
print("Surperformance :", S1-S2)

max_contrib = contrib[0]
min_contrib = contrib[0]
for i in range(len(contrib)) : 
    if contrib[i] > max_contrib : 
        max_contrib = contrib[i]
    if contrib[i] < min_contrib :
        min_contrib = contrib[i]
for j in range(len(contrib)): 
    if contrib[j] == max_contrib : 
        print("Meilleur actif : ", assets[j])
    if contrib[j] == min_contrib : 
        print("Pire actif : ", assets[j]) 
        