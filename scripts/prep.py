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
    S =+ r 
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
for i in range(1, len(portfolio_values)): 
    if portfolio_values[i] > max_historique : 
        max_historique = portfolio_values[i]
    drawdown = portfolio_values[i]/max_historique - 1