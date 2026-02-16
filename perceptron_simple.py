
import pandas as pd
import ast
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

#############################################################
#             TRACATAMENT DEL DATASET PEL MODEL             #         
#############################################################

#ELIMINAR LA ZONA GRIS 0,4 a 0,7
#CAL ARREGLAR EL DATASET
#DESFAREM EL COMBO

df_resum=pd.read_csv("dataset_final_amb_tassa_maligne.csv")
df_resum['combo'] = df_resum['combo'].apply(ast.literal_eval)

# 2. Creem les 5 columnes noves a partir de la tupla
noms_sensors = ["r002_Q", "r003_Q", "r004_Q", "r005_Q", "r006_Q"]
df_resum[noms_sensors] = pd.DataFrame(df_resum['combo'].tolist(), index=df_resum.index)

#ELIMINAR ZONA GRIS 

# Eliminem zona gris
df_perceptro = df_resum[(df_resum['maligno_ratio'] <= 0.3) | (df_resum['maligno_ratio'] >= 0.7)]  #ASK BEATRIZ (NO ACABO DE VEURE ELMINAR LA ZONA GRIS,NO SE QUNES MANEREES PODEM MILLORAR-HO....)

# Ara definim les etiquetes (el 0.5 ara és una frontera neta perquè no hi ha res al mig)
df_perceptro['label'] = (df_perceptro['maligno_ratio'] > 0.5).astype(int)


columnes_finals= noms_sensors + ['label']
df_perceptro=df_perceptro[columnes_finals]
df_perceptro.to_csv('dataset_perceptro.csv',index=False)

#############################################################
#                    PERCEPTRÓN SIPLE                       #
#############################################################


X=df_perceptro[noms_sensors]
y=df_perceptro['label']

#NOMRALITZACIÓ
scalat=StandardScaler()
X_scaled = scalat.fit_transform(X)

# 1. Separem en train i test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = Perceptron(max_iter=5000, tol=1e-4,eta0=0.1,class_weight='balanced', random_state=42)#El max_iter es el nombre d'intets,tol és la precisió d'aturada, eta la velocitat per apendre
model.fit(X_train, y_train)

#############################################################
#                    PESOS I AVALUACIÓ                      #
#############################################################

y_pred = model.predict(X_test)
precisio = accuracy_score(y_test, y_pred) 


#ACCURACY

print("---RESULTATS DEL MODEL---")
print(f"Precisió: {precisio*100:.2f}%")
print("\nInforme: ")
print(classification_report(y_test, y_pred))

# importància de cada columna
pesos = model.coef_[0]
importancia = pd.DataFrame({
    'Sensor': noms_sensors,
    'Pes (Influència)': pesos
}).sort_values(by='Pes (Influència)', ascending=False)

print("\n--- INFLUÈNCIA DE CADA SENSOR ---")
print("Un pes positiu indica que el sensor delata el malware.")
print(importancia)






