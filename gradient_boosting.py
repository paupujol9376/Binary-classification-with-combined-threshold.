import pandas as pd
import zipfile
import glob
import os
import numpy as np
from itertools import product
import time
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

#SEPAREM EL ZIP 

dataset_albert_zip = 'dataverse_files.zip'
with zipfile.ZipFile(dataset_albert_zip, 'r') as zip_ref:
    zip_ref.extractall('datos_albert')

#############################################################
#            PROCESAT DE DADES I ORDERNAR EL DATASET
#############################################################

benignes =["bitcoin", "bubble", "bzip2", "coremark", "dhrystone", "ffmpeg", 
    "mandelbrot", "matrix", "mybench", "polybench", "sha256sum", 
    "sieve", "speedtest", "stream", "stress_c", "stress_m"
]

contadors=["r002","r003","r004","r005","r006"]
total_de_dades=[]

# Llegim tots els CSV de la carpeta datos_alebrt / (dintre de datos_albert)* (qualsevol)
for fitxer in glob.glob("datos_albert/*.csv"):
   
    df_temp = pd.read_csv(fitxer, skiprows=[1])
    # Netegem els noms de les columnes per tenir només el codi rXXX
    df_temp.columns = [c.split('/')[0].strip() if '/' in c else c for c in df_temp.columns]
    
    # Mirem el nom del programa (abans del guió) per posar l'etiqueta
    nom_prog = os.path.basename(fitxer).split("-")[0]
    df_temp['label'] = 'benigne' if nom_prog in benignes else 'maligne'
    
    # Guardem només els 5 sensors i l'etiqueta per estalviar memòria
    total_de_dades.append(df_temp[contadors + ['label']])

# Creem la total_de_dades unificada
df = pd.concat(total_de_dades, ignore_index=True)
#Vull mirar si el dataset es coherent
df.to_csv('dataset_complet.csv', index=False)



#############################################################
#                    QUARTILS + TUPLAS
#############################################################

df= pd.read_csv("dataset_complet.csv")
contadors=["r002","r003","r004","r005","r006"]

# 1) Calcular quartils
quartils = {}

for c in contadors:
    Q1=df[c].quantile(0.25) # INFERIOR
    Q2=df[c].quantile(0.50) # MITJANA
    Q3=df[c].quantile(0.75) #SUPERIOR
    quartils[c]=(Q1, Q2, Q3)

print("Quartils por contadors: ")
for c, q in quartils.items():
    print(c,q)


#RECODIFICACIÓ FUNCIÓ
def recodificar(valor, Q1, Q2, Q3):
   if valor <= Q1: return 1 # ZONA 1
   if valor <= Q2: return 2 # ZONA 2
   if valor <= Q3: return 3 # ZONA 3
   else : return 4 # ZONA 4


for c in contadors:
    Q1,Q2,Q3 = quartils[c] # TINDREM ELS QUARTIL DEL CONTADORS R002, R003...
    df[c+"_Q"] = df[c].apply(lambda x: recodificar(x, Q1, Q2, Q3)) # LA LAMBDA M'AJUDA DESPRES A EMPRAR LA FUNC DECOFIDICAR


# Crear columna amb la combinació
df["combo"] = df[[c+"_Q" for c in contadors]].apply(lambda row: tuple(row), axis=1)

#MIRA LABEL BENIGNES I MALIGNES
conteo = df.groupby(["combo", "label"]).size().unstack(fill_value=0)   # GROUPBY --> agrupa files mateix combinaciói   UNSTACK --> transfromació oer tindre colm benigne i una altre maline , per millor compració visual
print("\n Conteo de combinaciones: ")
print(conteo)

# Exportar el dataset final amb totes les transformacions
df.to_csv('dataset_quartils.csv', index=False)

#############################################################
#            IMPLEMENTACIÓN DE GRADIENT BOOSTING CON GRID SEARCH
#############################################################

# 1. Preparar los datos
X = df[[c+"_Q" for c in contadors]]
y = df['label']

# 2. Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- CONFIGURACIÓ DEL GRID SEARCH ---

param_grid = {
    'n_estimators': [100,150,200],      # Reduït a valors clau
    'learning_rate': [0.6],   # Dos valors representatius
    'max_depth': [3, 4]             # Molt important: no passar de 4 estalvia molta CPU
}

print("\n" + "="*40)
print("BUSCANT MILLORS PARÀMETRES (GRID SEARCH)...")
print("="*40)

ini = time.time()

# Creem el model base
gb_base = GradientBoostingClassifier(random_state=42)

# Configurem la cerca
grid_search = GridSearchCV(estimator=gb_base, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=1)

# Entrenem totes les combinacions
grid_search.fit(X_train, y_train)

fin = time.time()

# --- ANÀLISI DETALLADA DE TOTES LES CONFIGURACIONS ---
print("\nAnalitzant rendiment per classe...")
detalls = []
for params in grid_search.cv_results_['params']:
    m = GradientBoostingClassifier(**params, random_state=42).fit(X_train, y_train)
    y_p = m.predict(X_test)
    rep = classification_report(y_test, y_p, output_dict=True)
    detalls.append({
        'params': str(params),
        'prec_benigne': rep['benigne']['precision'],
        'prec_maligne': rep['maligne']['precision'],
        'f1_benigne': rep['benigne']['f1-score'],
        'f1_maligne': rep['maligne']['f1-score']
    })

df_detalls = pd.DataFrame(detalls)
df_detalls.to_excel('resultats_detallats.xlsx', index=False)
print("Resultats desats a 'resultats_detallats.xlsx''")

# Extraiem el millor model trobat
best_model = grid_search.best_estimator_

print(f"\n✅ RECERCA COMPLETADA")
print(f"TEMPS TOTAL: {fin - ini:.4f} segons")
print(f"MILLORS PARÀMETRES: {grid_search.best_params_}")

# 4. Evaluación con el mejor modelo
y_pred = best_model.predict(X_test)

print("\nMatriz de Confusión (Millor Model):")
print(confusion_matrix(y_test, y_pred))

print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred))

# 5. Importancia de los sensores
importancias = pd.DataFrame({'sensor': X.columns, 'importancia': best_model.feature_importances_})
print("\nImportancia de cada sensor en la decisión:")
print(importancias.sort_values(by='importancia', ascending=False))

print(f"Precisión Entrenamiento: {best_model.score(X_train, y_train):.4f}")
print(f"Precisión Prueba: {best_model.score(X_test, y_test):.4f}")