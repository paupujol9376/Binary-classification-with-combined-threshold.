import pandas as pd
import zipfile
import glob
import os
import numpy as np
from itertools import product

#SEPAREM EL ZIP 

dataset_albert_zip = 'dataverse_files.zip'
with zipfile.ZipFile(dataset_albert_zip, 'r') as zip_ref:
    zip_ref.extractall('datos_albert')


"""
######AQUÍ ENTENEM QUE SON EL R002 I LA ESTRUCUTRA DE CADA PETICIÓ DE AL HARDWARE######

fitxer_benigne='datos_albert/bitcoin-all_perf_events.csv'
# Carreguem el fitxer benigne
df_mostra = pd.read_csv(fitxer_benigne)

# 3. Ensenyar les primeres 5 files i les columnes
print("--- Columnes detectades al fitxer (Aquests són els contadors) ---")
print(df_mostra.columns.tolist())

print("\n--- Primers valors del fitxer 'bitcoin' (Benigne) ---")
print(df_mostra.head())
"""


#PROCESAT DE DADES I ORDERNAR EL DATASET

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
    
    # Mirem el nom del programa (abans del guió) per posar l'etiqueta [cite: 128]
    nom_prog = os.path.basename(fitxer).split("-")[0]
    df_temp['label'] = 'benigne' if nom_prog in benignes else 'maligne'
    
    # Guardem només els 5 sensors i l'etiqueta per estalviar memòria
    total_de_dades.append(df_temp[contadors + ['label']])

# Creem la total_de_dades unificada
df = pd.concat(total_de_dades, ignore_index=True)
#Vull mirar si el dataset es coherent
df.to_csv('dataset_complet.csv', index=False)


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
   else : 4 # ZONA 4


for c in contadors:
    Q1,Q2,Q3 = quartils[c] # TINDREM ELS QUARTIL DEL CONTADORS R002, R003...
    df[c+"_Q"] = df[c].apply(lambda x: recodificar(x, Q1, Q2, Q3)) # LA LAMBDA M'AJUDA DESPRES A EMPRAR LA FUNC DECOFIDICAR


# VEURE QUE FA AIXÒ?????????????
df["combo"] = df[[c+"_Q" for c in contadors]].apply(lambda row: tuple(row), axis=1)

#MIRA LABEL BENIGNES I MALIGNES
conteo = df.groupby(["combo", "label"]).size().unstack(fill_value=0)   # GROUPBY --> agrupa files mateix combinaciói  UNSTACK --> transfromació oer tindre colm benigne i una altre maline , per millor compració visual
print("\n Conteo de combinaciones: ")
print(conteo)

#ORDENAR
conteo_ordenado = conteo.sort_values(by="maligno", ascending=False)    # ORDENACI PER VEURE ELQ UARTILS QUEHAN SIGUT MALIGNES
print(" \n Combinaciones más asociadas a malignos:")
print(conteo_ordenado.head(10))



#############################################################


