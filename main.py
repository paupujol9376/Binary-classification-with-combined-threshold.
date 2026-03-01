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

#############################################################
#         PROCESAT DE DADES I ORDERNAR EL DATASET
#############################################################

benignes = ["bitcoin", "bubble", "bzip2", "coremark", "dhrystone", "ffmpeg", 
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
#             DATASET 50%-50% 
#############################################################

#IGUALEM EL DATASET


df= pd.read_csv("dataset_complet.csv")
df_benigne =df[df.label == 'benigne']
df_maligne =df[df.label == 'maligne']

n_samples = min(len(df_benigne), len(df_maligne))

# Fem el re-mostreig (undersampling de la majoritària)
df_benigne_downsampled = df_benigne.sample(n=n_samples, random_state=42)
df_maligne_downsampled = df_maligne.sample(n=n_samples, random_state=42)

# Unim de nou
df_balanced = pd.concat([df_benigne_downsampled, df_maligne_downsampled])

df_balanced.to_csv('dataset_complet_balanced.csv', index=False)

#############################################################
#                     QUARTILS + TUPLAS
#############################################################

#MIRAREM EL QUARTILS NOMÉS AMB ELS CASOS BENIGNES QUE SERIA EL

df= pd.read_csv("dataset_complet_balanced.csv") #  dataset_complet.csv 
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


df["combo"] = df[[c+"_Q" for c in contadors]].apply(lambda row: tuple(row), axis=1)

#MIRA LABEL BENIGNES I MALIGNES
conteo = df.groupby(["combo", "label"]).size().unstack(fill_value=0)   # GROUPBY --> agrupa files mateix combinaciói  UNSTACK --> transfromació oer tindre colm benigne i una altre maline , per millor compració visual
print("\n Conteo de combinaciones: ")
print(conteo)

# Exportar el dataset final amb totes les transformacions
df.to_csv('dataset_quartils.csv', index=False)


df_final = pd.read_csv('dataset_quartils.csv')
balanc = df_final['label'].value_counts(normalize=True) * 100

print("\n--- BALANÇ DEL DATASET FINAL ---")
print(balanc)



#############################################################
#                     ANÁLISIS DE CONTEO Y DISTRIBUCIÓN
#############################################################

print("\n" + "="*40)
print("RESUMEN GENERAL DEL DATASET")
print("="*40)

# 1. Conteo general de filas por etiqueta
resumen_general = df['label'].value_counts()
print(f"Total de muestras Benignas: {resumen_general.get('benigne', 0)}")
print(f"Total de muestras Malignas: {resumen_general.get('maligne', 0)}")
print(f"Total global: {len(df)}")

# 2. Análisis de Tuplas (Combos)
print("\n" + "="*40)
print("ANÁLISIS DE TUPLAS (COMBOS)")
print("="*40)

# Cuántas combinaciones únicas existen
total_combos = df['combo'].nunique()
print(f"Número de combinaciones únicas detectadas: {total_combos}")

# Identificar colisiones (combos que aparecen en ambos bandos)
# Usamos el 'conteo' que ya calculaste con groupby
colisiones = conteo[(conteo['benigne'] > 0) & (conteo['maligne'] > 0)]

print(f"Combinaciones que aparecen en AMBOS (Conflicto): {len(colisiones)}")
print(f"Combinaciones puramente BENIGNAS: {len(conteo[conteo['maligne'] == 0])}")
print(f"Combinaciones puramente MALIGNAS: {len(conteo[conteo['benigne'] == 0])}")

# 3. Mostrar las 10 combinaciones más frecuentes para cada caso
print("\nTOP 5 COMBINACIONES MALIGNAS:")
print(conteo.sort_values(by='maligne', ascending=False)['maligne'].head(5))
# L'ascending = false los ordena de mayor importancia (más repetidos en maligne) a menor 
# Nos quedamos con los 5 primeros para tener una idea.
print("\nTOP 5 COMBINACIONES BENIGNAS:")
print(conteo.sort_values(by='benigne', ascending=False)['benigne'].head(5))


#############################################################
#             TRACATAMENT DEL DATASET PEL MODEL
#############################################################


# reset_index() fa que  "combo"  pasi com columna normal
conteo_export = conteo.reset_index()

# 2. Ordes de més a mnyes perilloses
conteo_export = conteo_export.sort_values(by="maligne", ascending=False)

# 3. Exportar
conteo_export.to_csv('resumen_frecuencias_tuplas.csv', index=False)

#POSEM LA COLUMNA AMB % DE MALIGNE

conteo_export['maligno_ratio'] = conteo_export['maligne'] / (conteo_export['maligne'] + conteo_export['benigne'])
conteo_export.to_csv('dataset_final_amb_tassa_maligne.csv', index=False)






