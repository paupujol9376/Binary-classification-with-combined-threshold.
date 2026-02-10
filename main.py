import pandas as pd
import zipfile
import glob
import os

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

contadors=["r002","r003","r004","r005"]
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


#QUARTILS 