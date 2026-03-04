import pandas as pd
import ast
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import time

#############################################################
#            TRACTAMENT DEL DATASET PEL MODEL               #         
#############################################################

# 1. Càrrega i neteja
df_resum = pd.read_csv("dataset_final_amb_tassa_maligne.csv")
df_resum['combo'] = df_resum['combo'].apply(ast.literal_eval)

# 2. Creem les 5 columnes dels sensors
noms_sensors = ["r002_Q", "r003_Q", "r004_Q", "r005_Q", "r006_Q"]
df_resum[noms_sensors] = pd.DataFrame(df_resum['combo'].tolist(), index=df_resum.index)

# 3. Assignació de la Label (0 o 1) segons el ratio de malignitat
df_resum['label'] = (df_resum['maligno_ratio'] > 0.5).astype(int)

# 4. Preparem les variables X i y
X = df_resum[noms_sensors]
y = df_resum['label']

# 5. Divisió en Train i Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#############################################################
#            OPTIMITZACIÓ AMB EXPLORACIÓ I EXCEL            #
#############################################################

param_grid = {
    'max_depth': [3, 4, 5, 6, 8],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

ini = time.time()
print("Iniciant exploració i generant Excel amb format de constructor...")

results_list = []
rows_for_excel = []

for params in ParameterGrid(param_grid):
    # Creem el model amb els paràmetres de la iteració
    clf = DecisionTreeClassifier(class_weight='balanced', random_state=42, **params)
    clf.fit(X_train, y_train)
    
    y_pred_iter = clf.predict(X_test)
    
    # Càlcul de mètriques
    acc = accuracy_score(y_test, y_pred_iter)
    precision, _, f1, _ = precision_recall_fscore_support(y_test, y_pred_iter, average=None, zero_division=0)
    
    # GENERACIÓ DE LA CADENA DE TEXT (FORMAT CONSTRUCTOR)
    # Exple: DecisionTreeClassifier(criterion='gini', max_depth=3, ...)
    param_str = ", ".join([f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in params.items()])
    model_name_excel = f"DecisionTreeClassifier({param_str}, class_weight='balanced', random_state=42)"
    
    # Guardem dades per l'Excel amb el format que has demanat
    row = {
        'Configuració Model': model_name_excel,
        'Prec_Benigne': precision[0],
        'Prec_Maligne': precision[1],
        'F1_Benigne': f1[0],
        'F1_Maligne': f1[1],
        'Accuracy': acc
    }
    rows_for_excel.append(row)
    results_list.append({'acc': acc, 'params': params, 'model': clf})

# Exportació a Excel
df_results = pd.DataFrame(rows_for_excel)
df_results.to_excel("resultats_gridsearch_tree.xlsx", index=False)

# Seleccionem el millor model per a la visualització final
best_run = max(results_list, key=lambda x: x['acc'])
model_tree = best_run['model']

fin = time.time()
print(f"TEMPS TOTAL: {fin - ini:.4f} segons. Fitxer generat.")

#############################################################
#                  AVALUACIÓ DEL MILLOR MODEL               #
#############################################################

y_pred = model_tree.predict(X_test)
precisio = accuracy_score(y_test, y_pred) 

print(f"\nMillors paràmetres trobats: {best_run['params']}")
print(f"Precisió final: {precisio*100:.2f}%")

# Importància de les variables
importancia = pd.DataFrame({
    'Sensor': noms_sensors,
    'Importància': model_tree.feature_importances_
}).sort_values(by='Importància', ascending=False)

#############################################################
#                    VISUALITZACIÓ GRÀFICA                  #
#############################################################

fig = plt.figure(figsize=(20, 10))

# 1. Gràfic de l'arbre
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
plot_tree(model_tree, feature_names=noms_sensors, class_names=['Benigne', 'Malware'], 
          filled=True, rounded=True, ax=ax1, fontsize=10)
ax1.set_title(f"Millor Arbre: {best_run['params']}")

# 2. Gràfic d'importància
ax2 = plt.subplot2grid((2, 2), (1, 0))
sns.barplot(x='Importància', y='Sensor', data=importancia, hue='Sensor', palette='viridis', legend=False, ax=ax2)
ax2.set_title("Importància de cada Sensor")

# 3. Matriu de Confusió
ax3 = plt.subplot2grid((2, 2), (1, 1))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benigne', 'Malware']).plot(ax=ax3, cmap='Blues')
ax3.set_title("Matriu de Confusió")

plt.tight_layout()
plt.show()