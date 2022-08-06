
print("Comenzado el proceso solicitado, espere un momento por favor .....")
# Step 0. Import libraries and custom modules
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
# Machine learning -----------------------------------------------------
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
# Preprocessing --------------------------------------------------------
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# Metrics --------------------------------------------------------------
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
# Exporting ------------------------------------------------------------
import pickle
print("")
print("1")
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')

df_interim = df_raw.copy()
print("terminando de cargar los datos del dataset")
# Me quedo con las columnas que me solicitan

df_interim = df_interim[['Latitude','Longitude','MedInc']]
print()
print("Comenzando con el entrenamiento")
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6, random_state=0).fit(df_interim)

clase = kmeans.predict(df_interim)

df_interim['clase']=clase

df_interim.clase = df_interim.clase.astype('category')
print()
print("="*80)
print(" Finalizado el proceso solicitado")
print("="*80)


