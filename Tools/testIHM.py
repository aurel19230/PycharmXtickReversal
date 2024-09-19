import tkinter as tk
from tkinter import ttk
import pandas as pd
import json
import os

# Exemple de DataFrame (à remplacer par ton propre DataFrame)
data = {'class_binaire': [1, 0], 'candleDir': [1, -1], 'date': ['2023-01-01', '2023-01-02'],
        'trade_category': ['A', 'B'], 'SessionStartEnd': [10, 20], 'total_count_abv': [5, 10],
        'total_count_blw': [3, 8], 'meanVolx': [2.5, 3.1], 'deltaTimestampOpening': [60, 45],
        'deltaTimestampOpeningSection5min': [15, 20], 'deltaTimestampOpeningSection5index': [3, 4],
        'deltaTimestampOpeningSection30min': [100, 120]}
df = pd.DataFrame(data)

# Fichier pour sauvegarder les colonnes sélectionnées
save_file = "selected_columns.json"

# Fonction pour charger les colonnes sélectionnées depuis le fichier
def load_selected_columns():
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            return json.load(f)
    else:
        # Si le fichier n'existe pas, aucune colonne n'est sélectionnée par défaut
        return {col: False for col in df.columns}

# Fonction pour enregistrer les colonnes sélectionnées dans un fichier
def save_selected_columns(selected_columns):
    with open(save_file, 'w') as f:
        json.dump(selected_columns, f)

# Fonction pour créer l'interface graphique de sélection des colonnes
def select_columns(df):
    root = tk.Tk()
    root.title("Sélection des colonnes")

    selected_columns = load_selected_columns()  # Charger les choix précédents

    def update_columns():
        nonlocal selected_columns
        selected_columns = {col: var_dict[col].get() for col in df.columns}
        save_selected_columns(selected_columns)  # Sauvegarder les choix
        root.quit()

    # Crée un dictionnaire pour stocker les variables associées à chaque checkbox
    var_dict = {col: tk.BooleanVar(value=selected_columns.get(col, False)) for col in df.columns}

    # Interface des colonnes à sélectionner
    for i, col in enumerate(df.columns):
        checkbox = ttk.Checkbutton(root, text=col, variable=var_dict[col])
        checkbox.grid(row=i, column=0, sticky='w')

    # Bouton pour confirmer la sélection
    confirm_button = ttk.Button(root, text="Confirmer", command=update_columns)
    confirm_button.grid(row=len(df.columns) + 1, column=0)

    root.mainloop()
    return [col for col in df.columns if var_dict[col].get()]

# Appel de la fonction de sélection des colonnes
feature_columns = select_columns(df)

print("Colonnes sélectionnées :", feature_columns)
