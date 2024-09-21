import sys
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton, QComboBox,QScrollArea,QGridLayout,QMessageBox
from PyQt5.QtCore import Qt
import pandas as pd
from standardFunc import load_data, split_sessions, print_notification
import os

import sys
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QPushButton, QComboBox, \
    QScrollArea, QMessageBox
from PyQt5.QtCore import Qt
import pandas as pd
from standardFunc import load_data, split_sessions, print_notification
import os
import sys
import os
import importlib.util
# Chemin absolu vers le fichier
module_name = "a6_evalution_InitialModel_bayNewTimeCrossed"
module_file = os.path.join("..\\data_preprocessing", f"{module_name}.py")  # Ajustez le chemin si nécessaire

spec = importlib.util.spec_from_file_location(module_name, module_file)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# Maintenant, vous pouvez accéder à la fonction
train_and_evaluate_XGBOOST_model = module.train_and_evaluate_XGBOOST_model
class FeatureSelector(QWidget):
    def __init__(self, df, columnToolsVariables_excluded, columnFeatures_excluded, process_function=None):
        super().__init__()
        self.df = df
        self.columnToolsVariables_excluded = columnToolsVariables_excluded
        self.columnFeatures_excluded = columnFeatures_excluded
        self.feature_columns = []
        self.feature_checkboxes = {}
        self.process_function = process_function  # Initialisation de process_function
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Scroll area for features
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QHBoxLayout(scroll_content)

        # Features comprises dans l'analyse
        features_layout = QHBoxLayout()
        columns = [col for col in self.df.columns if
                   col not in self.columnToolsVariables_excluded and col not in self.columnFeatures_excluded]

        # Divide columns into groups of 20
        column_groups = [columns[i:i + 20] for i in range(0, len(columns), 20)]

        for group in column_groups:
            group_layout = QVBoxLayout()
            group_layout.addWidget(QLabel("<b>Features comprises dans l'analyse</b>"))
            group_layout.setAlignment(Qt.AlignLeft)
            for col in group:
                feature_layout = QHBoxLayout()
                cb = QCheckBox()
                cb.setChecked(True)  # Set all checkboxes to checked by default
                cb.stateChanged.connect(self.updateFeatureColumns)
                self.feature_checkboxes[col] = cb
                feature_layout.addWidget(cb)

                label_layout = QVBoxLayout()
                label_layout.setSpacing(0)  # Remove spacing between labels
                label_layout.addWidget(QLabel(f"<b>{col}</b>"))

                # Ajout des valeurs min et max avec 3 chiffres après la virgule
                min_val, max_val = self.df[col].min(), self.df[col].max()
                label_layout.addWidget(QLabel(f"Min:{min_val:.3f},Max:{max_val:.3f}"))

                feature_layout.addLayout(label_layout)
                feature_layout.setAlignment(Qt.AlignLeft)
                group_layout.addLayout(feature_layout)

            features_layout.addLayout(group_layout)

        scroll_layout.addLayout(features_layout)
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)

        # Variables de travail et Features exclues
        excluded_layout = QHBoxLayout()

        work_vars_layout = QVBoxLayout()
        work_vars_layout.addWidget(QLabel("<b>Variables de travail</b>"))
        work_vars_layout.setAlignment(Qt.AlignLeft)
        for col in self.columnToolsVariables_excluded:
            work_vars_layout.addWidget(QLabel(f"<b>{col}</b>"), alignment=Qt.AlignLeft)
        excluded_layout.addLayout(work_vars_layout)

        excluded_features_layout = QVBoxLayout()
        excluded_features_layout.addWidget(QLabel("<b>Features exclues par défaut</b>"))
        excluded_features_layout.setAlignment(Qt.AlignLeft)
        for col in self.columnFeatures_excluded:
            feature_layout = QHBoxLayout()
            cb = QCheckBox()
            cb.setChecked(False)  # Set excluded features to unchecked by default
            cb.stateChanged.connect(self.updateFeatureColumns)
            self.feature_checkboxes[col] = cb
            feature_layout.addWidget(cb)

            label_layout = QVBoxLayout()
            label_layout.setSpacing(0)  # Remove spacing between labels
            label_layout.addWidget(QLabel(f"<b>{col}</b>"))

            # Ajout des valeurs min et max pour les features exclues
            min_val, max_val = self.df[col].min(), self.df[col].max()
            label_layout.addWidget(QLabel(f"Min:{min_val:.3f},Max:{max_val:.3f}"))

            feature_layout.addLayout(label_layout)
            feature_layout.setAlignment(Qt.AlignLeft)
            excluded_features_layout.addLayout(feature_layout)
        excluded_layout.addLayout(excluded_features_layout)

        main_layout.addLayout(excluded_layout)

        # Configuration selector and save button
        config_layout = QHBoxLayout()
        self.config_selector = QComboBox()
        self.config_selector.addItems(["All", "Filtre 1", "Filtre 2"])
        self.config_selector.currentTextChanged.connect(self.loadConfiguration)
        config_layout.addWidget(self.config_selector)

        save_button = QPushButton("Sauvegarder")
        save_button.clicked.connect(self.saveConfiguration)
        config_layout.addWidget(save_button)

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.run_analysis)
        config_layout.addWidget(run_button)

        main_layout.addLayout(config_layout)

        self.setLayout(main_layout)
        self.setWindowTitle('Sélecteur de Features')
        self.resize(800, 600)  # Ajuster la taille de la fenêtre si nécessaire

        # Initialize with "All" selected
        self.loadConfiguration("All")

        self.show()

    def updateFeatureColumns(self):
        self.feature_columns = [col for col, cb in self.feature_checkboxes.items() if cb.isChecked()]

    def loadConfiguration(self, config_name):
        if config_name == "All":
            for col, cb in self.feature_checkboxes.items():
                cb.setChecked(col not in self.columnFeatures_excluded)
        else:
            try:
                with open('feature_config.json', 'r') as f:
                    configs = json.load(f)
                if config_name in configs:
                    for col, checked in configs[config_name].items():
                        if col in self.feature_checkboxes:
                            self.feature_checkboxes[col].setChecked(checked)
                else:
                    QMessageBox.warning(self, "Configuration non trouvée",
                                        f"La configuration '{config_name}' n'existe pas.")
            except FileNotFoundError:
                QMessageBox.information(self, "Fichier non trouvé", "Aucune configuration sauvegardée.")
            except json.JSONDecodeError:
                QMessageBox.warning(self, "Erreur de fichier", "Le fichier de configuration est corrompu.")

    def saveConfiguration(self):
        config_name = self.config_selector.currentText()
        if config_name == "All":
            QMessageBox.warning(self, "Sauvegarde impossible", "Impossible de sauvegarder la configuration 'All'.")
            return

        config = {col: cb.isChecked() for col, cb in self.feature_checkboxes.items()}
        try:
            with open('feature_config.json', 'r+') as f:
                try:
                    configs = json.load(f)
                except json.JSONDecodeError:
                    configs = {}

                configs[config_name] = config
                f.seek(0)
                json.dump(configs, f, indent=4)
                f.truncate()

            QMessageBox.information(self, "Sauvegarde réussie",
                                    f"Configuration '{config_name}' sauvegardée avec succès.")
        except IOError as e:
            QMessageBox.critical(self, "Erreur de sauvegarde", f"Impossible de sauvegarder la configuration: {str(e)}")

    def get_selected_features_data(self):
        """
        Retourne un dictionnaire contenant les données nécessaires pour l'analyse.
        """
        self.updateFeatureColumns()  # Assurez-vous que feature_columns est à jour
        selected_data = {
            'feature_columns': self.feature_columns,
            'selectedFeatures_df': self.df[self.feature_columns],
            'initial_df': self.df  # Ajout du DataFrame complet
        }
        return selected_data

    def run_analysis(self):
        """
        Méthode appelée lorsque le bouton 'Run' est cliqué.
        Elle récupère les données sélectionnées et lance l'analyse.
        """
        selected_data = self.get_selected_features_data()

        # Affichage des informations dans la console
        print("Colonnes sélectionnées:")
      #  for column in selected_data['feature_columns']:
       #     print(column)

        # Affichage d'un message dans l'interface graphique
        QMessageBox.information(self, "Colonnes sélectionnées",
                                f"Nombre de colonnes sélectionnées: {len(selected_data['feature_columns'])}\n\n"
                                "Les colonnes ont été affichées dans la console.")

        # Appel de la fonction de traitement si elle est définie
        if self.process_function:
            results = self.process_function(selected_data)
            print("Résultats du traitement :", results)
        else:
            print("Aucune fonction de traitement n'a été définie.")

        # Vous pouvez également émettre un signal ici si vous voulez que le traitement
        # soit géré par une autre partie de votre application
        # self.analysis_requested.emit(selected_data)


# Fonction externe pour traiter les données
def process_selected_features(feature_data):
    """
    Traite les données des features sélectionnées.

    :param feature_data: dictionnaire contenant 'features' (liste de noms de colonnes)
                         et 'dataframe' (DataFrame pandas avec les colonnes sélectionnées)
    :return: résultats du traitement
    """
    features_list = feature_data['feature_columns']
    selectedFeatures_df = feature_data['selectedFeatures_df']
    initial_df = feature_data['initial_df']

    print(f"Lignes dans initial_df: {len(initial_df)}.")
    print(f"selectedFeatures_df: {len(selectedFeatures_df)} features.")
    print("Liste des features sélectionnées :")
    for feature in features_list:
        print(f"- {feature}")

    print("\nAperçu des données de initial_df:")
    print(initial_df.head())
    print("\nAperçu des données de selectedFeatures_df:")
    print(selectedFeatures_df.head())


    # 2. Définition des constantes et paramètres globaux
    DEVICE_ = 'cuda'  # Utiliser 'cuda' pour GPU, 'cpu' sinon
    USE_OPTIMIZED_THRESHOLD_ = False  # True pour optimiser le seuil, False pour utiliser un seuil fixe
    FIXED_THRESHOLD_ = 0.54  # Seuil fixe à utiliser si USE_OPTIMIZED_THRESHOLD est False
    NUM_BOOST_MIN_ = 400  # Nombre minimum de boosting rounds
    NUM_BOOST_MAX_ = 1000  # Nombre maximum de boosting rounds
    N_TRIALS_OPTIMIZATION_ = 7  # Nombre d'essais pour l'optimisation avec Optuna
    NB_SPLIT_TSCV_ = 8  # Nombre de splits pour la validation croisée temporelle
    NANVALUE_TO_NEWVAL_ = 0  # Valeur de remplacement pour les NaN si les NAN oont été remplacés par une valeur dans l'étude de procession des featurs

    # 3. Exécution de la fonction principale pour entraîner et évaluer le modèle
    results = train_and_evaluate_XGBOOST_model(initial_df=initial_df, n_trials_optimization=N_TRIALS_OPTIMIZATION_,
                                               device=DEVICE_, use_optimized_threshold=USE_OPTIMIZED_THRESHOLD_,
                                               fixed_threshold=FIXED_THRESHOLD_,
                                               num_boost_min=NUM_BOOST_MIN_, num_boost_max=NUM_BOOST_MAX_,
                                               nb_split_tscv=NB_SPLIT_TSCV_, nanvalue_to_newval=NANVALUE_TO_NEWVAL_)

    # 4. Utilisation des résultats de l'optimisation
    if results is not None:
        print("Meilleurs hyperparamètres trouvés:", results['study'].best_params)
        print("Meilleur score:", results['study'].best_value)
        print("Seuil optimal:", results['optimal_threshold'])
    else:
        print("L'entraînement n'a pas produit de résultats.")

    results = {
        "nombre_features": len(features_list)
       # "apercu_donnees": df.head().to_dict()
        # Ajoutez d'autres résultats pertinents ici
    }

    return results

# Ajoutez ici votre logiq
if __name__ == '__main__':
    #user_choice = input("Appuyez sur Entrée pour calculer les features sans la afficher. \n"
     #                   "Appuyez sur 'd' puis Entrée pour les calculer et les afficher : \n"
      #                  "Appuyez sur 's' puis Entrée pour les calculer et les afficher :")

    # Nom du fichier
    file_name = "Step5_Step4_Step3_Step2_MergedAllFile_Step1_2_merged_extractOnlyFullSession_OnlyShort_feat_winsorized.csv"

    # Chemin du répertoire
    directory_path = "C:\\Users\\aulac\\OneDrive\\Documents\\Trading\\VisualStudioProject\\Sierra chart\\xTickReversal\\simu\\4_0_4TP_1SL\\merge13092024"

    # Construction du chemin complet du fichier
    file_path = os.path.join(directory_path, file_name)
    app = QApplication(sys.argv)
    df = load_data(file_path)
    columnToolsVariables_excluded=['class_binaire', 'date', 'trade_category', 'SessionStartEnd','candleDir']
    columnFeatures_excluded=['deltaTimestampOpening', 'deltaTimestampOpeningSection5min', 'deltaTimestampOpeningSection5index',
     'deltaTimestampOpeningSection30min']

    # Création de l'instance FeatureSelector avec process_selected_features
    feature_selector = FeatureSelector(df, columnToolsVariables_excluded, columnFeatures_excluded,
                                       process_selected_features)
    sys.exit(app.exec_())