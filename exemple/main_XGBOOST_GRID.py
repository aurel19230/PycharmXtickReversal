import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

# Set random seeds for reproducibility
RANDOM_SEED = 10
np.random.seed(RANDOM_SEED)

# Hyperparameters
SEQUENCE_LENGTH = 3
TRAIN_WINDOW_SIZE = 120000
TEST_HORIZON = 17000
STEP_SIZE = TEST_HORIZON

# Data Preparation
file_path = 'C:/Users/aulac/OneDrive/Documents/Trading/VisualStudioProject/Sierra chart/xTickReversal/data_simuNew/4_0_5TP_1SL/merge'
file_name = "20190303_20240523_4TicksRevFeaturesCropped_wihSelection"
#file_name = "20190303_20240523_4TicksRevFeaturesCropped_ALL_1-99"
file_namePCA = "20190303_20240523_4TicksRevFeaturesPCA6"

# Demander à l'utilisateur de choisir le fichier
user_input = 'rrr'

if user_input.lower() == 'p':
    file_full_path = os.path.join(file_path, f'{file_namePCA}.csv')
    num_pca_columns = int(file_namePCA.split('PCA')[-1])
else:
    file_full_path = os.path.join(file_path, f'{file_name}.csv')

# Data Preparation
data = pd.read_csv(file_full_path, sep=';')

if user_input.lower() == 'p':
    pca_columns = [f'PCA_{i + 1}' for i in range(num_pca_columns)]
    X = data[pca_columns].values
else:
    X = data[[
        # 'deltaTimestamp',
        'diffPriceClosePoc_0_0',
        'diffPriceCloseHigh_0_0',
        'diffPriceCloseLow_0_0',
       #  'ratioVolCandleMeanx',
         'ratioVolPocVolCandle',
      #   'ratioPocDeltaPocVol',
         'ratioVolBlw',
         'ratioVolAbv',
        # 'ratioDeltaBlw',
        # 'ratioDeltaAbv',
        'imbFactorAskL',
        'imbFactorBidH',
        #'bidVolumeAtBarLow',
        #'askVolumeAtBarLow',
        #'bidVolumeAtBarHigh',
        #'askVolumeAtBarHigh',
         #'diffPocPrice_0_1',
         'diffHighPrice_0_1',
         'diffLowPrice_0_1',
         'diffVolCandle_0_1',
         'diffVolDelta_0_1',
        # 'diffPriceCloseVWAP',
        # 'diffPriceCloseVWAPsd3Top',
        # 'diffPriceCloseVWAPsd3Bot',
        'tradeDir'
    ]].values

y = data['tradeResult'].values

# Normalize input data
scaler = MinMaxScaler()
scaler = StandardScaler()
if user_input.lower() != 'p':
    X = scaler.fit_transform(X)
    print("X was scaled")

# Split data into input sequences and labels
def create_subsequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length + 1):
        seq_X = X[i:i + seq_length]
        if len(seq_X) == seq_length:
            X_seq.append(seq_X)
            y_seq.append(y[i + seq_length - 1])
    return np.array(X_seq), np.array(y_seq)

X_sequences, y_sequences = create_subsequences(X, y, SEQUENCE_LENGTH)
print(f"Nombre de lignes dans X_sequences : {X_sequences.shape[0]}")

# Convert class labels from (-1, 0, 1) to (0, 1, 2)
y_sequences = y_sequences + 1

# Reshape input sequences for XGBoost
X_sequences_reshaped = X_sequences.reshape(X_sequences.shape[0], -1)

# Split data into train, validation, and test sets
train_val_size = int(0.9 * len(X_sequences_reshaped))
X_train_val, X_test = X_sequences_reshaped[:train_val_size], X_sequences_reshaped[train_val_size:]
y_train_val, y_test = y_sequences[:train_val_size], y_sequences[train_val_size:]

# Calculer les poids de classe équilibrés
classes = np.array([0, 1, 2])  # Convertir en tableau NumPy
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train_val)
class_weights = dict(zip(classes, class_weights))

class XGBoostWalkForward(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(self, X, y, sample_weight=None):
        self.model = XGBClassifier(objective='multi:softprob',
                                   n_estimators=self.n_estimators,
                                   max_depth=self.max_depth,
                                   learning_rate=self.learning_rate)
        self.model.fit(X, y, sample_weight=sample_weight)
        self.classes_ = np.unique(y)  # Store the unique class labels
        return self

    def predict(self, X):
        return self.model.predict(X)

# Définir la grille des hyperparamètres
param_grid = {
   # 'n_estimators': [100, 200],
    #'max_depth': [3, 4],
    'learning_rate': [0.01, 0.1]
}

# Walk-Forward Validation avec recherche en grille des hyperparamètres
def walk_forward_validation_with_grid_search(X_train_val, y_train_val, anchored=True):
    total_val_accuracy = 0.0
    total_val_precision = np.zeros(3)  # Precision for each class
    total_val_recall = np.zeros(3)  # Recall for each class
    total_val_f1 = np.zeros(3)  # F1 score for each class

    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []

    start_idx = 0
    end_idx = TRAIN_WINDOW_SIZE + TEST_HORIZON

    best_avg_val_score = -1
    best_avg_model = None

    while end_idx <= X_train_val.shape[0]:
        print(f'end_idx: {end_idx} || X_train_val.shape[0]:{X_train_val.shape[0]} ')

        # Split data for the current window
        X_train = X_train_val[start_idx:end_idx]
        y_train = y_train_val[start_idx:end_idx]
        X_val = X_train_val[end_idx:end_idx + TEST_HORIZON]
        y_val = y_train_val[end_idx:end_idx + TEST_HORIZON]

        # Créer un vecteur de poids d'instance pour l'entraînement
        instance_weights = [class_weights[y] for y in y_train]
        instance_weights = np.array(instance_weights)

        # Configurer la recherche en grille
        grid_search = GridSearchCV(estimator=XGBoostWalkForward(), param_grid=param_grid,
                                   scoring='accuracy', cv=3, verbose=1)

        # Lancer la recherche en grille
        grid_search.fit(X_train, y_train, sample_weight=instance_weights)

        # Récupérer le meilleur modèle
        best_model = grid_search.best_estimator_

        # Vérifier si ce modèle a le meilleur score moyen de validation
        avg_val_score = np.mean(grid_search.cv_results_['mean_test_score'])
        if avg_val_score > best_avg_val_score:
            best_avg_val_score = avg_val_score
            best_avg_model = best_model

        # Check if there are any samples in the validation set
        if X_val.shape[0] > 0:
            # Make predictions on the validation set using the best model
            val_preds = best_model.predict(X_val)

            # Calculate validation metrics
            val_accuracy = accuracy_score(y_val, val_preds)
            val_precision = precision_score(y_val, val_preds, average=None, zero_division=0)  # Precision for each class
            val_recall = recall_score(y_val, val_preds, average=None, zero_division=0)  # Recall for each class
            val_f1 = f1_score(y_val, val_preds, average=None, zero_division=0)  # F1 score for each class

            # Check for absent predictions and log warnings
            for i, label in enumerate([0, 1, 2]):
                if np.isnan(val_precision[i]):
                    print(f"Warning: No predicted samples for class {label} in validation set")
                    val_precision[i] = 0
                if np.isnan(val_recall[i]):
                    val_recall[i] = 0
                if np.isnan(val_f1[i]):
                    val_f1[i] = 0

            # Store validation metrics for this loop
            val_accuracies.append(val_accuracy)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)
            val_f1_scores.append(val_f1)

            # Update total validation metrics
            total_val_accuracy += val_accuracy
            total_val_precision += val_precision
            total_val_recall += val_recall
            total_val_f1 += val_f1

            # Afficher les métriques de performance pour cette boucle
            print(f"Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Precision (0, 1, 2): {val_precision[0]:.4f}, {val_precision[1]:.4f}, {val_precision[2]:.4f}")
            print(f"Validation Recall (0, 1, 2): {val_recall[0]:.4f}, {val_recall[1]:.4f}, {val_recall[2]:.4f}")
            print(f"Validation F1 Score (0, 1, 2): {val_f1[0]:.4f}, {val_f1[1]:.4f}, {val_f1[2]:.4f}")

            # Afficher les statistiques pour chaque classe
            for i, label in enumerate([0, 1, 2]):
                num_predictions = np.sum(val_preds == label)
                num_targets = np.sum(y_val == label)
                print(f"Classe {label}: {num_predictions} prédictions, {num_targets} cibles")

            # Afficher les avertissements pour les classes sans prédictions
            for i, label in enumerate([0, 1, 2]):
                if val_precision[i] == 0:
                    print(f"\033[93mWarning: No predicted samples for class {label} in validation set\033[0m")
                else:
                    print(f"\033[92mSamples for all classes present.\033[0m")

        else:
            print(f"\033[93mWarning: No samples in validation set\033[0m")

        if not anchored:
            # Move the window forward for non-anchored walk-forward validation
            start_idx += STEP_SIZE
        end_idx += STEP_SIZE

    # Calculate average validation metrics
    num_windows = (X_train_val.shape[0] - TEST_HORIZON) // STEP_SIZE + 1
    avg_val_accuracy = total_val_accuracy / num_windows
    avg_val_precision = total_val_precision / num_windows
    avg_val_recall = total_val_recall / num_windows
    avg_val_f1 = total_val_f1 / num_windows

    return avg_val_accuracy, avg_val_precision, avg_val_recall, avg_val_f1, val_accuracies, val_precisions, val_recalls, val_f1_scores, grid_search, best_avg_model


# Demander à l'utilisateur de choisir entre la validation walk-forward ancrée et non ancrée
validation_type = input("Entrez 'a' pour la validation walk-forward ancrée ou une autre touche pour la validation non ancrée : ")
anchored = validation_type.lower() == 'a'

# Perform walk-forward validation with grid search
avg_val_accuracy, avg_val_precision, avg_val_recall, avg_val_f1, val_accuracies, val_precisions, val_recalls, val_f1_scores, grid_search, best_avg_model = walk_forward_validation_with_grid_search(X_train_val, y_train_val, anchored=anchored)



# Final evaluation on the test set
instance_weights = [class_weights[y] for y in y_train_val]
instance_weights = np.array(instance_weights)

# Configurer la recherche en grille pour l'évaluation finale
grid_search_final = GridSearchCV(estimator=XGBoostWalkForward(), param_grid=param_grid,
                                 scoring='accuracy', cv=3, verbose=1)

# Lancer la recherche en grille pour l'évaluation finale
grid_search_final.fit(X_train_val, y_train_val, sample_weight=instance_weights)

# Récupérer le meilleur modèle pour l'évaluation finale
best_model_final = grid_search_final.best_estimator_

print("Best Parameters for Final Best Model:")
print(grid_search_final.best_params_)

# Predictions and metrics for the final best model
test_preds_final = best_model_final.predict(X_test)
test_accuracy_final = accuracy_score(y_test, test_preds_final)
test_precision_final = precision_score(y_test, test_preds_final, average=None, zero_division=0)
test_recall_final = recall_score(y_test, test_preds_final, average=None, zero_division=0)
test_f1_final = f1_score(y_test, test_preds_final, average=None, zero_division=0)

print("Final Best Model Performance:")
print(f'Test Accuracy: {test_accuracy_final:.4f}')
print(f'Test Precision (0, 1, 2): {test_precision_final[0]:.4f}, {test_precision_final[1]:.4f}, {test_precision_final[2]:.4f}')
print(f'Test Recall (0, 1, 2): {test_recall_final[0]:.4f}, {test_recall_final[1]:.4f}, {test_recall_final[2]:.4f}')
print(f"Test F1 Score: 0: {test_f1_final[0]:.4f}, 1: {test_f1_final[1]:.4f}, 2: {test_f1_final[2]:.4f}")

print("\nBest Parameters for Best Model Based on Average Validation Scores:")
print(best_avg_model.get_params())

# Predictions and metrics for the best model based on average validation scores
test_preds_avg = best_avg_model.predict(X_test)
test_accuracy_avg = accuracy_score(y_test, test_preds_avg)
test_precision_avg = precision_score(y_test, test_preds_avg, average=None, zero_division=0)
test_recall_avg = recall_score(y_test, test_preds_avg, average=None, zero_division=0)
test_f1_avg = f1_score(y_test, test_preds_avg, average=None, zero_division=0)

print("Best Model Based on Average Validation Scores Performance:")
print(f'Test Accuracy: {test_accuracy_avg:.4f}')
print(f'Test Precision (0, 1, 2): {test_precision_avg[0]:.4f}, {test_precision_avg[1]:.4f}, {test_precision_avg[2]:.4f}')
print(f'Test Recall (0, 1, 2): {test_recall_avg[0]:.4f}, {test_recall_avg[1]:.4f}, {test_recall_avg[2]:.4f}')
print(f"Test F1 Score: 0: {test_f1_avg[0]:.4f}, 1: {test_f1_avg[1]:.4f}, 2: {test_f1_avg[2]:.4f}")