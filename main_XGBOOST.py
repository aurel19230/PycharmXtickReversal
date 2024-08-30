import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
RANDOM_SEED = 10
np.random.seed(RANDOM_SEED)

# Hyperparameters
SEQUENCE_LENGTH = 3
TRAIN_WINDOW_SIZE = 80000
TEST_HORIZON = 12000
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

# Define the XGBoost model
model = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.01, random_state=RANDOM_SEED, objective='multi:softprob')

# Walk-Forward Validation
# Walk-Forward Validation
# Walk-Forward Validation
def walk_forward_validation(X_train_val, y_train_val, model, anchored=True):
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
    # Calculate the total number of loops
    total_loops = (X_train_val.shape[0] - TEST_HORIZON) // STEP_SIZE + 1
    current_loop = 0

    while end_idx <= X_train_val.shape[0]:
        print(f'end_idx: {end_idx} || X_train_val.shape[0]:{X_train_val.shape[0]} ')
        current_loop += 1
        print(f"Loop {current_loop}/{total_loops}")
        # Split data for the current window
        X_train = X_train_val[start_idx:end_idx]
        y_train = y_train_val[start_idx:end_idx]
        X_val = X_train_val[end_idx:end_idx + TEST_HORIZON]
        y_val = y_train_val[end_idx:end_idx + TEST_HORIZON]

        # Créer un vecteur de poids d'instance pour l'entraînement
        instance_weights = [class_weights[y] for y in y_train]
        instance_weights = np.array(instance_weights)

        # Train the XGBoost model
        model.fit(X_train, y_train, sample_weight=instance_weights)

        # Check if there are any samples in the validation set
        if X_val.shape[0] > 0:
            # Make predictions on the validation set
            val_preds = model.predict(X_val)

            # Calculate validation metrics
            val_accuracy = accuracy_score(y_val, val_preds)
            val_precision = precision_score(y_val, val_preds, average=None, zero_division=0)  # Precision for each class
            val_recall = recall_score(y_val, val_preds, average=None, zero_division=0)  # Recall for each class
            val_f1 = f1_score(y_val, val_preds, average=None, zero_division=0)  # F1 score for each class

            # Check for absent predictions and log warnings
            for i, label in enumerate([0, 1, 2]):
                if np.isnan(val_precision[i]):
                    print(f"Warning: No predicted samples for class {label} in validation set during loop {current_loop}/{total_loops}")
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
            print(f"Loop {current_loop}/{total_loops} Validation Accuracy: {val_accuracy:.4f}")
            print(f"Loop {current_loop}/{total_loops} Validation Precision (0, 1, 2): {val_precision[0]:.4f}, {val_precision[1]:.4f}, {val_precision[2]:.4f}")
            print(f"Loop {current_loop}/{total_loops} Validation Recall (0, 1, 2): {val_recall[0]:.4f}, {val_recall[1]:.4f}, {val_recall[2]:.4f}")
            print(f"Loop {current_loop}/{total_loops} Validation F1 Score (0, 1, 2): {val_f1[0]:.4f}, {val_f1[1]:.4f}, {val_f1[2]:.4f}")

            # Afficher les statistiques pour chaque classe
         #   print("Statistiques pour cette boucle :")
            for i, label in enumerate([0, 1, 2]):
                num_predictions = np.sum(val_preds == label)
                num_targets = np.sum(y_val == label)
                print(f"Classe {label}: {num_predictions} prédictions, {num_targets} cibles")

            # Afficher les avertissements pour les classes sans prédictions
            for i, label in enumerate([0, 1, 2]):
                if val_precision[i] == 0:
                    print(f"\033[93mWarning: No predicted samples for class {label} in validation set during loop {current_loop}/{total_loops}\033[0m")
                else:
                    print(f"\033[92mSamples for all classes present.\033[0m")

        else:
            print(f"\033[93mWarning: No samples in validation set during loop {current_loop}/{total_loops}\033[0m")

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

    return avg_val_accuracy, avg_val_precision, avg_val_recall, avg_val_f1, val_accuracies, val_precisions, val_recalls, val_f1_scores

# Demander à l'utilisateur de choisir entre la validation walk-forward ancrée et non ancrée
validation_type = input("Entrez 'a' pour la validation walk-forward ancrée ou une autre touche pour la validation non ancrée : ")
anchored = validation_type.lower() == 'a'

# Perform walk-forward validation
avg_val_accuracy, avg_val_precision, avg_val_recall, avg_val_f1, val_accuracies, val_precisions, val_recalls, val_f1_scores = walk_forward_validation(X_train_val, y_train_val, model, anchored=anchored)

print(f'Average Validation Accuracy: {avg_val_accuracy:.4f}')
print(f'Average Validation Precision (0, 1, 2): {avg_val_precision[0]:.4f}, {avg_val_precision[1]:.4f}, {avg_val_precision[2]:.4f}')
print(f'Average Validation Recall (0, 1, 2): {avg_val_recall[0]:.4f}, {avg_val_recall[1]:.4f}, {avg_val_recall[2]:.4f}')
print(f'Average Validation F1 Score (0, 1, 2): {avg_val_f1[0]:.4f}, {avg_val_f1[1]:.4f}, {avg_val_f1[2]:.4f}')

# Final evaluation on the test set
instance_weights = [class_weights[y] for y in y_train_val]
instance_weights = np.array(instance_weights)
model.fit(X_train_val, y_train_val, sample_weight=instance_weights)
test_preds = model.predict(X_test)

test_accuracy = accuracy_score(y_test, test_preds)
test_precision = precision_score(y_test, test_preds, average=None, zero_division=0)
test_recall = recall_score(y_test, test_preds, average=None, zero_division=0)
test_f1 = f1_score(y_test, test_preds, average=None, zero_division=0)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Test Precision (0, 1, 2): {test_precision[0]:.4f}, {test_precision[1]:.4f}, {test_precision[2]:.4f}')
print(f'Test Recall (0, 1, 2): {test_recall[0]:.4f}, {test_recall[1]:.4f}, {test_recall[2]:.4f}')

# Créer un dictionnaire associant les classes aux scores F1
f1_scores = dict(zip([0, 1, 2], test_f1))

# Afficher les scores F1 pour chaque classe
print(f"Test F1 Score: 0: {f1_scores[0]:.4f}, 1: {f1_scores[1]:.4f}, 2: {f1_scores[2]:.4f}")


import matplotlib.pyplot as plt

# Afficher l'évolution de la précision de validation pour chaque classe
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i, ax in enumerate(axs):
    ax.plot(range(1, len(val_precisions) + 1), [val_precision[i] for val_precision in val_precisions])
    ax.set_title(f"Précision de validation pour la classe {i}")
    ax.set_xlabel("Boucle de validation")
    ax.set_ylabel("Précision")
plt.show()