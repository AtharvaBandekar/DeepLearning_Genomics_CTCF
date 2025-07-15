# Import required modules
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add the parent directory to the system path to import data_preparation.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.data_preparation import get_dataloaders 


# Define the model architecture
class DNAClassifier(nn.Module):
    def __init__(self, sequence_length=200, num_filters=128, kernel_size=10, dropout_rate=0.25):
        super(DNAClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=num_filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout = nn.Dropout(dropout_rate)

        conv_output_length = (sequence_length - kernel_size) + 1
        pooled_output_length = int(np.floor((conv_output_length - 3) / 3 + 1))

        self.fc1 = nn.Linear(num_filters * pooled_output_length, 64)
        self.fc2 = nn.Linear(64, 1)

# Define the forward pass sequence
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

# Defining the training loop method
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    print("Starting Model Training...")
    model.train()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device).view(-1, 1) # Ensure labels are (batch_size, 1)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Define the validation phase
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_predictions = []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device).view(-1, 1)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_val_labels.extend(labels.cpu().numpy())
                all_val_predictions.extend(outputs.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        val_preds_binary = (np.array(all_val_predictions) > 0.5).astype(int)
        val_accuracy = accuracy_score(all_val_labels, val_preds_binary)
        val_precision = precision_score(all_val_labels, val_preds_binary)
        val_recall = recall_score(all_val_labels, val_preds_binary)
        val_f1 = f1_score(all_val_labels, val_preds_binary)
        val_roc_auc = roc_auc_score(all_val_labels, all_val_predictions)

        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Acc: {val_accuracy:.4f}, "
              f"Val Prec: {val_precision:.4f}, "
              f"Val Rec: {val_recall:.4f}, "
              f"Val F1: {val_f1:.4f}, "
              f"Val ROC AUC: {val_roc_auc:.4f}")

        # Resave the model ONLY if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_save_path = os.path.join(os.path.dirname(__file__), '../data/models/dna_classifier_ctcf.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path} (Validation Loss improved to {best_val_loss:.4f})")
        
        model.train()
    print("Training Complete!")

# Define the evaluation loop wuth the test dataset
def evaluate_model(model, test_loader, device, results_dir):
    print("Starting Model Evaluation...")
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device).view(-1, 1)
            outputs = model(sequences)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    binary_predictions = (all_predictions > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, binary_predictions)
    precision = precision_score(all_labels, binary_predictions)
    recall = recall_score(all_labels, binary_predictions)
    f1 = f1_score(all_labels, binary_predictions)
    roc_auc = roc_auc_score(all_labels, all_predictions)

    print(f"Test Set Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    roc_curve_path = os.path.join(results_dir, 'roc_curve.png')
    plt.savefig(roc_curve_path)
    print(f"ROC Curve saved to {roc_curve_path}")
    plt.close()

    print("Model Evaluation Complete!")
    return accuracy, precision, recall, f1, roc_auc


# Main executable script
if __name__ == '__main__':
    # Define paths
    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
    POSITIVE_FASTA = os.path.join(PROCESSED_DATA_DIR, 'positive_sequences.fasta')
    NEGATIVE_FASTA = os.path.join(PROCESSED_DATA_DIR, 'negative_sequences.fasta')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '../data/models')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 200
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    NUM_FILTERS = 128
    KERNEL_SIZE = 10
    DROPOUT_RATE = 0.25

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        positive_fasta_path=POSITIVE_FASTA,
        negative_fasta_path=NEGATIVE_FASTA,
        batch_size=BATCH_SIZE,
        sequence_length=SEQUENCE_LENGTH
    )

    # Initialize model, loss function, and optimizer
    model = DNAClassifier(sequence_length=SEQUENCE_LENGTH, num_filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE).to(device)
    criterion = nn.BCELoss() # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=NUM_EPOCHS)

    # Load the best model
    model_path_to_load = os.path.join(MODELS_DIR, 'dna_classifier_ctcf.pth')
    if os.path.exists(model_path_to_load):
        model.load_state_dict(torch.load(model_path_to_load, map_location=device))
        print(f"Loaded best model from {model_path_to_load} for final evaluation.")
    else:
        print("Warning: Best model not found. Evaluating the last trained model state.")

    # Evaluate the model on test data
    test_accuracy, test_precision, test_recall, test_f1, test_roc_auc = evaluate_model(model, test_loader, device, RESULTS_DIR)

    # Print final metrics
    print(f"Final Test Metrics from model_training.py:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")