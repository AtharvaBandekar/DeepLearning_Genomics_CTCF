# Import required functions
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
import logomaker
import sys

# Add the parent directory to the system path to import the model definition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.model_training import DNAClassifier

# Define Motif visualization function
def visualize_motifs(model, results_dir):
    print("\n--- Starting Model Interpretation: Visualizing Motifs ---")
    
    motif_output_dir = os.path.join(results_dir, 'motifs')
    os.makedirs(motif_output_dir, exist_ok=True)

    conv1_weights = model.conv1.weight.data.cpu().numpy()

    num_filters = conv1_weights.shape[0]
    kernel_size = conv1_weights.shape[2]
    
    # Define nucleotide mapping
    nucleotides = ['A', 'C', 'G', 'T']

    # Iterate through each filter to generate a motif logo
    for i in range(num_filters):
        filter_weights = conv1_weights[i, :, :]
        
        # Scaling:
        motif_matrix = np.maximum(0, filter_weights)
        motif_matrix = motif_matrix / (motif_matrix.sum(axis=0) + 1e-9)

        # Columns correspond to positions, index corresponds to nucleotides
        import pandas as pd
        motif_df = pd.DataFrame(motif_matrix.T, columns=nucleotides)

        # Create the motif logo
        plt.figure(figsize=(kernel_size * 0.8, 3))
        lm = logomaker.Logo(motif_df,color_scheme='classic', vpad=.1,width=.8)
        for spine in lm.ax.spines.values():
            spine.set_visible(False)
        lm.ax.set_xticks([])
        lm.ax.set_yticks([])
        lm.ax.set_ylabel('')
        lm.ax.set_title(f'Filter {i+1} Motif')
        
        motif_filename = os.path.join(motif_output_dir, f'motif_{i+1}.png')
        plt.savefig(motif_filename, bbox_inches='tight')
        plt.close()

        if i < 5:
            print(f"Generated motif logo for Filter {i+1}: {motif_filename}")
        
    print(f"Generated {num_filters} motif logos in {motif_output_dir}")
    print("--- Model Interpretation Complete ---")


# Main execution script
if __name__ == '__main__':
    MODELS_DIR = os.path.join(os.path.dirname(__file__), '../data/models')
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../results')
    
    MODEL_PATH = os.path.join(MODELS_DIR, 'dna_classifier_ctcf.pth')

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model with the same architecture parameters used during training
    SEQUENCE_LENGTH = 200
    NUM_FILTERS = 128
    KERNEL_SIZE = 10
    DROPOUT_RATE = 0.25

    model = DNAClassifier(sequence_length=SEQUENCE_LENGTH, num_filters=NUM_FILTERS, kernel_size=KERNEL_SIZE, dropout_rate=DROPOUT_RATE)

    # Load the trained model weights
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print(f"Successfully loaded model from {MODEL_PATH}")
    else:
        print(f"Error: Model not found at {MODEL_PATH}. Please ensure model_training.py was run successfully.")
        sys.exit(1)

    # Visualize the motifs
    visualize_motifs(model, RESULTS_DIR)