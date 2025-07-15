# Import required modules
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

# Create a PyTorch Dataset class to load and format the data
class DNADataset(Dataset):
    def __init__(self, fasta_file, sequence_length=200):
        self.fasta_file = fasta_file
        self.sequence_length = sequence_length
        self.sequences = []
        self.labels = []
        self._load_fasta()

    def _load_fasta(self):
        print(f"Loading sequences from {self.fasta_file}...")
        is_positive_file = "positive_sequences.fasta" in self.fasta_file

        with open(self.fasta_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    seq_id, sequence = parts[0], parts[1].upper()
                    if len(sequence) < self.sequence_length:
                        sequence = sequence + 'N' * (self.sequence_length - len(sequence))
                    else:
                        sequence = sequence[:self.sequence_length]

                    self.sequences.append(sequence)
                    self.labels.append(1 if is_positive_file else 0)
                else:
                    print(f"Warning: Skipping malformed line in {self.fasta_file}: {line.strip()}")
        print(f"Loaded {len(self.sequences)} sequences from {self.fasta_file}.")

    # Encode the sequences
    def one_hot_encode(self, sequence):
        mapping = {
            'A': [1., 0., 0., 0.],
            'C': [0., 1., 0., 0.],
            'G': [0., 0., 1., 0.],
            'T': [0., 0., 0., 1.],
            'N': [0., 0., 0., 0.]
        }
        encoded_sequence = np.zeros((self.sequence_length, 4), dtype=np.float32)
        for i, base in enumerate(sequence):
            if base in mapping:
                encoded_sequence[i] = mapping[base]
        return encoded_sequence

    # Required methods in this class used to list and reformat data
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoded_sequence = self.one_hot_encode(sequence)
        return torch.tensor(encoded_sequence, dtype=torch.float32).permute(1, 0), torch.tensor(label, dtype=torch.float32)

# Create a dataloaders class for training, validation, and test sets.
def get_dataloaders(positive_fasta_path, negative_fasta_path, batch_size=64, sequence_length=200, train_split=0.8, val_split=0.1, test_split=0.1):
    if not (os.path.exists(positive_fasta_path) and os.path.exists(negative_fasta_path)):
        raise FileNotFoundError(f"One or both FASTA files not found: {positive_fasta_path}, {negative_fasta_path}")

    positive_dataset = DNADataset(positive_fasta_path, sequence_length)
    negative_dataset = DNADataset(negative_fasta_path, sequence_length)

    min_samples = min(len(positive_dataset), len(negative_dataset))
    print(f"Balancing datasets: Using {min_samples} samples from each class.")

    positive_indices = torch.randperm(len(positive_dataset))[:min_samples]
    negative_indices = torch.randperm(len(negative_dataset))[:min_samples]

    balanced_positive_dataset = torch.utils.data.Subset(positive_dataset, positive_indices)
    balanced_negative_dataset = torch.utils.data.Subset(negative_dataset, negative_indices)

    full_dataset = torch.utils.data.ConcatDataset([balanced_positive_dataset, balanced_negative_dataset])
    print(f"Combined dataset size: {len(full_dataset)}")

    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size 

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 or 1, pin_memory=True)

    return train_loader, val_loader, test_loader

# Script for main execution
if __name__ == '__main__':

    PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/processed')
    POSITIVE_FASTA = os.path.join(PROCESSED_DATA_DIR, 'positive_sequences.fasta')
    NEGATIVE_FASTA = os.path.join(PROCESSED_DATA_DIR, 'negative_sequences.fasta')

    print("--- Testing data_preparation.py ---")
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            positive_fasta_path=POSITIVE_FASTA,
            negative_fasta_path=NEGATIVE_FASTA,
            batch_size=64,
            sequence_length=200
        )

        print(f"\nNumber of batches in Train Loader: {len(train_loader)}")
        print(f"Number of batches in Val Loader: {len(val_loader)}")
        print(f"Number of batches in Test Loader: {len(test_loader)}")

        print("\nChecking sample batch shape from Train Loader:")
        for i, (seqs, labels) in enumerate(train_loader):
            print(f"Batch {i+1}: Sequences shape: {seqs.shape}, Labels shape: {labels.shape}")
            if i == 0:
                assert seqs.shape[1] == 4, "Channels dimension is not 4"
                assert seqs.shape[2] == 200, "Sequence length dimension is not 200"
                print("Sample batch shapes are correct!")
                break

    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure you have run all data preprocessing steps.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("\n--- data_preparation.py test complete ---")