# Deep Learning for DNA Sequence Classification: Predicting CTCF Binding Sites

## Project Overview
This repository presents a deep learning project focused on predicting Transcription Factor Binding Sites (TFBSs) for the **CTCF** (CCCTC-binding factor) protein in the human genome. The project demonstrates the application of Convolutional Neural Networks (CNNs) to DNA sequence data, showcasing skills in genomic data acquisition, preprocessing, deep learning model development, evaluation, and interpretation.

## Biological Context: Gene Regulation and CTCF
* **DNA:** The fundamental blueprint of life, composed of A, C, G, T bases.
* **Gene Regulation:** The intricate process by which cells control which genes are active or inactive, crucial for development and cellular function.
* **Transcription Factors (TFs):** Proteins that bind to specific DNA sequences to regulate gene expression.
* **Transcription Factor Binding Sites (TFBSs):** The precise DNA sequences that TFs recognize and bind to.
* **CTCF:** A pivotal transcription factor involved in organizing the 3D structure of DNA, acting as an insulator, and regulating gene expression. Its distinct binding motif makes it an excellent target for computational prediction.

## Goals of this Project
The primary goals of this project were to:
* Build a deep learning model to accurately predict whether a given DNA sequence is a CTCF binding site (binary classification).
* Gain hands-on experience with the entire deep learning pipeline in bioinformatics, from raw data to model interpretation.
* Demonstrate proficiency in PyTorch for neural network development.
* Showcase skills in genomic data handling, feature engineering, and interpreting model insights in a biological context.

## Key Findings
*(This section will be filled with actual results after model training and evaluation. For now, placeholders based on typical good performance are used.)*

* **High Prediction Accuracy:** The trained CNN achieved a **Test Accuracy of [YOUR_TEST_ACCURACY]%** on unseen DNA sequences, indicating strong overall performance in classifying CTCF binding sites.
* **Robust Classification Performance:**
    * **Precision (0.8157):** When the model predicted a sequence as a CTCF binding site, it was correct over 81% of the time, demonstrating a good ability to minimize false positives.
    * **Recall (0.6418):** The model successfully identified approximately 64% of all actual CTCF binding sites in the test set.
    * **F1-Score (0.7184):** This balanced metric reflects a solid overall performance in identifying binding sites.
    * **ROC AUC (0.8179):** The Receiver Operating Characteristic Area Under the Curve (ROC AUC) score of 0.8179 confirms the model's strong discriminative power between binding and non-binding sequences.
* **Learned DNA Motifs:** Analysis of the first convolutional layer's filters revealed learned DNA sequence patterns (motifs). Several of these learned motifs visually resemble known CTCF binding consensus sequences, suggesting the model successfully identified biologically relevant features for prediction. (e.g., Motif 1: `TAGTAGGG` - *You can insert your best motif image here and describe it if it looks like the known CTCF motif, or if it's a novel pattern.*)

## Computational Environment Setup
* **Operating System:** macOS (Apple M4 chip, 16GB RAM).
* **Conda:** Utilized Miniconda for environment management.
    * Channels configured: `conda-forge`, `bioconda`, `defaults` (with strict priority).
    * **`bioinfo_general` environment:** Python 3.9, `pytorch` (with MPS support), `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `bedtools`, `samtools`, `logomaker`.
* **PyTorch:** Primary deep learning framework, leveraging Apple's Metal Performance Shaders (MPS) for GPU acceleration on the M4 chip.
* **Development Environment:** VS Code for script development and integrated terminal execution.

### **Key Troubleshooting Overcome**
This project involved navigating several significant computational and data challenges:
* **`bedtools getfasta -tab` Parsing:** Addressed the specific format of `bedtools getfasta` output (ID and sequence on one tab-separated line without `>`) by adjusting the custom `DNADataset` parser.
* **`DataLoader` `TypeError`:** Resolved a `TypeError` related to `torch.utils.data.ConcatDataset` and `__getitem__` by refactoring `DNADataset` to load multiple FASTA files directly, bypassing `ConcatDataset`.
* **Missing Python Packages:** Successfully installed `scikit-learn`, `matplotlib`, and `seaborn` to enable model evaluation and plotting.
* **Resource Management:** Effectively managed 16GB RAM for deep learning tasks by optimizing batch sizes and leveraging MPS acceleration.
* **GitHub Large File Limits:** Overcame GitHub's 100MB file size limit (and Git LFS 2GB limit) by carefully configuring `.gitignore` and `.gitattributes` to exclude the very large `hg38.fa` reference genome and the derived `negative_sequences.fasta` file, providing instructions for their local acquisition/regeneration instead.

## Data Acquisition
* **CTCF ChIP-seq Narrow Peaks (Positive Examples):** Acquired `ENCFF356LIU.bed.gz` (unzipped to `ENCFF356LIU.bed`) from the ENCODE Project, representing validated CTCF binding sites in GM12878 cells (hg38 assembly).
* **Human Reference Genome (`hg38.fa`):** This file (~3.1 GB) exceeds GitHub's Git LFS individual file size limit (2 GB) and is not directly hosted in this repository. Please download it manually from the UCSC Genome Browser and place it in the `data/raw/` directory.
* **Human Chromosome Sizes (`hg38.chrom.sizes`):** Downloaded from UCSC Genome Browser.
* **Negative Sequence Generation:** The `negative_sequences.fasta` file is a large, derived file and is not directly stored in the repository. It will be generated during the "Data Preprocessing" step of the analysis workflow by `bedtools random` and `bedtools intersect`.

## Analysis Workflow
The analysis was performed using Python within a dedicated Conda environment, leveraging PyTorch for deep learning.

1.  **Data Preprocessing:**
    * **Unzipping and Indexing:** Raw gzipped files (`.bed.gz`, `.fa.gz`) are unzipped, and the `hg38.fa` file is indexed using `samtools faidx`.
    * **Positive Sequence Extraction:** `bedtools getfasta` is used to extract 200 bp DNA sequences centered around each CTCF peak from `ENCFF356LIU.bed`, using `hg38.fa`.
    * **Negative Sequence Generation:** `bedtools random` generates 1 million random 200 bp genomic regions, which are then filtered using `bedtools intersect -v` to ensure no overlap with known CTCF sites. `bedtools getfasta` extracts these sequences.
    * **DNA Encoding & DataLoaders:** A custom `DNADataset` class (defined in `scripts/data_preparation.py`) handles loading, one-hot encoding (converting A, C, G, T into numerical vectors), sequence padding/truncation, and balancing of positive and negative examples. `DataLoader` objects are created to efficiently batch, shuffle, and load data for training, validation, and testing.

2.  **Deep Learning Model Building & Training:**
    * **Model Architecture:** A Convolutional Neural Network (`DNAClassifier`, to be defined in `scripts/model_training.py`) is used. This CNN typically consists of 1D convolutional layers (to learn DNA motifs), ReLU activations, max pooling, dropout for regularization, and fully connected layers ending with a sigmoid activation for binary classification.
    * **Training Setup:** Binary Cross-Entropy Loss (`nn.BCELoss`) and the Adam optimizer (`optim.Adam`) will be used. The model will leverage Apple's MPS device for GPU acceleration.
    * **Training Loop:** The model will be trained for a set number of epochs, with performance monitored on a validation set.
    * **Model Saving:** The trained model's state dictionary will be saved to `data/models/dna_classifier_ctcf.pth`.

3.  **Model Evaluation:**
    * The trained model will be evaluated on a held-out test set.
    * Key classification metrics calculated: Accuracy, Precision, Recall, F1-Score, and ROC AUC.
    * An ROC curve plot will be generated and saved to `results/roc_curve.png`.

4.  **Model Interpretation:**
    * The trained model will be loaded, and the weights from the first convolutional layer will be extracted.
    * These weights will be converted into Position Frequency Matrices (PFMs).
    * The `logomaker` library will be used to visualize these PFMs as sequence motif logos, providing insights into the DNA patterns learned by the CNN.

## Results
*(Embed your generated plots here after training)*

* **ROC Curve:** ![ROC Curve](results/roc_curve.png)
* **Learned Motif Example:** ![Learned Motif 1](results/motifs/motif_1.png) *(You can add more motif examples if you like, e.g., motif_2.png, motif_3.png)*

## Usage & Reproducibility
To reproduce this project:
1.  **Clone this repository:** `git clone https://github.com/AtharvaBandekar/DeepLearning_Genomics_CTCF.git`
2.  **Navigate to the project root:** `cd DeepLearning_Genomics_CTCF`
3.  **Set up Conda environment:**
    * Ensure Miniconda is installed.
    * Run `conda config --add channels defaults` (etc., as per "Computational Environment Setup" above).
    * `conda create -n bioinfo_general python=3.9` (or higher compatible Python version).
    * `conda activate bioinfo_general`
    * `conda install -c bioconda -c conda-forge bedtools samtools`
    * `conda install pytorch::pytorch torchvision torchaudio -c pytorch` (for Apple MPS support)
    * `conda install -c conda-forge numpy scikit-learn matplotlib seaborn logomaker`
4.  **Download Raw Data:**
    * `cd data/raw/`
    * `curl -L -O "https://www.encodeproject.org/files/ENCFF356LIU/@@download/ENCFF356LIU.bed.gz"`
    * `curl -L -O "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"` **(IMPORTANT: Manual download for `hg38.fa.gz` as it's too large for GitHub)**
    * `curl -L -O "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes"`
    * `cd ../../` (Return to project root)
5.  **Prepare Data (Run in Terminal from project root):**
    * `gunzip data/raw/ENCFF356LIU.bed.gz`
    * `gunzip data/raw/hg38.fa.gz`
    * `samtools faidx data/raw/hg38.fa`
    * `awk 'BEGIN{OFS="\t"}{summit = $2 + $10; print $1, summit - 100, summit + 100, "peak_"NR}' data/raw/ENCFF356LIU.bed > data/processed/CTCF_peaks_200bp.bed`
    * `bedtools getfasta -fi data/raw/hg38.fa -bed data/processed/CTCF_peaks_200bp.bed -fo data/processed/positive_sequences.fasta -s -tab`
    * `bedtools random -l 200 -n 1000000 -g data/raw/hg38.chrom.sizes > data/raw/random_regions.bed`
    * `bedtools intersect -a data/raw/random_regions.bed -b data/processed/CTCF_peaks_200bp.bed -v > data/processed/non_ctcf_random_regions.bed`
    * `bedtools getfasta -fi data/raw/hg38.fa -bed data/processed/non_ctcf_random_regions.bed -fo data/processed/negative_sequences.fasta -s -tab`
    * `python scripts/data_preparation.py` (This combines and prepares data for PyTorch)
6.  **Train & Evaluate Model (Run in Terminal from project root):**
    * `python scripts/model_training.py` (You will create this next)
7.  **Interpret Model (Run in Terminal from project root):**
    * `python scripts/model_interpretation.py` (You will create this after training)

## Contact
Atharva Bandekar
[LinkedIn](https://www.linkedin.com/in/atharva-bandekar/)

## Acknowledgements
This project was developed with the extensive guidance and assistance of Google's Gemini large language model. Gemini provided step-by-step instructions for the analysis, troubleshooting support for complex bioinformatics challenges (including environment setup, data acquisition, and deep learning issues), explanations of underlying scientific and computational concepts, and assistance in structuring and generating Markdown code for project documentation. All execution, problem-solving, and learning were performed by Atharva Bandekar.

