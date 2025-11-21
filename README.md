# Medical SigLIP Fine-Tuning Project

This repository contains code for fine-tuning the **Google SigLIP (Sigmoid Loss for Language Image Pre-training)** model on a medical dermatology dataset. The project has been significantly refactored to handle class imbalance, multi-label data, and provide robust evaluation metrics.


## Contrastive Loss Functions

Contrastive learning aims to learn representations by pulling similar (positive) pairs closer together in an embedding space while pushing dissimilar (negative) pairs apart. The specific mathematical implementation of this goal can vary.

### Loss Function Comparison

| Model | Loss Function | Description |
|-------|---------------|-------------|
| **CLIP** | Symmetric Cross-Entropy (Softmax-based InfoNCE loss) | This loss function operates over all possible image-text pairings within a batch, requiring a "global view" for normalization via the softmax function. It treats the task as an N-class classification problem where the goal is to identify the single correct pairing among N × N possibilities within the batch. |
| **SigLIP** | Pairwise Sigmoid Loss | SigLIP replaces the softmax with a simpler sigmoid-based loss that treats each image-text pair independently as a binary classification problem (match or no match). This eliminates the need for global normalization across the entire batch. |

### Key Differences

The primary distinction is the normalization process:

- **CLIP's softmax loss** requires that the sum of probabilities for all items in the batch equals one, creating a relative comparison where increasing the probability of a positive pair must decrease the probability of a negative pair.

- **SigLIP's sigmoid loss** processes each pair's similarity score in isolation, which simplifies distributed training and allows for better scaling with various batch sizes.


# 🌟 Advanced SigLIP Fine-Tuning: Geometric Loss Layering

This document outlines the final, production-ready architecture for fine-tuning the SigLIP (Sigmoid Loss for Language Image Pre-training) model on a complex, fine-grained medical dataset. This approach incorporates multi-stage loss layering, data stratification, and aggressive hard negative mining, as recommended for building a world-class discriminative model.

---

## 1. ⚙️ Key Architectural Improvements (The Solution)

The previous architectural weaknesses (model collapse, zero-confidence prediction, and ignoring domain structure) have been solved by implementing a multi-stage system:

### Feature | Fix Implemented | Rationale
| Feature | Fix | Rationale |
|--------|------|------------|
| **Loss & Stability** | Weighted Sigmoid Loss | Cures the "lazy model" problem by heavily penalizing false negatives (missing the positive match). |
| **Numerical Stability** | Explicit float32 Casting | Prevents numerical instability (NaN/Inf generation) common when using complex losses in float16 precision. |
| **Data Quality** | Stratified Sampling & Augmentation | Ensures rare classes are represented across all splits and strengthens the model's invariance to external noise. |
| **Metric Calculation** | `prediction_step` Override | Enables robust calculation of R@1/R@5 Accuracy, Loss, Precision, and F1-score simultaneously. |

---
# Custom Loss: Training Layers and Their Roles

| Layer | Type | Role in Training |
|-------|------|------------------|
| **1. ICD Dimensionality** ($\mathcal{L}_{\text{ICD}}$) | Metric/Geometric | **Teaches Domain Structure.** Forces the model's embedding space to respect the external medical hierarchy (cousins are closer than strangers). |
| **2. Contrastive Loss** ($\mathcal{L}_{\text{Contr}}$) | Task Alignment | **Establishes Retrieval Baseline.** Pulls the correct image-text pairs together. |
| **3. Wasserstein Distance** ($\mathcal{L}_{\text{Wass}}$) | Distributional | **Ensures Stability.** Minimizes the "cost" (Earth Mover's Distance) of transforming the image embedding distribution into the text embedding distribution, promoting smoother, better-separated clusters. |
| **4. Final Contrastive Loss** ($\mathcal{L}_{\text{Contr}}$) | Refinement | **Ensures the final, geometrically optimized embeddings still maintain a clear, high similarity score for the correct answer.** |
---
### **Loss Function Layering Formula**

## 1. The Layering Formula

The debug test uses the following formula to combine the three active loss terms:

$$\mathcal{L}_{\text{Total}} = (1 - \alpha) \cdot \mathcal{L}_{\text{Sigmoid}} + \alpha \cdot \mathcal{L}_{\text{ICD}} + \beta \cdot \mathcal{L}_{\text{Wasserstein}}$$

In this specific debug test, the assumed modulation coefficients are the typical starting points:

- **Alpha** ($\alpha$): Set to $0.5$. This means the Sigmoid and ICD layers contribute equally.
- **Beta** ($\beta$): Set to a small value, $0.1$, to prevent the distributional term from overwhelming the structural terms (since the absolute values of $\mathcal{L}_{\text{Sigmoid}}$ are much larger).

## 2. Calculation Breakdown

Using the output values and the assumed coefficients:

```
DEBUG: Expected Sigmoid Loss (L_Sigmoid): 15.000000
DEBUG: Expected ICD Loss (L_ICD): 0.250669
DEBUG: Expected Wasserstein Loss (L_Wass): 0.001892
DEBUG: Expected Total Layered Loss: 7.625524
```


| Term | Value (Observed) | Coefficient | Weighted Contribution |
|------|------------------|-------------|----------------------|
| Sigmoid Loss ($\mathcal{L}_{\text{Sigmoid}}$) | $15.000000$ | $(1 - \alpha) = 0.5$ | $0.5 \times 15.000000 = \mathbf{7.500000}$ |
| ICD Loss ($\mathcal{L}_{\text{ICD}}$) | $0.250669$ | $\alpha = 0.5$ | $0.5 \times 0.250669 = \mathbf{0.125335}$ |
| Wasserstein Loss ($\mathcal{L}_{\text{Wass}}$) | $0.001892$ | $\beta = 0.1$ | $0.1 \times 0.001892 = \mathbf{0.000189}$ |
| **Total** | | | $\mathbf{7.625524}$ |


### Final Calculation

$$\mathcal{L}_{\text{Total}} = 7.500000 + 0.125335 + 0.000189 = \mathbf{7.625524}$$

---
### The Modulated Loss Calculation Flow

| Step | Action within `training_step` | Value of Coefficient |
|------|-------------------------------|---------------------|
| **0 - 1000** | $\alpha$ ramp ($\mathcal{L}_{\text{ICD}}$) | $\alpha$ goes from $0.0 \to 1.0$; $\beta, \gamma$ stay at $0.0$. |
| **1001 - 2000** | $\beta$ ramp ($\mathcal{L}_{\text{Wass}}$) | $\beta$ goes from $0.0 \to 1.0$; $\alpha$ stays fixed at $1.0$. |
| **2001 - 2500** | $\gamma$ ramp ($\mathcal{L}_{\text{HardNeg}}$) | $\gamma$ goes from $0.0 \to 1.0$; $\alpha, \beta$ stay fixed at $1.0$. |
---

## 2. 🧬 Phase 1: Geometric Loss Layering Strategy

To achieve the best possible performance for fine-grained discrimination, the final architecture adopts a multi-term loss function,  
$\mathcal{L}_{\text{Total}}$, layered over several epochs.  
This forces the model to learn the fundamental medical structure (ICD geometry) before specializing in specific visual features.

### **Loss Definition**

$\mathcal{L}_{\text{Total}} = (1 - \alpha) \cdot \mathcal{L}_{\text{Sigmoid}} + \alpha \cdot \mathcal{L}_{\text{ICD}} + \text{[Optional } \beta \cdot \mathcal{L}_{\text{Wass}} + \dots]$

### Implementation Roadmap
|   Step    | Loss Term & Focus | Action & Status |
|------|-------------------|---------|
| **0. Preprocessing** | ICD Geometry | REQUIRED: Map all 66 conditions to ICD-10 codes and calculate the numerical Dissimilarity Matrix (𝑫) based on code proximity. | |
| **1. Baseline Alignment** | \(\mathcal{L}_{\text{Sigmoid}}\) (Weighted) | Train initially with only the stable, weighted diagonal Sigmoid Loss. (Current working code uses this). | |
| **2. Structure Layering** | \(\mathcal{L}_{\text{ICD}}\) (Geometric) | Introduce the ICD Loss term. This pushes embeddings based on their structural relationship (e.g., Eczema must be closer to Dermatitis than to Melanoma). | |
| **3. Modulation (α)** | Layering Scheduler | Implement a custom scheduler that slowly ramps the coefficient α from 0 → 1. This allows the model to absorb the structural knowledge without catastrophic forgetting. | |

---

## 3. 🎯 Final Evaluation (R@5 Accuracy)

For a diagnostic tool, providing the most relevant list of potential diagnoses is more useful than a single guess.  
The evaluation logic has been configured for this:

### **Top-K Metric**
Since medical image diagnosis are always differential disagnois (more than 1 options or macthes to consider), so that a doctor can make the judgement call.
Accuracy is measured using **Top-K Retrieval Accuracy** (R@K, default K=5).  
A prediction is counted as a **SUCCESS** if the correct condition is ranked **anywhere in the top 5** predicted candidates.

### **Retrieval Task**
The custom `test_retrieval` function performs a **1-shot retrieval test** against 9 random distractors, providing confidence scores (Sigmoid) for deeper debugging.

---

## 4. 💻 Project Configuration Summary

| Setting | Value | Rationale |
|---------|--------|-----------|
| **Model** | `google/siglip-base-patch16-224` | Base model for fine-tuning. |
| **Loss** | Weighted Sigmoid BCE | Robust loss function for multi-label, imbalanced data. |
| **LoRA Targets** | Q/V/K/Out/FC1/FC2 | Maximized capacity; includes FFNs for deep domain adaptation. |
| **LR Scheduler** | cosine | Promotes stable convergence and better final minimums. |
| **Data Filter** | Filtered (0-label rows removed) | Ensures clean training signal. |


## 🚀 Recent Changes & Improvements

### 1. Data Balancing & Stratification
To address the severe class imbalance (e.g., 700+ Eczema samples vs. <5 Rare Condition samples), we completely overhauled the data splitting logic:
* **Stratified Split:** Replaced simple random splitting with a "Stratified Split" strategy. This ensures that rare conditions are explicitly distributed across Train, Validation, and Test sets so they are not missed during evaluation.
* **Oversampling:** Implemented a `balance_training_data` function that oversamples rare classes in the training set. Conditions with fewer than ~30 samples are duplicated to ensure the model sees them frequently enough to learn features.
* **Full Data Utilization:** The script now loads the entire dataset first before splitting, ensuring no data is arbitrarily excluded.

### 2. Enhanced Text Supervision (Verbose Labels)
We shifted from using single-word labels (e.g., "Eczema") to verbose, descriptive text prompts.
* **Why:** SigLIP is trained on natural language sentences. Providing semantic context produces better embeddings than isolated keywords.
* **Implementation:** The dataset class (`FineGrainedContrastiveDataset`) now prioritizes the `description` column from the metadata.
* **Templates:** If a description is missing, we use structured templates like:
    * *"A dermatological image showing {condition}."*
    * *"Clinical presentation of {condition}: {description}"*

### 3. Robust Loss Function (Sigmoid vs. Softmax)
We explicitly implemented **Sigmoid Loss** (`binary_cross_entropy_with_logits`) instead of the standard Contrastive (Softmax) loss.
* **Reasoning:** The dataset is multi-label (images can validly match multiple related conditions). Standard contrastive loss forces mutual exclusivity (if Image A matches Text A, it *cannot* match Text B), which confuses the model when Text A and Text B are medically similar.
* **Effect:** Sigmoid loss treats every image-text pair as an independent binary classification problem, allowing for "softer" boundaries and better handling of label noise.

### 4. Custom Trainer & Metrics
We replaced the default Hugging Face training loop with a `CustomTrainer` to handle SigLIP's specific needs:
* **Metrics Calculation:** Added a standalone `compute_metrics` function that calculates **Accuracy, Precision, Recall, and F1-score** (Macro-averaged).
* **`prediction_step` Override:** Custom implementation to manually compute loss *and* return raw logits during evaluation, ensuring that metrics are reported correctly (fixing the "N/A" metrics bug).
* **Tuple Handling:** Fixed critical bugs where the trainer returned tuples of logits `(image_logits, text_logits)`, causing crashes in metric computation. The code now robustly extracts the image-to-text logits.

### 5. Specialized Retrieval Evaluation
Added a dedicated `test_retrieval` function to measure **Top-1 Retrieval Accuracy**.
* **Mechanism:** For a given test image, the model must select the correct condition description from a pool of 9 random distractors (other conditions).
* **Optimization:** The evaluation loop was optimized to encode the image *once* and compare it against all candidate texts via matrix multiplication, speeding up testing by ~10x compared to the naive approach.

---

## 🛠 Technical Implementation Details

### Model Architecture
* **Base Model:** `google/siglip-base-patch16-224`
* **LoRA (Low-Rank Adaptation):** Applied to `q_proj` and `v_proj` layers to fine-tune efficiently without destroying pre-trained knowledge.
* **Precision:** Training in `float16` (mixed precision) on CUDA devices.

### Critical Code Fixes
* **Image Shape Mismatch:** Added `.squeeze(0)` in the dataset `__getitem__` to prevent the `AutoProcessor` from adding an extra batch dimension that caused `ValueError` during training.
* **Attribute Errors:** Fixed `AttributeError: 'SigLipModel' object has no attribute 'device'` by dynamically retrieving the device from the input tensors (`logits.device`) rather than the model wrapper.
* **Dictionary Output:** Added checks to handle cases where the model returns a plain `dict` instead of a `ModelOutput` object.

### Configuration Summary
```python
# Key Training Arguments used
args = TrainingArguments(
    prediction_loss_only=False,  # CRITICAL: Must be False to compute metrics
    remove_unused_columns=False, # Keep pixel_values available
    evaluation_strategy="epoch", # Evaluate every epoch
    save_strategy="epoch",       # Save checkpoints every epoch
    lr_scheduler_type="cosine",
    load_best_model_at_end=True, # Keep the best model
)

Your current loss function uses torch.eye(batch_size), which forces the diagonal to be 1 and everything else to be 0. The Scenario: Image 0 is "Eczema". Image 5 is also "Eczema". The Problem: The loss function tells the model: "Image 0 matches Text 0" (Correct), but also "Image 0 MUST NOT match Text 5" (Wrong!). The Fix: Multi-Positive Sigmoid Loss
