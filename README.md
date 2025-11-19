# Medical SigLIP Fine-Tuning Project

This repository contains code for fine-tuning the **Google SigLIP (Sigmoid Loss for Language Image Pre-training)** model on a medical dermatology dataset. The project has been significantly refactored to handle class imbalance, multi-label data, and provide robust evaluation metrics.

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
