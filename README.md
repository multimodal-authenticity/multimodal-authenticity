# Title

## Overview

This repository contains the code and data used in the research paper *" Title  "*.

Our work builds and compares **16 multimodal machine learning (MMML) models**, including (1) unimodal, bimodal, and trimodal MulT-based models, (2) trimodal MulT-based models with text(t)/audio(a)/image(i) crossmodal, and (3) benchmark fusion models (e.g., LF-LSTM, EF-LSTM, TFN, LMF, MFN).

We introduce a multimodal authenticity measurement pipeline that captures **temporal, semantic, and crossmodal signals** from influencer content.

---

## Key Findings

- **Panel A (MulT Unimodal vs. Multimodal Comparison)**:  
  Incorporating **audio and image modalities** into the language-only MulT model significantly improves prediction performance. 

- **Panel B (Crossmodal Interaction Effects)**:  
  Crossmodal interaction mechanisms—whether **bimodal or trimodal (text-audio-image)**—significantly outperform non-interaction counterparts. This suggests that **inter-modality context modeling** plays a vital role in authenticity detection.

- **Panel C (Comparison with Benchmark Models)**:  
  Our full MulT model with **trimodal crossmodal integration (A11_MulT_tai_Crossmodal_TIA)** consistently outperforms baseline models (e.g., LF-LSTM, EF-LSTM, TFN, LMF, MFN) across all metrics: **Accuracy**, **F1 score**, **MAE**, and **Correlation**.

---

## Installation

```bash
pip install torch pandas statsmodels numpy scikit-learn
```

---

## Dataset

Due to GitHub storage limitations, only the **test data** is included in the repo. Full data and pretrained models can be accessed via:  
📂 [Google Drive Dataset](https://drive.google.com/drive/folders/1obcRpOnTbqu2M0_orEyzQHelAOyivFjw)

---

## Multimodal Features

| Modality | Feature Extractor | Dimensions |
|----------|-------------------|------------|
| Text     | BERT              | 768        |
| Audio    | Covarep           | 74         |
| Image    | OpenFace 2.0      | 35         |

---

## Model Structure and Naming

This repository implements and evaluates **16 models** structured into four categories:

### 1. Unimodal MulT Models

| Model        | Modalities | Description            |
|--------------|------------|------------------------|
| A1_MulT_t     | Text       | MulT with text only    |
| A2_MulT_a     | Audio      | MulT with audio only   |
| A3_MulT_i     | Image      | MulT with image only   |

### 2. Bimodal MulT Models

| Model        | Modalities     | Description             |
|--------------|----------------|-------------------------|
| A4_MulT_ta    | Text + Audio   | Bimodal MulT            |
| A5_MulT_ai    | Audio + Image  | Bimodal MulT            |
| A6_MulT_ti    | Text + Image   | Bimodal MulT            |

### 3. Trimodal MulT Models (with/without crossmodal interaction)

| Model        | Modalities                | Description                                  |
|--------------|---------------------------|----------------------------------------------|
| A7_MulT_tai   | Text + Audio + Image      | Standard trimodal MulT                       |
| A8_MulT_tai_Crossmodal_T | Crossmodal focus: Text    | Text-focused interaction                     |
| A9_MulT_tai_Crossmodal_A | Crossmodal focus: Audio   | Audio-focused interaction                    |
| A10_MulT_tai_Crossmodal_I | Crossmodal focus: Image   | Image-focused interaction                    |
| A11_MulT_tai_Crossmodal_TIA | Full Crossmodal (TIA) | Best-performing trimodal interaction model   |

### 4. Benchmark Fusion Models

| Model             | Fusion Strategy       | Description                                                                 |
|------------------|-----------------------|-----------------------------------------------------------------------------|
| A12_LF_LSTM_tai  | Late Fusion           | Independent LSTM encoders for each modality; outputs fused at decision level. |
| A13_EF_LSTM_tai  | Early Fusion          | Modality features concatenated at input level and passed into a single LSTM. |
| A14_TFN_tai      | Tensor Fusion         | Models intra- and inter-modal interactions via a tensor outer product across modalities. |
| A15_LMF_tai      | Low-rank Fusion       | Employs low-rank tensor approximation to efficiently model multimodal interactions. |
| A16_MFN_tai      | Memory Fusion         | Captures dynamic temporal dependencies across modalities using attention and shared memory. |


---

## Usage

```bash
# Clone the repository
git clone https://github.com/multimodal-authenticity/multimodal-authenticity.git
cd multimodal-authenticity



### Run All Evaluation Scripts

To evaluate all 16 models in one go, simply run:

```bash
python main_run_evaluation_pretrained_model.py

---


## Evaluation Metrics

Each model is evaluated using:

- ✅ Accuracy  
- ✅ F1 Score  
- ✅ Mean Absolute Error (MAE)  
- ✅ Pearson Correlation (Corr)  


---
