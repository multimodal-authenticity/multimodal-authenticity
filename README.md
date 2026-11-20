## GenAI Supported Measure of Authenticity in Tiktok Videos: How and When Authenticity Sells in Social Media Influencer Marketing



## Overview

This repository contains the code and data used in the  paper *"GenAI Supported Measure of Authenticity in Tiktok Videos: How and When Authenticity Sells in Social Media Influencer Marketing"*.

This work builds and compares **16 multimodal machine learning (MMML) models**, including (1) unimodal, bimodal, and trimodal MulT-based models, (2) trimodal MulT-based models with text(t)/audio(a)/image(i) crossmodal, and (3) benchmark fusion models (e.g., LF-LSTM, EF-LSTM, TFN, LMF, MFN).

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
python -m pip install -r requirements.txt
```

---

## Dataset

Due to GitHub storage limitations, only the **test data** is included in the repo. Full data and pretrained models can be accessed via:  
📂 [Google Drive Dataset](https://drive.google.com/drive/folders/1obcRpOnTbqu2M0_orEyzQHelAOyivFjw)

---

## Multimodal Features

| Modality | Feature Extractor | Dimensions |
| -------- | ----------------- | ---------- |
| Text     | BERT              | 768        |
| Audio    | Covarep           | 74         |
| Image    | OpenFace 2.0      | 35         |

---

## Model Structure and Naming

The 16 model IDs below follow the order of the three panels in Table 2. Within Panel B, `X → Y` indicates that modality X updates the representation of modality Y. I, A, and T denote image, audio, and text.

| Model ID | Model in Table 2 | Encoding or fusion |
|---|---|---|
| **Panel A. Independent modality encoding (no crossmodal attention)** | | |
| `A1_MulT_t` | MulT with Text | Text only |
| `A2_MulT_a` | MulT with Audio | Audio only |
| `A3_MulT_i` | MulT with Image | Image only |
| `A4_MulT_ta` | MulT with Text + Audio | Independent text and audio encoding |
| `A5_MulT_ai` | MulT with Image + Audio | Independent image and audio encoding |
| `A6_MulT_ti` | MulT with Image + Text | Independent image and text encoding |
| `A7_MulT_tai` | MulT with Image + Audio + Text | Independent trimodal encoding |
| **Panel B. Crossmodal encoding (with crossmodal attention)** | | |
| `A7_MulT_tai` | MulT with Image + Audio + Text (reference) | Same model as Panel A |
| `A8_MulT_tai_Crossmodal_T` | Trimodal MulT with crossmodal updates to text | I → T and A → T |
| `A9_MulT_tai_Crossmodal_A` | Trimodal MulT with crossmodal updates to audio | I → A and T → A |
| `A10_MulT_tai_Crossmodal_I` | Trimodal MulT with crossmodal updates to image | A → I and T → I |
| `A11_MulT_tai_Crossmodal_TIA` | Full crossmodal trimodal MulT | I → T, A → T, A → I, T → I, I → A, and T → A |
| **Panel C. Benchmark models** | | |
| `A12_LF_LSTM_tai` | LF-LSTM | Late fusion |
| `A13_EF_LSTM_tai` | EF-LSTM | Early fusion |
| `A14_TFN_tai` | TFN | Tensor fusion |
| `A15_LMF_tai` | LMF | Low-rank fusion |
| `A16_MFN_tai` | MFN | Memory fusion |

---

## Usage

```bash
# Clone the repository
git clone https://github.com/multimodal-authenticity/multimodal-authenticity.git
cd multimodal-authenticity
```



### Run All Evaluation Scripts

To evaluate all 16 models in one go, simply run:

```bash
python main_run_evaluation_pretrained_model.py
```

---


## Evaluation Metrics

Each model is evaluated using:

- ✅ Accuracy  
- ✅ F1 Score  
- ✅ Mean Absolute Error (MAE)  
- ✅ Pearson Correlation (Corr)  


---

```
