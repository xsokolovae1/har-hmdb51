# Human Activity Recognition from Video: Eating vs Drinking

Teaching materials for the course **Hlboké neurónové siete** (Deep Neural Networks), FEI STU Bratislava.

The notebooks show how to recognise human activities in short video clips with deep neural networks in PyTorch. The task is binary classification of two HMDB51 classes, **eat** and **drink**. The goal is to compare several model structures and run the best one on a live camera.

## Approach

Every clip is reduced to 16 evenly spaced frames. Each frame goes through a ResNet18 pretrained on ImageNet and becomes a 512-dimensional feature vector, so a clip is described by a `(16, 512)` sequence. The models below are trained on these features:

| Model | How the time dimension is used |
|---|---|
| CNN + SVM / Logistic Regression | features are averaged over the 16 frames, so frame order is ignored |
| CNN + LSTM | the 16 feature vectors are read as an ordered sequence |

The ResNet18 is used in two variants:

- **frozen**, using the ImageNet weights as they are;
- **fine-tuned**, with the last block and the classifier trained on frames from the training clips only.

## Repository structure

| File | Content |
|---|---|
| `utils.py` | shared frame loading and preprocessing (used in training and for the camera) |
| `data_preparing.ipynb` | clip index, train/val/test split, data inspection |
| `features.ipynb` | feature extraction with the frozen ResNet18 → `features.npy` |
| `finetune.ipynb` | fine-tuning ResNet18 on frames → `features_ft.npy`, `resnet18_ft.pt` |
| `cnn_svm.ipynb` | CNN + SVM / Logistic Regression baseline |
| `index.csv` | list of clips with labels and split assignment |

## Dataset

HMDB51 (Kuehne et al., 2011) is **not included** in this repository. This project uses the version with pre-extracted frames, organised as `HMDB51/<class>/<clip>/<frame>.jpg`. Only two classes are needed:

```bash
unzip -q hmdb51.zip 'HMDB51/eat/*' 'HMDB51/drink/*' -d .
```

This gives 108 *eat* and 164 *drink* clips.

The split into train/val/test (~56/14/30 %) is grouped by **source video**. Many clips in HMDB51 are cut from the same movie, and a random split would leak near-identical clips between training and test data.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python scikit-learn pandas matplotlib ipykernel
```

A CUDA-capable GPU is recommended for fine-tuning. Feature extraction and the classifiers also run on CPU.

## How to run

Run the notebooks in this order with the `venv` kernel:

1. `data_preparing.ipynb`
2. `features.ipynb`
3. `finetune.ipynb`
4. `cnn_svm.ipynb`

## Results

Test set: 80 clips (34 eat, 46 drink). A majority-class baseline reaches 0.575 accuracy (0.5 balanced accuracy).

| Features | Model | Test accuracy | Balanced accuracy |
|---|---|---|---|
| frozen ResNet18 | Logistic Regression | 0.65 | 0.65 |
| fine-tuned ResNet18 | … | … | … |
| … | CNN + LSTM | … | … |

*Work in progress.*
