# Rozpoznávanie ľudských činností z videa: jedenie vs. pitie

Materiály a návody k predmetu **Hlboké neurónové siete** (FEI STU Bratislava).

Notebooky ukazujú, ako v prostredí Python (PyTorch, scikit-learn) rozpoznať z krátkeho videoklipu,
či človek **je** alebo **pije**. Porovnávajú sa tri prístupy a najlepší sa aplikuje na kameru.

| Model | Príznaky | Ako sa spracuje čas |
|---|---|---|
| CNN-SVM | ResNet18 (ImageNet), 512 na snímku | priemer cez 16 snímok |
| CNN-LSTM | ResNet18 (ImageNet), 512 na snímku | LSTM číta snímky postupne |
| R(2+1)D + SVM | video sieť R(2+1)D-18 (Kinetics-400), 512 na klip | 3D konvolúcie v sieti |

## Štruktúra

| Súbor | Obsah |
|---|---|
| `har_utils.py` | spoločné funkcie: načítanie snímok, predspracovanie, protokol hodnotenia |
| `01_data.ipynb` | dataset, zoznam klipov, delenie podľa zdrojového videa |
| `02_features_resnet.ipynb` | príznaky snímok pomocou ResNet18 → `features_resnet.npy` |
| `03_features_video.ipynb` | príznaky klipov pomocou R(2+1)D → `features_r21d.npy` |
| `04_cnn_svm.ipynb` | model CNN-SVM |
| `05_cnn_lstm.ipynb` | model CNN-LSTM |
| `06_video_cnn.ipynb` | model R(2+1)D + SVM, uloženie finálneho modelu |
| `07_comparison.ipynb` | porovnanie všetkých modelov |
| `08_camera.py` | rozpoznávanie z kamery alebo videa v reálnom čase |

## Dataset

HMDB51 vo verzii s extrahovanými snímkami: https://www.kaggle.com/datasets/easonlll/hmdb51
(nie je súčasťou repozitára). Stačia dve triedy:

```bash
unzip -q hmdb51.zip 'HMDB51/eat/*' 'HMDB51/drink/*' -d .
```

## Inštalácia

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python scikit-learn pandas matplotlib joblib ipykernel
```

V editore vyberte ako kernel notebookov `venv`. Odporúča sa GPU s CUDA, ale všetko beží aj na CPU.

## Postup

Notebooky spúšťajte v poradí `01` → `07`. Potom:

```bash
python 08_camera.py                      # webkamera
python 08_camera.py --source video.mp4   # videosúbor
```

## Protokol hodnotenia

- Delenie train/test (70/30) **podľa zdrojového videa** – klipy z jedného filmu nie sú naraz v tréningu aj v teste.
- Každý model sa hodnotí na **10 rôznych deleniach**, výsledok je priemer ± smerodajná odchýlka.
- Hyperparametre sa vyberajú len na trénovacích dátach (krížová validácia pre SVM, validačná časť a early stopping pre LSTM).
- Hlavná metrika je **balanced accuracy**, keďže triedy sú nevyvážené (108 eat, 164 drink).
