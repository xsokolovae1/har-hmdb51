# Rozpoznávanie ľudských činností z videa: jedenie vs. pitie

Materiály a návody k predmetu **Hlboké neurónové siete** (FEI STU Bratislava).

Notebooky ukazujú, ako v prostredí Python (PyTorch, scikit-learn) rozpoznať z krátkeho videoklipu,
či človek **je** alebo **pije**. Porovnávajú sa modely CNN-SVM, CNN-LSTM a video CNN
a najlepší sa aplikuje na kameru.

| Model | Príznaky | Ako sa spracuje čas |
|---|---|---|
| CNN-SVM | ResNet18 (ImageNet), 512 na snímku | priemer cez 16 snímok |
| CNN-LSTM | ResNet18 (ImageNet), 512 na snímku | LSTM číta snímky postupne |
| R(2+1)D + SVM | video sieť R(2+1)D-18 (Kinetics-400), 512 na klip | 3D konvolúcie v sieti |
| R(2+1)D doladená | tá istá sieť, posledný blok dotrénovaný na eat/drink | 3D konvolúcie v sieti |

## Štruktúra

| Súbor | Obsah |
|---|---|
| `har_utils.py` | spoločné funkcie: načítanie snímok, predspracovanie, protokol hodnotenia |
| `download_charades.py` | stiahnutie vybraných videí Charades (voliteľné) |
| `00_charades_clips.ipynb` | vystrihnutie klipov eat/drink z Charades (voliteľné) |
| `01_data.ipynb` | zoznam klipov (`index.csv`), delenie podľa zdroja, ukážky |
| `02_features_resnet.ipynb` | príznaky snímok pomocou ResNet18 → `features_resnet.npy` |
| `03_features_video.ipynb` | príznaky klipov pomocou R(2+1)D → `features_r21d.npy` |
| `04_cnn_svm.ipynb` | model CNN-SVM |
| `05_cnn_lstm.ipynb` | model CNN-LSTM |
| `06_video_cnn.ipynb` | model R(2+1)D + SVM, uloženie finálneho modelu pre kameru |
| `06b_video_finetune.ipynb` | doladenie (fine-tuning) siete R(2+1)D |
| `07_comparison.ipynb` | porovnanie všetkých modelov |
| `08_camera.py` | rozpoznávanie z kamery alebo videa v reálnom čase |

## Dáta

**HMDB51** (klipy z filmov), verzia s extrahovanými snímkami:
https://www.kaggle.com/datasets/easonlll/hmdb51 . Stačia dve triedy:

```bash
unzip -q hmdb51.zip 'HMDB51/eat/*' 'HMDB51/drink/*' -d .
```

**Charades** (voliteľné, videá nahrané doma, bližšie podmienkam webkamery):
https://prior.allenai.org/projects/charades (licencia na nekomerčné výskumné účely).

```bash
mkdir -p charades && cd charades
wget https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip
unzip Charades.zip && cd ..
pip install remotezip
python download_charades.py        # stiahne len ~1000 potrebných videí, nie celý 13 GB archív
```

Potom spustite `00_charades_clips.ipynb`. Bez Charades všetko funguje len s HMDB51.

## Inštalácia

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python scikit-learn pandas matplotlib joblib ipykernel
```

V editore vyberte ako kernel notebookov `venv`. Odporúča sa GPU s CUDA, ale všetko beží aj na CPU.

## Postup

Notebooky spúšťajte v poradí `00` (ak používate Charades) → `07`. Potom:

```bash
python 08_camera.py                      # webkamera
python 08_camera.py --source video.mp4   # videosúbor
```

## Protokol hodnotenia

- Delenie train/test (70/30) **podľa zdroja** – klipy z jedného filmu (HMDB51) alebo od jedného
  človeka (Charades) nie sú naraz v tréningu aj v teste.
- Každý model sa hodnotí na **10 rôznych deleniach**, výsledok je priemer ± smerodajná odchýlka.
- Hyperparametre sa vyberajú len na trénovacích dátach (krížová validácia pre SVM,
  validačná časť a early stopping pre LSTM a doladenie).
- Hlavná metrika je **balanced accuracy**, keďže triedy sú nevyvážené.
- Výsledky sa uvádzajú zvlášť pre testovacie klipy z HMDB51 a z Charades.
