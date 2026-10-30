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
| R(2+1)D + LSTM | R(2+1)D, 4 úseky klipu po 512 príznakov | 3D konvolúcie + LSTM cez úseky |

## Štruktúra

Všetky súbory sú v priečinku `har-hmdb51/` a príkazy nižšie sa spúšťajú v ňom (`cd har-hmdb51`).

| Súbor | Obsah |
|---|---|
| `har_utils.py` | spoločné funkcie: načítanie snímok, predspracovanie, protokol hodnotenia |
| `convert_hmdb_videos.py` | prevod pôvodných videí HMDB51 na snímky |
| `01_data.ipynb` | zoznam klipov (`index.csv`), delenie podľa zdroja, ukážky |
| `02_features_resnet.ipynb` | príznaky snímok pomocou ResNet18 → `features_resnet.npy` |
| `03_features_video.ipynb` | príznaky klipov pomocou R(2+1)D → `features_r21d.npy` |
| `04_cnn_svm.ipynb` | model CNN-SVM |
| `05_cnn_lstm.ipynb` | model CNN-LSTM |
| `06_video_cnn.ipynb` | model R(2+1)D + SVM, uloženie finálneho modelu pre kameru |
| `06b_video_finetune.ipynb` | doladenie (fine-tuning) siete R(2+1)D |
| `06c_video_lstm.ipynb` | model R(2+1)D + LSTM |
| `07_comparison.ipynb` | porovnanie všetkých modelov |
| `08_camera.py` | rozpoznávanie z kamery alebo videa v reálnom čase |

## Dáta

**HMDB51** – pôvodné videá (klipy z filmov a YouTube, 30 snímok/s) z oficiálneho repozitára
https://huggingface.co/datasets/Serrelab/hmdb51 (nie sú súčasťou repozitára). Stačia dve triedy:

```bash
sudo apt install unrar
wget -c "https://huggingface.co/datasets/Serrelab/hmdb51/resolve/main/hmdb51_org.rar"
unrar x hmdb51_org.rar drink.rar eat.rar      # z veľkého archívu len potrebné triedy
unrar x drink.rar videos/
unrar x eat.rar videos/
python convert_hmdb_videos.py                 # videá -> HMDB51/<trieda>/<klip>/<snímka>.jpg
```

Triedy sa nastavujú v `har_utils.py` (`CLASSES`). Po zmene `har_utils.py` treba reštartovať kernel notebookov.

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
- Hyperparametre sa vyberajú len na trénovacích dátach (krížová validácia pre SVM,
  validačná časť a early stopping pre LSTM a doladenie).
- Hlavná metrika je **balanced accuracy**, keďže triedy sú nevyvážené.

## Výsledky

Balanced accuracy, priemer ± smerodajná odchýlka cez 10 delení (pôvodné videá HMDB51).
Náhodný odhad / väčšinová trieda = 0,50.

| Model | eat / drink (hlavná úloha) | clap / wave | eat / clap |
|---|---|---|---|
| CNN-SVM (ResNet18) | 0,62 | 0,65 | 0,87 |
| CNN-LSTM (ResNet18) | 0,61 | 0,68 | 0,87 |
| R(2+1)D + SVM | **0,80 ± 0,02** | **0,78** | **0,94** |

Pozorovania:

- **Video sieť je najlepšia vo všetkých dvojiciach.** R(2+1)D je predtrénovaná na videách (Kinetics-400),
  takže jej príznaky už obsahujú pohyb. ResNet18 vidí len jednotlivé obrázky, a priemer ani LSTM
  nad nimi túto informáciu nedoplnia.
- **eat / drink je ťažká dvojica.** Obe činnosti vyzerajú podobne (ruka ide k ústam), líšia sa
  hlavne malým predmetom v ruke. Výsledok ~0,80 zodpovedá bežným hodnotám pre túto úlohu.
- **clap / wave** je tiež pohybovo podobná dvojica (ruky pred telom), preto výsledky sú podobné.
- **eat / clap** sú vizuálne odlišné činnosti a všetky modely dosahujú výrazne viac.
  Ukazuje to, že pipeline funguje správne a obmedzením je podobnosť tried, nie kód.
- Rozšírenie trénovacích dát o videá z Charades výsledok na HMDB51 nezlepšilo (iné prostredie a štýl záberov),
  preto sa v konečnej verzii nepoužíva.

Triedy sa menia v `har_utils.py` (`CLASSES`); po zmene treba spustiť notebooky `01` → `07` znova.
