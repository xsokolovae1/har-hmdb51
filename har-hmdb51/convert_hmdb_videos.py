"""Prevedie pôvodné videá HMDB51 (*.avi) na priečinky so snímkami.

Výsledná štruktúra je rovnaká ako pri verzii z Kaggle: HMDB51/<trieda>/<klip>/<snímka>.jpg,
ale obsahuje VŠETKY snímky videa (25–30 snímok/s), nie preriedené.

Predtým (v priečinku projektu):
    sudo apt install unrar
    unrar x hmdb51_org.rar drink.rar eat.rar     # z veľkého archívu len potrebné triedy
    unrar x drink.rar videos/                     # -> videos/drink/*.avi
    unrar x eat.rar videos/                       # -> videos/eat/*.avi

Spustenie:  python convert_hmdb_videos.py
"""
from pathlib import Path

import cv2

VIDEOS = Path("videos")       # videos/<trieda>/*.avi
OUT = Path("HMDB51")          # HMDB51/<trieda>/<klip>/<snímka>.jpg

for video in sorted(VIDEOS.glob("*/*.avi")):
    out_dir = OUT / video.parent.name / video.stem
    if out_dir.exists():
        continue                                   # už prevedené
    out_dir.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{i:05d}.jpg"), frame)
        i += 1
    cap.release()
    print(f"{video.parent.name}/{video.stem}: {i} snímok")