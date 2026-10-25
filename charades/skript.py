import pandas as pd
from pathlib import Path
from remotezip import RemoteZip

URL = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip"
NEED = {"c106", "c156", "c065"}          # drinking, eating something, eating a sandwich

ids = set()
for f in ["Charades/Charades_v1_train.csv", "Charades/Charades_v1_test.csv"]:
    ann = pd.read_csv(f)
    for vid, acts in zip(ann.id, ann.actions.fillna("")):
        if any(a.split()[0] in NEED for a in acts.split(";") if a):
            ids.add(vid)
print("нужных роликов:", len(ids))

out = Path("videos")
with RemoteZip(URL) as z:
    members = {Path(n).stem: n for n in z.namelist() if n.endswith(".mp4")}
    for i, vid in enumerate(sorted(ids)):
        if vid in members and not (out / members[vid]).exists():   # уже скачанные пропускаем
            z.extract(members[vid], out)
        if i % 50 == 0:
            print(i, "/", len(ids))