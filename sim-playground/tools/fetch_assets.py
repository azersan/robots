"""Download the CC0 textures the scenes use (ambientCG, https://ambientcg.com, CC0 1.0).

    python tools/fetch_assets.py

Only each material's color map is kept, resized to 2048 px, under assets/textures/
(git-ignored; rerun this script on a fresh checkout).
"""

import io
import os
import sys
import zipfile

import httpx
from PIL import Image

MATERIALS = {
    "grass": "Grass005",
    "concrete": "Concrete048",
    "asphalt": "Asphalt031",
    "bark": "Bark012",
}
SIZE = 2048
OUT = os.path.join(os.path.dirname(__file__), "..", "assets", "textures")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, asset in MATERIALS.items():
        dest = os.path.join(OUT, f"{name}.jpg")
        if os.path.exists(dest):
            print(f"{name}: have {dest}")
            continue
        url = f"https://ambientcg.com/get?file={asset}_2K-JPG.zip"
        print(f"{name}: downloading {asset} ...", flush=True)
        r = httpx.get(url, follow_redirects=True, timeout=120.0)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            color = [n for n in z.namelist() if n.endswith("_Color.jpg")]
            if not color:
                sys.exit(f"{asset}: no _Color.jpg in {z.namelist()}")
            img = Image.open(io.BytesIO(z.read(color[0]))).convert("RGB")
        img.resize((SIZE, SIZE), Image.LANCZOS).save(dest, quality=92)
        print(f"{name}: saved {dest}")


if __name__ == "__main__":
    main()
