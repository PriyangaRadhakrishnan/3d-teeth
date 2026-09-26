import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


VIEWS = ["top", "front", "back", "left", "right"]


def contact_sheet(sample: Path, category: str, output: Path) -> None:
    tiles = []
    for view in VIEWS:
        path = sample / category / f"{view}.png"
        if path.exists():
            image = Image.open(path).convert("RGB").resize((320, 320))
            canvas = Image.new("RGB", (320, 350), "white")
            canvas.paste(image, (0, 30))
            ImageDraw.Draw(canvas).text((8, 8), view, fill="black")
            tiles.append(canvas)
    sheet = Image.new("RGB", (960, 700), "#dddddd")
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * 320, (index // 3) * 350))
    sheet.save(output)


parser = argparse.ArgumentParser(description="Validate rendered Phase 1-5 artifacts")
parser.add_argument("sample")
args = parser.parse_args()
sample = Path(args.sample)
metadata = json.loads((sample / "metadata.json").read_text(encoding="utf-8"))
for category in ("images", "depth_vis", "normals", "masks"):
    contact_sheet(sample, category, sample / "reports" / f"{category}_contact_sheet.png")
for view in VIEWS:
    for relative in (f"images/{view}.png", f"depth/{view}.exr", f"normals/{view}.png", f"masks/{view}.png"):
        if not (sample / relative).exists():
            raise FileNotFoundError(sample / relative)
if metadata["views"] != VIEWS:
    raise ValueError(f"Unexpected view list: {metadata['views']}")
print(f"Validated {sample}: five RGB, depth, normal, and mask views")