from glob import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
size = 256
folder = "/mnt/HDD/imageNet30000/imagenet_images"
paths = glob(f"{folder}/*/*")
print("filtering_image")
keeps = []
for path in tqdm(paths):
    img = Image.open(path)
    w, h = img.size
    if w >= size and h >= size:
        keeps.append(path.replace(f"{folder}/", ""))
print("there are", len(keeps))

df = pd.DataFrame({"path": keeps})
df.to_csv("/mnt/HDD/imageNet30000/metadata.csv", index=False)