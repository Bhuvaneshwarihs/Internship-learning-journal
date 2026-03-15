import os
from PIL import Image

folder = "dataset/images"

valid = 0
invalid = 0

for file in os.listdir(folder):

    path = os.path.join(folder, file)

    try:
        img = Image.open(path)
        img.verify()  # check image integrity
        valid += 1

    except:
        os.remove(path)
        invalid += 1
        print("Removed broken image:", file)

print("Valid images:", valid)
print("Removed images:", invalid)