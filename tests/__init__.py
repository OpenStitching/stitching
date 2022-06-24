"""Downloads Test Data if not existing"""

import os
import shutil

import requests

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(TEST_DIR, "testdata"))


def download_img(url, img):
    r = requests.get(url + img, stream=True)
    if r.status_code == 200:
        with open(img, "wb") as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        raise Exception("Failed to download " + url + img)


with open("TEST_IMAGES.txt", "r") as f:
    required_imgs = f.read().splitlines()
download_url = required_imgs.pop(0)

for img in required_imgs:
    if not os.path.isfile(img):
        print("Downloading " + img)
        download_img(download_url, img)
