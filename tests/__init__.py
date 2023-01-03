"""Downloads Test Data if not existing"""

import os
import shutil
from urllib.parse import urlparse

import requests

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
os.chdir(os.path.join(TEST_DIR, "testdata"))


def download_img(url, img_name):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        parse = urlparse(url)
        img_name = os.path.basename(parse.path)
        with open(img_name, "wb") as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)
    else:
        raise Exception("Failed to download " + url)


with open("TEST_IMAGES.txt", "r") as f:
    img_urls = f.read().splitlines()
    img_names = [os.path.basename(urlparse(url).path) for url in img_urls]

    for url, img_name in zip(img_urls, img_names):
        if not os.path.isfile(img_name):
            print("Downloading " + img_name)
            download_img(url, img_name)
