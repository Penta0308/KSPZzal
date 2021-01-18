import modules
import os
from PIL import Image
import numpy as np


for s in modules.config_get("dirs"):
    for q in s:
        tr = []
        for t in os.listdir(q):
            img = Image.open(q + "/" + t, 'r')
            print(t)
            if modules.hasmeaning(np.array(img)):
                continue
            tr.append(t)
        for t in tr:
            os.remove(q + "/" + t)
            print(t)