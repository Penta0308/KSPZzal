import hashlib
from PIL import Image
import numpy as np

while True:
    print(hashlib.blake2b(np.array(Image.open(input(), 'r'))).hexdigest())