import numpy as np
import os
from numrec import Recognizer
from PIL import Image

r = Recognizer()

path = os.path.join(os.getcwd(), r'..\Pics')
im = Image.open(os.path.join(path, 'single3.jpg'))
imraw = [np.array(im)[:, :, 0].flatten()]

print(r.rec(imraw))
