import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


for b in range(1, 8):
    d=1
    os.makedirs("./dl-microstructures/class_wise_data/Class_%d"%b, mode=0o777)
    for a in range(5, 51, 5):
        for c in range(1, 51):
            f = open("./dl-microstructures/data/Phase_%d/Class_%d/%d.txt"%(a, b, c), "r")
            im = np.empty(shape=(256,256), dtype=np.uint8)
            for line in f:
                l = list(map(int, line.split()))
                im[l[0]-1][l[1]-1] = l[2]*255
            
            
            img = Image.fromarray(im, 'L')
            img.save("./dl-microstructures/class_wise_data/Class_%d/%d.png"%(b, d))
            d+=1