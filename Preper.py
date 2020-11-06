import numpy as np

import os

import natsort
import matplotlib.pyplot as plt

from skimage import io, img_as_float

from itertools import groupby

from operator import itemgetter

import pandas as pd

from IPython.display import display

np.seterr(divide='ignore', invalid='ignore')

import matplotlib.colorbar

matplotlib.axes.Axes.imshow

matplotlib.pyplot.imshow

matplotlib.figure.Figure.colorbar

matplotlib.pyplot.colorbar

import imageio

from PIL import Image
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import numpy as np
from matplotlib import pyplot as plt

# Definir width and height
w, h = 695,725


# lecture du fichier selectionnèe
with open('without_hdr/olive/b2') as f:
    d1 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img1 = Image.fromarray(d1)
with open('without_hdr/olive/b3') as f:
    d2 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img2 = Image.fromarray(d2)
with open('without_hdr/olive/b4') as f:
    d3 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img3 = Image.fromarray(d3)
with open('without_hdr/olive/b5') as f:
    d4 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img4 = Image.fromarray(d4)
with open('without_hdr/olive/b6') as f:
    d5 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img5 = Image.fromarray(d5)
with open('without_hdr/olive/b7') as f:
    d6 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img6 = Image.fromarray(d6)
with open('without_hdr/olive/b8') as f:
    d7 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img7 = Image.fromarray(d7)
with open('without_hdr/olive/b8a') as f:
    d8 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img8 = Image.fromarray(d8)
with open('without_hdr/olive/b11') as f:
    d9= np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img9 = Image.fromarray(d9)
with open('without_hdr/olive/b12') as f:
    d10 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img10 = Image.fromarray(d10)

y10 = np.expand_dims(img1, axis=2)
y1 = np.expand_dims(img2, axis=2)
y2 = np.expand_dims(img3, axis=2)
y3 = np.expand_dims(img4, axis=2)
y4 = np.expand_dims(img5, axis=2)
y5 = np.expand_dims(img6, axis=2)
y6 = np.expand_dims(img7, axis=2)
y7 = np.expand_dims(img8, axis=2)
y8 = np.expand_dims(img9, axis=2)
y9 = np.expand_dims(img10, axis=2)

imgolive=np.dstack((y10,y1,y2,y3,y4,y5,y6,y7,y8,y9))
from scipy.io import savemat
savemat("olive.mat",{"foo":imgolive})



w1, h1 = 4232,6729


# lecture du fichier selectionnèe
with open('without_hdr/non_olivier/b2') as f:
    dno1 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno1 = Image.fromarray(dno1)
with open('without_hdr/non_olivier/b3') as f:
    dno2 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno2 = Image.fromarray(dno2)
with open('without_hdr/non_olivier/b4') as f:
    dno3 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno3 = Image.fromarray(dno3)
with open('without_hdr/non_olivier/b5') as f:
    dno4 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno4 = Image.fromarray(dno4)
with open('without_hdr/non_olivier/b6') as f:
    dno51 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno5 = Image.fromarray(dno51)
with open('without_hdr/non_olivier/b7') as f:
    dno6 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno6 = Image.fromarray(dno6)
with open('without_hdr/non_olivier/b8') as f:
    dno7 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno7 = Image.fromarray(dno7)
with open('without_hdr/non_olivier/b8a') as f:
    dno8 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno8 = Image.fromarray(dno8)
with open('without_hdr/non_olivier/b11') as f:
    dno9= np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)

# enregistrer l'image
imgno9 = Image.fromarray(dno9)
with open('without_hdr/non_olivier/b12') as f:
    dno10 = np.fromfile(f,dtype=np.uint16,count=w1*h1).reshape(h1,w1)
imgno10 = Image.fromarray(dno10)
# enregistrer l'image


yn10 = np.expand_dims(imgno1, axis=2)
yn1 = np.expand_dims(imgno2, axis=2)
yn2 = np.expand_dims(imgno3, axis=2)
yn3 = np.expand_dims(imgno4, axis=2)
yn4 = np.expand_dims(imgno5, axis=2)
yn5 = np.expand_dims(imgno6, axis=2)
yn6 = np.expand_dims(imgno7, axis=2)
yn7 = np.expand_dims(imgno8, axis=2)
yn8 = np.expand_dims(imgno9, axis=2)
yn9 = np.expand_dims(imgno10, axis=2)


imgNonOlive=np.dstack((yn10,yn1,yn2,yn3,yn4,yn5,yn6,yn7,yn8,yn9))
from scipy.io import savemat
savemat("non olive.mat",{"foo":imgNonOlive})



size=[16,16]

strides=[8,8]
def Convolution_opMS(Image, size, strides):
   
    start_x = 0
    start_y = 0
    end_x = Image.shape[0] - size[0]
    end_y = Image.shape[1] - size[1]

    n_rows = (end_x//strides[0]) + 1
    n_columns = (end_y//strides[1]) + 1
    small_images = []
    
    for i in range(n_rows):
        for j in range(n_columns):
            
            new_start_x = start_x+i*strides[0]
            new_start_y= start_y+j*strides[1]
            #print('x',new_start_x)
            #print('y ',new_start_y)
            if (Image[new_start_x,new_start_y,2]>0) & (Image[new_start_x+15,new_start_y,2] >0) & (Image[new_start_x,new_start_y+15,2] >0 ) & (Image[new_start_x+15,new_start_y+15,2] >0 ):
                small_images.append(Image[new_start_x:new_start_x+16,new_start_y:new_start_y+16])        
        m=np.asarray(small_images)
    return m




#len(Y)
#savemat("nonolivecoup.mat",{"foo":Y})
""" print(len(z))
allonlist=z.tolist()
savemat("all.mat",{"fo":z})
savemat("allonlist.mat",{"fo":allonlist})
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional
import os
import numpy as np

import os

import natsort
import matplotlib.pyplot as plt

from skimage import io, img_as_float

from itertools import groupby

from operator import itemgetter

import pandas as pd

from IPython.display import display

np.seterr(divide='ignore', invalid='ignore')

import matplotlib.colorbar

matplotlib.axes.Axes.imshow

matplotlib.pyplot.imshow

matplotlib.figure.Figure.colorbar

matplotlib.pyplot.colorbar

import imageio

from PIL import Image
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
import natsort
import matplotlib.pyplot as plt

from skimage import io, img_as_float

from itertools import groupby

from operator import itemgetter

import pandas as pd

from IPython.display import display

np.seterr(divide='ignore', invalid='ignore')

import matplotlib.colorbar

matplotlib.axes.Axes.imshow

matplotlib.pyplot.imshow

matplotlib.figure.Figure.colorbar

matplotlib.pyplot.colorbar

import imageio

from PIL import Image
import numpy as np


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
import numpy as np
from matplotlib import pyplot as plt

# Definir width and height
w, h = 10980,10980


# lecture du fichier selectionnèe
with open('without_hdr/test/b2') as f:
    d1 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img1 = Image.fromarray(d1)
with open('without_hdr/test/b4') as f:
    d2 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img2 = Image.fromarray(d2)
with open('without_hdr/test/b4') as f:
    d3 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img3 = Image.fromarray(d3)
with open('without_hdr/test/b4') as f:
    d4 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img4 = Image.fromarray(d4)
with open('without_hdr/test/b6') as f:
    d5 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img5 = Image.fromarray(d5)
with open('without_hdr/test/b7') as f:
    d6 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img6 = Image.fromarray(d6)
with open('without_hdr/test/b8a') as f:
    d7 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img7 = Image.fromarray(d7)
with open('without_hdr/test/b8a') as f:
    d8 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img8 = Image.fromarray(d8)
with open('without_hdr/test/b11') as f:
    d9= np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img9 = Image.fromarray(d9)

with open('without_hdr/test/b12') as f:
    d10 = np.fromfile(f,dtype=np.uint16,count=w*h).reshape(h,w)

# enregistrer l'image
img10 = Image.fromarray(d10)

y10 = np.expand_dims(img1, axis=2)
y1 = np.expand_dims(img2, axis=2)
y2 = np.expand_dims(img3, axis=2)
y3 = np.expand_dims(img4, axis=2)
y4 = np.expand_dims(img5, axis=2)
y5 = np.expand_dims(img6, axis=2)
y6 = np.expand_dims(img7, axis=2)
y7 = np.expand_dims(img8, axis=2)
y8 = np.expand_dims(img9, axis=2)
y9 = np.expand_dims(img10, axis=2)


test=np.dstack((y10,y1,y2,y3,y4,y5,y6,y7,y8,y9))
def opMS(Image, size, strides):  
    start_x = 0
    start_y = 0
    end_x = Image.shape[0] - size[0]
    end_y = Image.shape[1] - size[1]

    n_rows = (end_x//strides[0]) + 1
    n_columns = (end_y//strides[1]) + 1
    small_images = []
 
    for i in range(n_rows):
        for j in range(n_columns):
            
            new_start_x = start_x+i*strides[0]
            new_start_y= start_y+j*strides[1]
            #print('x',new_start_x)
            #print('y ',new_start_y)
            #if (Image[new_start_x,new_start_y,2]>0) & (Image[new_start_x+15,new_start_y,2] >0) & (Image[new_start_x,new_start_y+15,2] >0 ) & (Image[new_start_x+15,new_start_y+15,2] >0 ):
            small_images.append(Image[new_start_x:new_start_x+16,new_start_y:new_start_y+16])        
        m=np.asarray(small_images)
    return m


 