import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as im
from scipy.interpolate import interpn

# Nalozi sliko
def loadImage(iPath):
    oImage = np.array(im.open(iPath))
    return oImage

# Prikazi sliko
def showImage(iImage, iTitle=''):
    plt.figure()
    plt.imshow(iImage, cmap = 'gray')
    plt.suptitle(iTitle)
    plt.xlabel('x')
    plt.ylabel('y')

# Pretvori v sivinsko sliko
def colorToGray(iImage):
    dtype = iImage.dtype
    r = iImage[:,:,0].astype('float')
    g = iImage[:,:,1].astype('float')
    b = iImage[:,:,2].astype('float')
    # Testirajte funkcijo geomCalibErr

# Load an image
iCalImage = loadImage('path_to_your_image_file')

# Now you can use iCalImage
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(colorToGray(iCalImage), cmap='gray')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(colorToGray(iCalImage), cmap='gray')

points = []
def onclick(event):
    if event.key == 'shift':
        x, y = event.xdata, event.ydata
        points.append((x, y))
        ax.plot(x, y, 'or')
        fig.canvas.draw()
    
ka = fig.canvas.mpl_connect('button_press_event', onclick)
print(points)

# VRSTNI RED POMEMBEN KLIKAMO (SHIFT+klik) V SMERI URINEGA KAZALCA