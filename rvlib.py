# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 08:47:50 2015

@author: Žiga Špiclin

RVLIB: knjižnica funkcij iz laboratorijskih vaj
       pri predmetu Robotski vid
"""
import numpy as np
import PIL.Image as im
import matplotlib.pyplot as plt
import matplotlib.cm as cm # uvozi barvne lestvice
import scipy.ndimage as ndi

def loadImageRaw(iPath, iSize, iFormat):
    '''
    Naloži sliko iz raw datoteke
    
    Parameters
    ----------
    iPath : str 
        Pot do datoteke
    iSize : tuple 
        Velikost slike
    iFormat : str
        Tip vhodnih podatkov
    
    Returns
    ---------
    oImage : numpy array
        Izhodna slika
    
    
    '''
    
    oImage = np.fromfile(iPath, dtype=iFormat) # nalozi raw datoteko
    oImage = np.reshape(oImage, iSize) # uredi v matriko
    
    return oImage


def showImage(iImage, iTitle=''):
    '''
    Prikaže sliko iImage in jo naslovi z iTitle
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika 
    iTitle : str 
        Naslov za sliko
    
    Returns
    ---------
    Nothing
    
    
    '''
    plt.figure() # odpri novo prikazno okno
    
    if iImage.ndim == 3 and iImage.shape[0] == 3:
        iImage = np.transpose(iImage,[1,2,0])

    plt.imshow(iImage, cmap = cm.Greys_r) # prikazi sliko v novem oknu
    plt.suptitle(iTitle) # nastavi naslov slike
    plt.xlabel('x')
    plt.ylabel('y')


def saveImageRaw(iImage, iPath, iFormat):
    '''
    Shrani sliko na disk
    
    Parameters
    ----------
    iImage : numpy.ndarray
        Vhodna slika za shranjevanje
    iPath : str
        Pot in ime datoteke, v katero želimo sliko shraniti
    iFormat : str
        Tip podatkov v matriki slike
    
    Returns
    ---------
    Nothing
    '''
    iImage = iImage.astype(iFormat)
    iImage.tofile(iPath) # zapisi v datoteko


def loadImage(iPath):
    '''
    Naloži sliko v standardnih formatih (bmp, jpg, png, tif, gif, idr.)
    in jo vrni kot matriko
    
    Parameters
    ----------
    iPath - str
        Pot do slike skupaj z imenom
        
    Returns
    ----------
    oImage - numpy.ndarray
        Vrnjena matrična predstavitev slike
    '''
    oImage = np.array(im.open(iPath))
    if oImage.ndim == 3:
        oImage = np.transpose(oImage,[2,0,1])
    elif oImage.ndim == 2:
        oImage = np.transpose(oImage,[1,0])   
    return oImage


def saveImage(iPath, iImage, iFormat):
    '''
    Shrani sliko v standardnem formatu (bmp, jpg, png, tif, gif, idr.)
    
    Parameters
    ----------
    iPath : str
        Pot do slike z željenim imenom slike
    iImage : numpy.ndarray
        Matrična predstavitev slike
    iFormat : str
        Željena končnica za sliko (npr. 'bmp')
    
    Returns
    ---------
    Nothing

    '''
    if iImage.ndim == 3:
        iImage = np.transpose(iImage,[1,2,0])
    elif iImage.ndim ==2:
        iImage = np.transpose(iImage,[1,0])     
    img = im.fromarray(iImage) # ustvari slikovni objekt iz matrike
    img.save(iPath.split('.')[0] + '.' + iFormat)


def drawLine(iImage, iValue, x1, y1, x2, y2):
    ''' Narisi digitalno daljico v sliko

        Parameters
        ----------
        iImage : numpy.ndarray
            Vhodna slika
        iValue : tuple, int
            Vrednost za vrisavanje (barva daljice).
            Uporabi tuple treh elementov za barvno sliko in int za sivinsko sliko
        x1 : int
            Začetna x koordinata daljice
        y1 : int
            Začetna y koordinata daljice
        x2 : int
            Končna x koordinata daljice
        y2 : int
            Končna y koordinata daljice
    '''    
    
    oImage = iImage    
    
    if iImage.ndim == 3:
        assert type(iValue) == tuple, 'Za barvno sliko bi paramter iValue moral biti tuple treh elementov'
        for rgb in range(3):
            drawLine(iImage[rgb,:,:], iValue[rgb], x1, y1, x2, y2)
    
    elif iImage.ndim == 2:
        assert type(iValue) == int, 'Za sivinsko sliko bi paramter iValue moral biti int'
    
        dx = np.abs(x2 - x1)
        dy = np.abs(y2 - y1)
        if x1 < x2:
            sx = 1
        else:
            sx = -1
        if y1 < y2:
            sy = 1
        else:
            sy = -1
        napaka = dx - dy
     
        x = x1
        y = y1
        
        while True:
            oImage[y-1, x-1] = iValue
            if x == x2 and y == y2:
                break
            e2 = 2*napaka
            if e2 > -dy:
                napaka = napaka - dy
                x = x + sx
            if e2 < dx:
                napaka = napaka + dx
                y = y + sy
    
    return oImage
    
    
def colorToGray(iImage):
    '''
    Pretvori barvno sliko v sivinsko.
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna barvna slika
        
    Returns
    -------
    oImage : numpy.ndarray
        Sivinska slika
    '''
    dtype = iImage.dtype
    r = iImage[0,:,:].astype('float')
    g = iImage[1,:,:].astype('float')
    b = iImage[2,:,:].astype('float')
    
    return (r*0.299 + g*0.587 + b*0.114).astype(dtype)
    
    
def computeHistogram(iImage, iNumBins, iRange=[], iDisplay=False, iTitle=''):
    '''
    Izracunaj histogram sivinske slike
    
    Parameters
    ---------
    iImage : numpy.ndarray
        Vhodna slika, katere histogram želimo izračunati

    iNumBins : int
        Število predalov histograma
        
    iRange : tuple, list
        Minimalna in maksimalna sivinska vrednost 

    iDisplay : bool
        Vklopi/izklopi prikaz histograma v novem oknu

    iTitle : str
        Naslov prikaznega okna
        
    Returns
    -------
    oHist : numpy.ndarray
        Histogram sivinske slike
    oEdges: numpy.ndarray
        Robovi predalov histograma
    '''    
    iImage = np.asarray(iImage)
    iRange = np.asarray(iRange)
    if iRange.size == 2:
        iMin, iMax = iRange
    else:
        iMin, iMax = np.min(iImage), np.max(iImage)
    oEdges = np.linspace(iMin, iMax+1, iNumBins+1)
    oHist = np.zeros([iNumBins,])
    for i in range(iNumBins):
        idx = np.where((iImage >= oEdges[i]) * (iImage < oEdges[i+1]))
        if idx[0].size > 0:
            oHist[i] = idx[0].size
    if iDisplay:
        plt.figure()
        plt.bar(oEdges[:-1], oHist)
        plt.suptitle(iTitle)

    return oHist, oEdges
    
    
def computeContrast(iImages):
    '''
    Izracunaj kontrast slik
    
    Parameters
    ---------
    iImages : list of numpy.ndarray
        Vhodne slike, na katerih želimo izračunati kontrast
        
    Returns : list
        Seznam kontrastov za vsako vhodno sliko
    '''
    oM = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        fmin = np.percentile(iImages[i].flatten(),5)
        fmax = np.percentile(iImages[i].flatten(),95)
        oM[i] = (fmax - fmin)/(fmax + fmin)
    return oM
    
    
def computeEffDynRange(iImages):
    '''
    Izracunaj efektivno dinamicno obmocje
    
    Parameters
    ----------
    iImages : numpy.ndarray
        Vhodne slike
        
    Returns
    --------
    oEDR : float
        Vrednost efektivnega dinamicnega obmocja
    '''
    L = np.zeros((len(iImages,)))
    sig = np.zeros((len(iImages),))
    for i in range(len(iImages)):
        L[i] = np.mean(iImages[i].flatten())
        sig[i] = np.std(iImages[i].flatten())
    oEDR = np.log2((L.max() - L.min())/sig.mean())
    return oEDR
    

def computeSNR(iImage1, iImage2):
    '''
    Vrne razmerje signal/sum
    
    Paramters
    ---------
    iImage1, iImage2 : np.ndarray
        Sliki področij zanimanja, med katerima računamo SNR
        
    Returns
    ---------
    oSNR : float
        Vrednost razmerja signal/sum
    '''
    mu1 = np.mean(iImage1.flatten())
    mu2 = np.mean(iImage2.flatten())
    
    sig1 = np.std(iImage1.flatten())
    sig2 = np.std(iImage2.flatten())
    
    oSNR = np.abs(mu1 - mu2)/np.sqrt(sig1**2 + sig2**2)
            
    return oSNR

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

# Funkcija za prikazovanje grayscale slik:
def showImage(iImage, iTitle = ''):
    '''
    Funkcija za prikazovanje grayscale slik
    '''
    plt.imshow(iImage, cmap="gray")
    plt.title(iTitle)
    plt.xlabel('x')
    plt.ylabel('y')
    

def convertRGB2HSV(iImage): 
    '''
    Pretvarjanje iz rgb slike (np.array) v hsv sliko (np. array) in obratno
    '''
    iImage = iImage.astype('float')
    iImage = iImage/255 # bo samo 8-bitna
    r, g, b = iImage[:,:,0],iImage[:,:,1],iImage[:,:,2]
    
    h = np.zeros_like(r)
    s = np.zeros_like(r)
    v = np.zeros_like(r)
    
    Cmax = np.maximum(r, np.maximum(g,b))
    Cmin = np.minimum(r, np.minimum(g,b))
    delta = Cmax - Cmin + 1e-7
    
    h[Cmax == r] = 60.0 * ((g[Cmax==r] - b[Cmax==r])/ delta[Cmax == r] % 6.0) # 60 deg
    
    h[Cmax == g] = 60.0 * ((b[Cmax==g] - r[Cmax==g])/ delta[Cmax == g] + 2.0)
    
    h[Cmax == b] = 60.0 * ((r[Cmax==b] - g[Cmax==b])/ delta[Cmax == b]  + 4.0)
    
    s[delta!=0.0] = delta[delta!=0.0]/(Cmax[delta!=0.0] + 1e-7)
    
    v = Cmax
    
    # ustvari izhodno sliko
    oImage = np.zeros_like(iImage)
    oImage[:,:,0] = h
    oImage[:,:,1] = s
    oImage[:,:,2] = v
    
    return oImage


def convertHSV2RGB(iImage):
    '''
    Pretvarjanje iz hsv slike (np.array) v rgb sliko (np. array)
    '''
    iImage = iImage.astype('float')
    
    h, s, v = iImage[:,:,0], iImage[:,:,1], iImage[:,:,2]
    
    C = v * s
    X = C * (1.0 - np.abs(((h/60.0) % 2) -1))
    m = v - C 
    
    r = np.zeros_like( h )
    g = np.zeros_like( h )
    b = np.zeros_like( h )
    
    #  0<= H < 60
    r[(h>=0.0) & (h<60.0)] = C[(h>=0.0) & (h<60.0)]
    g[(h>=0.0) & (h<60.0)] = X[(h>=0.0) & (h<60.0)]
    # b[(h>=0) & h(<60.0)] = 0

    # 60 <= H <120
    r[(h>=60.0) & (h<120.0)] = X[(h>=60.0) & (h<120.0)]
    g[(h>=60.0) & (h<120.0)] = C[(h>=60.0) & (h<120.0)]
    
    # 120 <= H < 180
    g[(h>=120.0) & (h<180.0)] = C[(h>=120.0) & (h<180.0)]
    b[(h>=120.0) & (h<180.0)] = X[(h>=120.0) & (h<180.0)]
                                  
    # 180 <= H < 240 
    g[(h>=180.0) & (h<240.0)] = X[(h>=180.0) & (h<240.0)]
    b[(h>=180.0) & (h<240.0)] = C[(h>=180.0) & (h<240.0)]
                                  
    # 240 <= H < 300
    r[(h>=240.0) & (h<300.0)] = X[(h>=240.0) & (h<300.0)]
    b[(h>=240.0) & (h<300.0)] = C[(h>=240.0) & (h<300.0)]
                                 
    # 300 <= H <360
    r[(h>=300.0) & (h<360.0)] = C[(h>=300.0) & (h<360.0)]
    b[(h>=300.0) & (h<360.0)] = X[(h>=300.0) & (h<360.0)]
                                  
    r = r + m
    g= g + m
    b = b + m
    
    # ustvari izhodno sliko
    oImage = np.zeros_like(iImage)
    oImage[:,:,0] = r
    oImage[:,:,1] = g
    oImage[:,:,2] = b
    
    # zaokrozevanje vrednsti
    oImage = 255.0 * oImage
    oImage[oImage>255.0]  = 255.0
    oImage[oImage<0.0] = 0.0
    
    oImage = oImage.astype('uint8')
    return oImage



