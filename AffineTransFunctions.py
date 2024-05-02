import numpy as np
import matplotlib.pyplot as plt


''' TESTIRANJE FUNKCIJ z spodnjo kodo -------------------------

# nalozi testno obliko
step = 20
letala = np.load('data/letala.npy', allow_pickle=True)       
# izberi oblike za poravnavo
let = np.array(letala[0], dtype='float')
iPtsRef = let[::step]
let = np.array(letala[12], dtype='float')
iPtsMov = let[::step]
# doloci matriko preslikave

oMat2D_t, oErr = at.alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50, plotProgress=True)
# narisi izhod
iPtsMov_t = np.dot(at.addHomCoord2D(iPtsMov), oMat2D_t.transpose())
plt.figure()
plt.plot(iPtsRef[:,0], iPtsRef[:,1], 'ob')    
plt.plot(iPtsMov[:,0], iPtsMov[:,1], 'or')   
plt.plot(iPtsMov_t[:,0], iPtsMov_t[:,1], 'om')
'''

deg2rad = lambda a: a*np.pi/180

def transAffine2D(iScale =(1,1), iTrans=(0,0), iRot=0, iShear=(0,0)):
    iRot = deg2rad(iRot)
    oMatScale = np.array([[iScale[0], 0, 0], 
                          [0, iScale[1], 0], 
                          [0, 0, 1]])
    
    oMatTrans = np.array([[1, 0, iTrans[0]], 
                          [0, 1, iTrans[1]], 
                          [0, 0, 1]])
    
    oMatRot = np.array([[np.cos(iRot), -np.sin(iRot), 0], 
                        [np.sin(iRot), np.cos(iRot), 0], 
                        [0, 0, 1]])
    
    oMatShear = np.array([[1, iShear[0], 0], 
                          [iShear[1], 1, 0], 
                          [0, 0, 1]])
    
    #oMat2D = oMatScale @ oMatTrans @ oMatRot @ oMatScale

    oMat2D = np.dot(oMatTrans, np.dot(oMatShear, np.dot(oMatRot, oMatScale)))
    return oMat2D

def addHomCoord2D(iPts):
    if iPts.shape[-1] == 3:
        return iPts
    iPts = np.hstack((iPts, np.ones((iPts.shape[0], 1))))
    return iPts

def mapAffineApprox2D(iPtsRef, iPtsMov): # referencne = U
    """Afina aproksimacijska poravnava"""
    # U = TX
    # UX'(XX')^(-1) = T
    iPtsRef = np.matrix(iPtsRef) # U
    iPtsMov = np.matrix(iPtsMov) # X
    # po potrebi dodaj homogeno koordinato
    iPtsRef = addHomCoord2D(iPtsRef)
    iPtsMov = addHomCoord2D(iPtsMov)
    #afina aproksimacia (s psevdoinvrzom XX')
    iPtsRef = iPtsRef.transpose()
    iPtsMov = iPtsMov.transpose()
    #psevdoinverz
    oMat2D = np.dot(iPtsRef, np.linalg.pinv(iPtsMov))
    #oMat2D = iPtsRef*iPtsMov.transpose() * np.linalg.inv(iPts......)
    # Lahko uporabljamo *, ker imamo matrike. Če bi imeli array bi morali uporabljati np.dot 
    return oMat2D 

def findCorrespondingPoints(iPtsRef, iPtsMov):
    """Poisci korespondence kot najblizje tocke"""
    # inicializiraj polje indeksov
    iPtsMov = np.array(iPtsMov)
    iPtsRef = np.array(iPtsRef)
    
    idxPair = -np.ones((iPtsRef.shape[0],1), dtype='int32')
    idxDist = np.ones((iPtsRef.shape[0], iPtsMov.shape[0])) # za shranjevanje razdalij
    
    for i in range(iPtsRef.shape[0]):
        for j in range(iPtsMov.shape[0]):
            idxDist[i,j] = np.sum((iPtsRef[i, :2] - iPtsMov[j,:2])**2)
    # doloci bijektivno preslikavo
    while not np.all(idxDist == np.inf):
        i,j = np.where(idxDist == np.min(idxDist))
        idxPair[i[0]] = j[0]
        idxDist[i[0],:] = np.inf # nasli smo match in smo dali vse ostale pare, ki vsebuje eno od teh tock na Inf
        idxDist[:,j[0]] = np.inf
    # doloci pare tock
    idxValid, idxNotValid = np.where(idxPair>=0)
    idxValid = np.array(idxValid)
    iPtsRef_t = iPtsRef[idxValid,:]
    iPtsMov_t = iPtsMov[idxPair[idxValid].flatten(),:]
    return iPtsRef_t, iPtsMov_t

def alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50, plotProgress=False):
    """Postopek iterativno najblizje tocke"""
    # inicializiraj izhodne parametre
    curMat=[]; # seznam za shranjevanje matrik na vsakem koraku
    oErr = []; iCurIter= 0 ;
    if plotProgress:
        iPtsMov0 = np.matrix(iPtsMov)
        fig = plt.figure()
        ax = fig.add_subplot(111)
    
    #zacni iterativni postopek
    while True:
        #poisci korespondencn tocke
        iPtsRef_t, iPtsMov_t = findCorrespondingPoints(iPtsRef, iPtsMov)
        #doloci afino aproskimacijsko preslkavo 
        oMat2D = mapAffineApprox2D(iPtsRef_t, iPtsMov_t)
        #pososdobi posamezne tocke
        iPtsMov = np.dot(addHomCoord2D(iPtsMov), oMat2D.transpose())
        # izracunaj napako
        curMat.append(oMat2D)
        oErr.append(np.sqrt(np.sum((iPtsRef_t[:,:2] - iPtsMov_t[:,:2])**2)))
        iCurIter +=1
        # preveri kontroln parametre
        dMat = np.abs(oMat2D - transAffine2D())
        if iCurIter>iMaxIter or np.all(dMat<iEps):
            break
    # doloci kompozitum preslikav
    oMat2D = transAffine2D()
    for i in range(len(curMat)):
        
        if plotProgress:
            iPtsMov_t = np.dot(addHomCoord2D(iPtsMov0), oMat2D.transpose())
            ax.clear()
            ax.plot(iPtsRef[:,0], iPtsRef[:,1], 'ob')
            ax.plot(iPtsMov_t[:,0], iPtsMov_t[:,1], 'om')
            fig.canvas.draw()
            plt.pause(1)
        oMat2D = np.dot(curMat[i], oMat2D)
    return oMat2D, oErr


















