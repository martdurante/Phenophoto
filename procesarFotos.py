import numpy as np
import itertools
import cv2
import os
import time
from distutils import util
import argparse
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats

# a little helper function for getting all dettected marker ids
# from the reference image markers
def which(x, values):
    indices = []
    for ii in list(values):
        if ii in x:
            indices.append(list(x).index(ii))
    return indices

### 1- GENERO ARUCO DICTINARY
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

### 2- CARGO IMAGEN DE REFERENCIA
refImage = cv2.cvtColor(cv2.imread('panelAru.jpg'), cv2.COLOR_BGR2GRAY)
(refCorners, refIds, refRejected) = detector.detectMarkers(refImage)
# creo bounding box de las dimensiones de la imagen de referencia
rect = np.array([[[0,0],
                  [refImage.shape[1],0],
                  [refImage.shape[1],refImage.shape[0]],
                  [0,refImage.shape[0]]]], dtype = "float32")


#FotoImport = '78f1c1f1.Foto_D100H060.142141.jpg'
FotoImport = 'fb484fd4.Foto_D100H060.154133.jpg'
resImageColor = cv2.cvtColor(cv2.imread(FotoImport), cv2.COLOR_BGR2RGB)
resImage = cv2.cvtColor(resImageColor, cv2.COLOR_BGR2GRAY)
############################################################
#### Detecto Aruco Markers y calculo homografia
############################################################
(res_corners, res_ids, _) = detector.detectMarkers(resImage)
# find which markers in frame match those in reference image
idx = which(refIds, res_ids)
# flatten the array of corners in the frame and reference image
these_res_corners = np.concatenate(res_corners, axis = 1)
these_ref_corners = np.concatenate([refCorners[x] for x in idx], axis = 1)
# estimate homography matrix
h, s = cv2.findHomography(these_ref_corners, these_res_corners, cv2.RANSAC, 5.0)
# transform the rectangle using the homography matrix
newRect = cv2.perspectiveTransform(rect, h)
# draw the rectangle on the frame
z = resImageColor *1
z = cv2.polylines(z, np.int32(newRect), True, (0,0,0), 10)
z = cv2.aruco.drawDetectedMarkers(z, res_corners, res_ids)
M = cv2.getPerspectiveTransform(newRect, rect)
# use cv2.warpPerspective() to warp your image to a top-down view
warped = cv2.warpPerspective(resImage, M, (refImage.shape[1], refImage.shape[0]), flags=cv2.INTER_LINEAR)
warpedColor = cv2.warpPerspective(resImageColor, M, (refImage.shape[1], refImage.shape[0]), flags=cv2.INTER_LINEAR)
############################################################
#### Binarizo
#### https://note.nkmk.me/en/python-numpy-opencv-image-binarization/
############################################################
area = warped.shape[0]*warped.shape[1]
th, recGrayThresOtsu = cv2.threshold(warped, 128, 1, cv2.THRESH_OTSU)
recGrayThresOtsuInv = recGrayThresOtsu == 0
recGrayThresOtsuInv = cv2.bitwise_or(recGrayThresOtsuInv*1,recGrayThresOtsuInv*1)
# tapo los aruco  markers
(warped_corners, warped_ids, _) = detector.detectMarkers(warped)
cv2.fillPoly(recGrayThresOtsuInv, pts=warped_corners[0].astype(int), color=(0, 0, 0))
cv2.fillPoly(recGrayThresOtsuInv, pts=warped_corners[1].astype(int), color=(0, 0, 0))
# tapo la escala de colores
anchoColores = int(warped_corners[0][0][1][0])*5
altoColores = int(warped_corners[0][0][3][1]*0.5)
warpedEscalaColores = np.array([[[0,0],[anchoColores,0],[anchoColores,altoColores],[0,altoColores]]], dtype=int)
cv2.fillPoly(recGrayThresOtsuInv, pts=warpedEscalaColores, color=(0, 0, 0))

# Calculo area con vegetación, perfil de altura y estadísticos
cobUmbral = recGrayThresOtsuInv.sum() / area
altura = np.mean(recGrayThresOtsuInv, axis=0)[150:-150] # calculo la altura del pasto para cada columna ELIMINO bordes por las dudas
alt_min = stats.describe(altura)[1][0]
alt_max = stats.describe(altura)[1][1]
alt_mean = stats.describe(altura)[2]
alt_var = stats.describe(altura)[3]
alt_skew = stats.describe(altura)[4]
alt_kurt = stats.describe(altura)[5]

# Tabla con datos
Tabla = pd.DataFrame({'Foto':FotoImport, 'cobUmbral':[cobUmbral],'alt_min':[alt_min],'alt_max':[alt_max],'alt_mean':[alt_mean],'alt_var':[alt_var],'alt_skew':[alt_skew],'alt_kurt_D100H130':[alt_kurt]})
Tabla

# Grafico imagen original, recorte y procesada
plt.subplot(1,3,1),plt.imshow(z,cmap = 'gray')
plt.subplot(1,3,2),plt.imshow(warpedColor,cmap = 'gray')
plt.subplot(1,3,3),plt.imshow(recGrayThresOtsuInv,cmap = 'gray')
plt.show()

# Grafico imagen procesada con linea de altura
y =  ((1-altura)*recGrayThresOtsuInv.shape[0]).tolist()
x = range(150, len(y)+150)
plt.plot(x, y, color="blue", linewidth=1.5) 
plt.imshow(recGrayThresOtsuInv,cmap = 'gray') 
plt.show() 
