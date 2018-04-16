#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""

@author: CatalinaBernal
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

#Ingresa por usuario . type = array
# (.py , .png , freq)

entra = sys.argv
imagen = entra[1]
freq = str(entra[2])

# carga la imagen con parametro ingresado por usuario
img = plt.imread(imagen)

#asigna las dimensiones de la imagen a las variables    
(y_alto, x_ancho, z_capas) = np.shape(img)
# Define transformada de Fourier para un array de datos: puede ser la matriz de gauss o la iamgen.
#recibe por parametro la matriz a transformar
def fourier(mat):
    dim = np.shape(mat)
    y = dim[0]
    x = dim[1]
    
    #condicional para usar con matriz de cualquier dimension
    if(len(dim)>2):
        z = dim[2] 
        fourier = np.zeros((y, x, z), dtype = complex)
    
        for cap in range(z):
            for j in range(y):
                for i in range(x):
                    suma = 0.0
            
                    for m in range(y):
                        for n in range(x):
                            pixel = mat[m][n][cap]
                            equis = (i*n)/(float(x))
                            ye = (j*m)/(float(y))
                            f = np.exp(-1j*2.0*np.pi*(equis + ye))

                            suma += f*pixel                    
    
                    fourier[j][i][cap] = suma
     
    else:
        fourier = np.zeros((y, x), dtype = complex)
        
        for j in range(y):
            for i in range(x):
                suma = 0.0
        
                for m in range(y):
                    for n in range(x):
                        pixel = mat[m][n]
                        equis = (i*n)/(float(x))
                        ye = (j*m)/(float(y))
                        f = np.exp(-1j*2.0*np.pi*(equis + ye))

                        suma += f*pixel                    
    
                
                fourier[j][i] = suma
    
    return fourier

# Define transformada inversa de fourier para un array de datos
# recibe por parametro la matriz a transformar    
def fourier_inv(mat):
    dim = np.shape(mat)
    y = dim[0]
    x = dim[1]
    
    if(len(dim)>2):
        z = dim[2]
        fourier_inv = np.zeros((y, x, z))
    
        for cap in range(z):
            for j in range(y):
                for i in range(x):
                    suma = 0.0
            
                    for m in range(y):
                        for n in range(x):
                            pixel = mat[m][n][cap]
                            equis = (i*n)/(float(x))
                            ye = (j*m)/(float(y))
                            f = np.exp(1j*2.0*np.pi*(equis + ye))

                        suma += f*pixel                    
    
                    fourier_inv[j][i][cap] = int((suma.real/float(x)/float(y)))
            
    
    else: 
        fourier_inv = np.zeros((y, x))
        for j in range(y):
            for i in range(x):
                suma = 0.0
        
                for m in range(y):
                    for n in range(x):
                        pixel = mat[m][n]
                        equis = (i*n)/(float(x))
                        ye = (j*m)/(float(y))
                        f = np.exp(1j*2.0*np.pi*(equis + ye))

                    suma += f*pixel                    
    
                fourier_inv[j][i] = (suma.real/float(x)/float(y))
            
    
    return fourier_inv

#elabora un array con las coordenadas x y y del centro de la imagen. Se utilizara para definir la gaussiana
def centro_img(x, y):
    p_x = 0.0
    p_y = 0.0
    
    if(x%2 != 0):
        p_x = (x/2) + 1
    else:
        p_x = x/2
        
    if(y%2 != 0):
        p_y = (y/2) + 1
    else:
        p_y = y/2
   
    centro = [p_x, p_y]
    return centro    


#Define la funcion que crea la matriz cuyo contenido son los puntos transformados
#recibe por parametro: c = una lista 2d 
# equivalente al centro de la imagen, n_pixel_kernel = ancho de gaussiana, m_y y n_x dimensiones de imagen.
    
def gauss(c, m_y , n_x, z):
    
    gauss = np.zeros((m_y, n_x, z))
    sig = 1.0
    # de acuerdo a la teoria de Anchura a media altura FWHM = 2.35482*sigma
    #sigma : desviacion estandar ; n_pix = FWHM
    
    #constante de normalizacion
    cons = 1.0/(2.0*np.pi*(sig**2))

    #Construye la matriz de valores "gaussianos".
    for cap in range(z):
        for j in range(m_y):
            for i in range(n_x):
                equis = ((i-c[0])**2)/float(2*(sig**2))
                ye = ((j-c[1])**2)/float(2*(sig**2))
    
                gauss[j][i][cap] = cons * np.exp(-(equis+ye))
            
    return gauss

 
#Este metodo permite el paso de frecuencias bajas. Recibe como parametro una matriz
def bajas(mat):
    dim = np.shape(mat)
    y = dim[0]
    x = dim[1]
    z = dim[2]
    
# cutoff y w son valores criticos definidos  en "Image Filtering in the Frequency Domain" 
#con base en ese documento se realiza el filtro de frecuencias
    cutoff = y_alto/7.0
    w = y_alto/50.0  
    
    for cap in range(z):
        for j in range(y):
            for i in range(x):
                if(mat[j][i][cap] < (cutoff - w)):
                    mat[j][i][cap] = 1
                    
                elif(mat[j][i][cap] > (cutoff + w)):
                    mat[j][i][cap] = 0
                    
                elif((mat[j][i][cap] > (cutoff -w)) and (mat[j][i][cap] < (cutoff + w))):
                    mat[j][i][cap] = 0.5*(1- np.sin(np.pi*(mat[j][i][cap] - cutoff)/(2.0*float(w))))
    return mat

#Este metodo permite el paso de frecuencias altas. Recibe como parametro una matriz
def altas(mat):
    dim = np.shape(mat)
    y = dim[0]
    x = dim[1]
    z = dim[2]
     
# cutoff y w son valores criticos definidos  en "Image Filtering in the Frequency Domain" 
#con base en ese documento se realiza el filtro de frecuencias
    cutoff = y_alto/7.0
    w = y_alto/50.0  
    
    for cap in range(z):
        for j in range(y):
            for i in range(x):
                if(mat[j][i][cap] < (cutoff - w)):
                    mat[j][i][cap] = 0
                    
                elif(mat[j][i][cap] > (cutoff + w)):
                    mat[j][i][cap] = 1
                    
                elif((mat[j][i][cap] > (cutoff -w)) and (mat[j][i][cap] < (cutoff + w))):
                    mat[j][i][cap] = 0.5*(1- np.sin(np.pi*(mat[j][i][cap] - cutoff)/(2.0*float(w))))
    return mat           
    

c = centro_img(x_ancho, y_alto)
g = gauss(c, y_alto, x_ancho, z_capas)
#Transf de fourier a la matriz gaussiana
f_G = fourier(g)

#Tranf de fourier a la imagen
f_Imag = fourier(img) 


if(freq == "bajo"):
#funcion de filtro de frec altas    
    bajo = bajas(f_Imag)
#convolucion entre nueva matriz y la gaussiana transformada (fourier)
    m_convol = f_G*bajo
#transformada invesa de fourier a la matriz de convolucion
    inv_convol = fourier_inv(m_convol)

#se guarda la matriz resultado como imagen filtrada
    plt.imshow(inv_convol)
    plt.savefig('bajas.png')    


if(freq == "alto"):
#funcion de filtro de frec altas    
    alto = altas(f_Imag)
#convolucion entre nueva matriz y la gaussiana transformada (fourier)
    m_convol = f_G*alto
#transformada invesa de fourier a la matriz de convolucion
    inv_convol = fourier_inv(m_convol)

#se guarda la matriz resultado como imagen filtrada
    plt.imshow(inv_convol)
    plt.savefig('altas.png')    
   