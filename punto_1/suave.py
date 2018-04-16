#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: CatalinaBernal
"""
import numpy as np
import matplotlib.pyplot as plt
import sys

#Ingresa por usuario . type = array
# (.py , .png , n_pixel_kernel)

#entra = sys.argv
#imagen = str(entra[1])
#n_pixel_kernel = entra[2]

print "n_pixel_kernel debe ser modificado desde archivo.py"
#carga y lectura de imagen

img = plt.imread('imagen.png')
n_pixel_kernel = 3

#asigna las dimensiones de la imagen a las variables    
(y_alto, x_ancho, z_capas) = np.shape(img)

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
    
def gauss(c, n_pixel_kernel, m_y , n_x, z):
    
    gauss = np.zeros((m_y, n_x, z))
    
    # de acuerdo a la teoria de Anchura a media altura FWHM = 2.35482*sigma
    #sigma : desviacion estandar ; n_pix = FWHM
    sig = float(n_pixel_kernel)/2.35482    
    
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

# Organiza la matriz resultante de la convolucion (la matriz que se
# debe obtener es simetrica). Recibe como parametro una matriz y un vector 
# con el centro de la imagen.
#
#def  org_mat(mat, centro):
#    
#    dim = np.shape(mat)
#    y = dim[0]
#    x = dim[1]
#    z = dim[2]
#    p_x = centro[0]
#    p_y = centro[1]
#    #estos centros han sido desviados para los casos en los que 
#    # la matriz posee dimensiones impares
#    px = centro[0] - 1
#    py = centro[1] - 1
#    
#    new = np.copy(mat)
#    old = np.copy(mat)
#    
#    for cap in range(z):
#        x1 = old[:p_y, :p_x, cap]
#        x2 = old[:p_y, p_x:x, cap]
#        x3 = old[p_y:y, :p_x, cap]
#        x4 = old[p_y:y, p_x:x, cap]
#    
#        if(x%2 == 0):
#            if(y%2 == 0):
#                new[p_y:y, p_x:x, cap] = x1
#                new[:p_y, p_x:x, cap] = x3
#                new[p_y:y, :p_x, cap] = x2
#                new[:p_y, :p_x, cap] = x4
#                    
#            else:
#                new[py:y, p_x:x, cap] = x1
#                new[:py, p_x:x, cap] = x3
#                new[py:y, :p_x, cap] = x2
#                new[:py, :p_x, cap] = x4
#            
#        elif(x%2 !=0):
#            if(y%2==0):
#                new[p_y:y, px:x, cap] = x1
#                new[:p_y, px:x, cap] = x3
#                new[p_y:y, :px, cap] = x2
#                new[:p_y, :px, cap] = x4
#            
#            else:
#                new[py:y, px:x, cap] = x1
#                new[:py, px:x, cap] = x3
#                new[py:y, :px, cap] = x2
#                new[:py, :px, cap] = x4
#            
#    return new


c = centro_img(x_ancho, y_alto)
g = gauss(c, n_pixel_kernel, y_alto, x_ancho, z_capas)
 #Transf de fourier a la matriz gaussiana
f_G = fourier(g)
#Tranf de fourier a la imagen
f_Imag = fourier(img)

#Realiza la convolucion de la transformada de la matriz gaussiana y
# la transformada de la imagen (Fourier) 
m_convol = f_G*f_Imag
inv_convol = fourier_inv(m_convol)
     
#organiza las matrices de cada capa de la imagen
#smooth = org_mat(inv_convol, c)

#muestra y guarda la imagen
plt.imshow(inv_convol)
plt.savefig('suave.png')         
            
        
            
            

            