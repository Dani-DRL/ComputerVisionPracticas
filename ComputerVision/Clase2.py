import cv2
import numpy as np
import matplotlib.pyplot as plt

#Cargo la imagen
imagen = cv2.imread('/Users/dani/Downloads/imagenes/ivvi_684x684.jpg')

#Compruebo que se haya cargado la imagen
if imagen is None:
    print("La imagen no se ha cargado correctamente")
    exit()

#Muestro la imagen original y cuando se pulse cualquier tecla destruyo la ventana
cv2.imshow('Imagen original', imagen)
cv2.waitKey()
cv2.destroyAllWindows()
#Separacion por canales
b_channel, g_channel, r_channel = cv2.split(imagen)

#Configuracion del histograma
histSize = 256  #Valores del 0 al 255
histRange = [0, 256] #Rango de intensidades
uniform = True
accumulate = False

#Calcular el histograma para cada canal
b_hist = cv2.calcHist([b_channel], [0], None, [histSize], histRange)
g_hist = cv2.calcHist([g_channel], [0], None, [histSize], histRange)
r_hist = cv2.calcHist([r_channel], [0], None, [histSize], histRange)


# 5. Graficar los histogramas de cada canal
plt.figure(figsize=(10, 6))
plt.title("Histograma de canales separados (B, G, R)")
plt.xlabel("Intensidad")
plt.ylabel("Número de píxeles")

# Graficamos cada histograma con su color correspondiente
plt.plot(b_hist, color='blue', label='Canal Azul')
plt.plot(g_hist, color='green', label='Canal Verde')
plt.plot(r_hist, color='red', label='Canal Rojo')
plt.xlim([0, histSize])
plt.legend()
plt.show()

print("\nAhora vamos a realizar cambios en la imagen en HSV:")
imagen_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
h_channel, s_channel, v_channel = cv2.split(imagen_hsv)
uHue = 0
uSaturation = 0
uValue = 0

while True:
    try:
        uHue = int(input("Introduce un valor para Hue entre 0 y 180: "))
        if 0 <= uHue <= 180:
            break
        else:
            print("\nIntroduce un valor valido")
    except ValueError:
        print("Introduce un valor valido entre 0 y 180")

while True:
    try:
        uSaturation = int(input("Introduce un valor para saturacion entre 0 y 255: "))
        if 0 <= uSaturation <= 255:
            break
        else:
            print("\nIntroduce un valor valido")
    except ValueError:
        print("Introduce un valor valido entre 0 y 255")

while True:
    try:
        uSaturation = int(input("Introduce un valor para value entre 0 y 255: "))
        if 0 <= uSaturation <= 255:
            break
        else:
            print("\nIntroduce un valor valido")
    except ValueError:
        print("Introduce un valor valido entre 0 y 255")

new_h = (h_channel.astype(np.int32) + uHue)%180
new_s = np.clip((s_channel.astype(np.int32) + uSaturation), 0, 255)
new_v = np.clip((v_channel.astype(np.int32) + uValue), 0, 255)

new_h = new_h.astype(np.uint8)
new_s = new_s.astype(np.uint8)
new_v = new_v.astype(np.uint8)

newImagenHSV = cv2.merge([new_h, new_v, new_s])

newImagenRGB = cv2.cvtColor(newImagenHSV, cv2.COLOR_HSV2RGB)

newImagenBGR = cv2.cvtColor(newImagenRGB, cv2.COLOR_RGB2BGR)
cv2.namedWindow("Imagen Modificada", cv2.WINDOW_NORMAL)
cv2.imshow("Imagen Modificada", newImagenBGR)
cv2.waitKey(0)
cv2.destroyAllWindows()