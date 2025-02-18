import cv2
import numpy as np

img1 = cv2.imread("/Users/dani/Downloads/imagenes/campus1.jpg")
img2 = cv2.imread("/Users/dani/Downloads/imagenes/campus2.jpg")
img3 = cv2.imread("/Users/dani/Downloads/imagenes/campus3.jpg")

if img1 is None or img2 is None or img3 is None:
    print("Error al cargar las imágenes.")
    exit(0)

# Convertir a escala de grises (necesario para la detección de características)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

akaze = cv2.AKAZE_create()

# Detectar keypoints y calcular descriptores en cada imagen
kp1, des1 = akaze.detectAndCompute(gray1, None)
kp2, des2 = akaze.detectAndCompute(gray2, None)
kp3, des3 = akaze.detectAndCompute(gray3, None)

# ============================
# Paso 2: Emparejamiento de descriptores
# ============================
# Como AKAZE genera descriptores binarios, usamos BFMatcher con NORM_HAMMING
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches12 = bf.match(des1, des2)
matches23 = bf.match(des2, des3)

# Ordenar los matches por distancia (menor distancia = mejor match)
matches12 = sorted(matches12, key=lambda x: x.distance)
matches23 = sorted(matches23, key=lambda x: x.distance)

# ============================
# Paso 3: Calcular homografías
# ============================
# Homografía de img1 a img2
src_pts12 = np.float32([kp1[m.queryIdx].pt for m in matches12 ]).reshape(-1, 1, 2)
dst_pts12 = np.float32([kp2[m.trainIdx].pt for m in matches12 ]).reshape(-1, 1, 2)
H12, mask12 = cv2.findHomography(src_pts12, dst_pts12, cv2.RANSAC, 5.0)

# Homografía de img2 a img3
src_pts23 = np.float32([ kp2[m.queryIdx].pt for m in matches23 ]).reshape(-1, 1, 2)
dst_pts23 = np.float32([ kp3[m.trainIdx].pt for m in matches23 ]).reshape(-1, 1, 2)
H23, mask23 = cv2.findHomography(src_pts23, dst_pts23, cv2.RANSAC, 5.0)

# Usamos la imagen central (img2) como referencia.
# Para mapear img3 a la referencia, usamos la inversa de H23:
H32 = np.linalg.inv(H23)

# ============================
# Paso 4: Crear el panorama (definir el tamaño y la traslación)
# ============================
# Obtener dimensiones de cada imagen
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]
h3, w3 = img3.shape[:2]

# Esquinas de cada imagen
corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
corners3 = np.float32([[0, 0], [w3, 0], [w3, h3], [0, h3]]).reshape(-1, 1, 2)

# Transformar las esquinas al sistema de coordenadas de la imagen central (img2)
warped_corners1 = cv2.perspectiveTransform(corners1, H12)
warped_corners3 = cv2.perspectiveTransform(corners3, H32)

# Unir todas las esquinas para conocer la extensión total
all_corners = np.concatenate((warped_corners1, corners2, warped_corners3), axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# Calcular la traslación para que todas las imágenes queden en el lienzo
translation = [-x_min, -y_min]
panorama_size = (x_max - x_min, y_max - y_min)
T = np.array([[1, 0, translation[0]],
              [0, 1, translation[1]],
              [0, 0, 1]], dtype=np.float32)

# ============================
# Paso 5: Warp de cada imagen y fusión en el panorama
# ============================
# Warpear la imagen 1 al sistema de la imagen central
panorama = cv2.warpPerspective(img1, T.dot(H12), panorama_size)
# Fusionar la imagen central
panorama = cv2.warpPerspective(img2, T, panorama_size, dst=panorama, borderMode=cv2.BORDER_TRANSPARENT)
# Warpear y fusionar la imagen 3
panorama = cv2.warpPerspective(img3, T.dot(H32), panorama_size, dst=panorama, borderMode=cv2.BORDER_TRANSPARENT)

# Mostrar y guardar el resultado
cv2.imshow("Panorama", panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()