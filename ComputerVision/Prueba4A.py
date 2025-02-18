import cv2
import numpy as np

# Cargamos las imágenes en escala de grises
img_object = cv2.imread("/Users/dani/Downloads/imagenes/starwars.jpg", cv2.IMREAD_GRAYSCALE)
img_scene  = cv2.imread("/Users/dani/Downloads/imagenes/habitacion_poster.jpg", cv2.IMREAD_GRAYSCALE)

if img_object is None or img_scene is None:
    print(" --(!) No se encontraron las imágenes")
    exit(0)

# ========================
# Paso 1 y 2: Detección y descripción de keypoints con AKAZE
# ========================
# Creamos el detector AKAZE usando create()
detector = cv2.AKAZE_create()

# Utilizamos detectAndCompute() para obtener los keypoints y descriptores en una sola llamada.
keypoints_object, descriptors_object = detector.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene   = detector.detectAndCompute(img_scene, None)

# (Opcional) Dibujar los keypoints detectados con drawKeypoints()
img_object_kp = cv2.drawKeypoints(img_object, keypoints_object, None, color=(0, 255, 0),
                                  flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
img_scene_kp  = cv2.drawKeypoints(img_scene, keypoints_scene, None, color=(0, 255, 0),
                                  flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# Puedes visualizar estas imágenes para comprobar la detección:
# cv2.imshow("Keypoints Objeto", img_object_kp)
# cv2.imshow("Keypoints Escena", img_scene_kp)
# cv2.waitKey(0)

# ========================
# Paso 3: Emparejar los descriptores
# ========================
# Usamos un matcher de fuerza bruta (BruteForce)
matcher = cv2.DescriptorMatcher_create("BruteForce")
matches = matcher.match(descriptors_object, descriptors_scene)

# ========================
# Paso 4: Filtrar los emparejamientos (good matches)
# ========================
# Calculamos la distancia mínima entre descriptores para filtrar
min_dist = min(match.distance for match in matches)
max_dist = max(match.distance for match in matches)
print("Min dist:", min_dist)
print("Max dist:", max_dist)

# Filtramos los emparejamientos: seleccionamos aquellos con distancia menor a 2*min_dist
good_matches = [m for m in matches if m.distance < 2 * min_dist]

# Dibujamos los matches filtrados
img_matches = cv2.drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
                              good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# ========================
# Paso 5: Calcular la homografía y localizar el objeto
# ========================
if len(good_matches) >= 4:
    # Extraer los puntos de los good matches
    obj_pts   = np.float32([ keypoints_object[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
    scene_pts = np.float32([ keypoints_scene[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

    # Calcular la homografía usando RANSAC
    H, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 3)
    print("Homografía:\n", H)

    # Tomar las esquinas de la imagen del objeto
    h, w = img_object.shape
    obj_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    # Proyectar las esquinas del objeto en la imagen de la escena
    scene_corners = cv2.perspectiveTransform(obj_corners, H)

    # Dado que drawMatches muestra ambas imágenes lado a lado,
    # desplazamos las coordenadas de la escena en el eje x (ancho de la imagen objeto)
    offset = np.float32([w, 0])
    scene_corners_offset = scene_corners + offset

    pt1 = tuple(np.int32(scene_corners_offset[0][0]))
    pt2 = tuple(np.int32(scene_corners_offset[1][0]))
    pt3 = tuple(np.int32(scene_corners_offset[2][0]))
    pt4 = tuple(np.int32(scene_corners_offset[3][0]))
    cv2.line(img_matches, pt1, pt2, (0, 255, 0), 4)
    cv2.line(img_matches, pt2, pt3, (0, 255, 0), 4)
    cv2.line(img_matches, pt3, pt4, (0, 255, 0), 4)
    cv2.line(img_matches, pt4, pt1, (0, 255, 0), 4)
    cv2.imshow("Emparejamientos & Detección", img_matches)
    cv2.imwrite("./imagenes/deteccion.jpg", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontraron suficientes good matches para calcular la homografía")

import cv2
import numpy as np

# Cargamos las imágenes en escala de grises
img_object = cv2.imread("/Users/dani/Downloads/imagenes/starwars.jpg", cv2.IMREAD_GRAYSCALE)
img_scene  = cv2.imread("/Users/dani/Downloads/imagenes/habitacion_poster.jpg", cv2.IMREAD_GRAYSCALE)

if img_object is None or img_scene is None:
    print(" --(!) No se encontraron las imágenes")
    exit(0)

# ========================
# Paso 1 y 2: Detección y descripción de keypoints con AKAZE
# ========================
detector = cv2.AKAZE_create()
keypoints_object, descriptors_object = detector.detectAndCompute(img_object, None)
keypoints_scene, descriptors_scene   = detector.detectAndCompute(img_scene, None)

# (Opcional) Dibujar los keypoints detectados
img_object_kp = cv2.drawKeypoints(img_object, keypoints_object, None, color=(0, 255, 0),
                                  flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
img_scene_kp  = cv2.drawKeypoints(img_scene, keypoints_scene, None, color=(0, 255, 0),
                                  flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# ========================
# Paso 3: Emparejar los descriptores
# ========================
# Utilizamos BFMatcher con NORM_HAMMING, ideal para descriptores binarios, y activamos crossCheck para obtener matches más robustos.
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptors_object, descriptors_scene)

# ========================
# Paso 4: Filtrar los emparejamientos (good matches)
# ========================
min_dist = min(match.distance for match in matches)
max_dist = max(match.distance for match in matches)
print("Min dist:", min_dist)
print("Max dist:", max_dist)

# Reducimos la sensibilidad usando un factor menor (por ejemplo, 1.5 en lugar de 2)
good_matches = [m for m in matches if m.distance < 1.5 * min_dist]

# Dibujar los matches filtrados
img_matches = cv2.drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
                              good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# ========================
# Paso 5: Calcular la homografía y localizar el objeto
# ========================
if len(good_matches) >= 4:
    # Extraer los puntos de los good matches
    obj_pts   = np.float32([ keypoints_object[m.queryIdx].pt for m in good_matches ]).reshape(-1, 1, 2)
    scene_pts = np.float32([ keypoints_scene[m.trainIdx].pt for m in good_matches ]).reshape(-1, 1, 2)

    # Calcular la homografía usando RANSAC
    H, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 3)
    print("Homografía:\n", H)

    # Tomar las esquinas de la imagen del objeto
    h, w = img_object.shape
    obj_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

    # Proyectar las esquinas del objeto en la imagen de la escena
    scene_corners = cv2.perspectiveTransform(obj_corners, H)

    # Dado que drawMatches muestra ambas imágenes lado a lado,
    # desplazamos las coordenadas de la escena en el eje x (ancho de la imagen objeto)
    offset = np.float32([w, 0])
    scene_corners_offset = scene_corners + offset

    # Convertir las coordenadas a enteros para dibujar las líneas
    pt1 = tuple(np.int32(scene_corners_offset[0][0]))
    pt2 = tuple(np.int32(scene_corners_offset[1][0]))
    pt3 = tuple(np.int32(scene_corners_offset[2][0]))
    pt4 = tuple(np.int32(scene_corners_offset[3][0]))

    cv2.line(img_matches, pt1, pt2, (0, 255, 0), 4)
    cv2.line(img_matches, pt2, pt3, (0, 255, 0), 4)
    cv2.line(img_matches, pt3, pt4, (0, 255, 0), 4)
    cv2.line(img_matches, pt4, pt1, (0, 255, 0), 4)

    cv2.imshow("Emparejamientos & Detección", img_matches)
    cv2.imwrite("./imagenes/deteccion.jpg", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontraron suficientes good matches para calcular la homografía")