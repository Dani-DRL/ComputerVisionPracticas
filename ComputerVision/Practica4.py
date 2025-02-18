import cv2
import numpy as np

imagenPoster = cv2.imread("/Users/dani/Downloads/imagenes/starwars.jpg")
imagenRoom = cv2.imread("/Users/dani/Downloads/imagenes/habitacion_poster.jpg")

if imagenPoster is None or imagenRoom is None:
    print("-- Images not found --")
    exit(0)

detector = cv2.AKAZE_create()
keypointPoster, descriptorPoster = detector.detectAndCompute(imagenPoster, None)
keypointRoom, descriptorRoom = detector.detectAndCompute(imagenRoom, None)

imgPosterKP = cv2.drawKeypoints(imagenPoster, keypointPoster, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

imgRoomKP = cv2.drawKeypoints(imagenRoom, keypointRoom, None, color=(0, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )

matcher = cv2.DescriptorMatcher_create("BruteForce")
matches = matcher.match(descriptorPoster, descriptorRoom)

minDist = min(match.distance for match in matches)
maxDist = max(match.distance for match in matches)
print("\nMin value: " + str(minDist) + "\nMax value: " + str(maxDist))
goodMatches = [m for m in matches if m.distance < 2 * minDist]

imgMatches = cv2.drawMatches(imagenPoster, keypointPoster, imagenRoom, keypointRoom, goodMatches, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

if len(goodMatches) >= 4:
    posterPoints = np.float32([keypointPoster[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    # print("\n" + str(posterPoints) + "\n")
    roomPoints = np.float32([keypointRoom[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(posterPoints, roomPoints, cv2.RANSAC, 3)
    print("Homografia:\n", H)

    h, w = imagenPoster.shape[:2]
    posterCorners = np.float32([[0,0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    roomCorners = cv2.perspectiveTransform(posterCorners, H)

    offset = np.float32([w, 0])
    roomCorners_offset = roomCorners + offset

    pt1 = tuple(np.int32(roomCorners_offset[0][0]))
    pt2 = tuple(np.int32(roomCorners_offset[1][0]))
    pt3 = tuple(np.int32(roomCorners_offset[2][0]))
    pt4 = tuple(np.int32(roomCorners_offset[3][0]))

    cv2.line(imgMatches, pt1, pt2, (0, 255, 0), 4)
    cv2.line(imgMatches, pt2, pt3, (0, 255, 0), 4)
    cv2.line(imgMatches, pt3, pt4, (0, 255, 0), 4)
    cv2.line(imgMatches, pt4, pt1, (0, 255, 0), 4)
    cv2.imshow("Emparejamientos & Detección", imgMatches)
    cv2.imwrite("./imagenes/deteccion.jpg", imgMatches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No se encontraron suficientes good matches para calcular la homografía")