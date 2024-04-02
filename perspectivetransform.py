# import required libraries
import cv2
import numpy as np

def perspective_transform(frame):
    bl = [173, 346]
    tl = [266, 276]
    tr = [486, 276]
    br = [686, 341]

    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)

    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [852, 0], [852, 480]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (852, 480))
    secmatrix = cv2.getPerspectiveTransform(pts2, pts1)
    warped = cv2.warpPerspective(result, secmatrix, (852, 480))

    final = cv2.addWeighted(frame, 1, warped, 0.7, 0)

    return final
