# import required libraries
import cv2
import numpy as np

from optimization import optimize
from lineDraw import drawing
from haardetect import haar_detect1, haar_detect2

arrow = cv2.imread('kjhfd.png')

def perspective_transform(frame):
    
    # coordinates for perspective transform
    bl = [173, 346]
    tl = [266, 276]
    tr = [486, 276]
    br = [686, 341]

    # puts all coords in a list
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([[0, 0], [0, 480], [852, 0], [852, 480]])

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(frame, matrix, (852, 480))
    secmatrix = cv2.getPerspectiveTransform(pts2, pts1)
    warped = cv2.warpPerspective(result, secmatrix, (852, 480))

    # grayscale
    grey = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # uses the functions from the other files to optimize, make a rectangle, and crop
    # optimize will employ Gaussian Blur and Canny Edge Detection
    optimized = optimize(grey)

    # warped = haar_detect1(warped)
    # warped = haar_detect2(warped)

    # detect the lines using HoughLines
    lines = cv2.HoughLinesP(optimized, 0.5, np.pi / 180, 15, None, 40, 500)

    # draw the final lines
    result = drawing(warped, lines)

    final = cv2.addWeighted(frame, 1, warped, 0.7, 0)

    final = cv2.add(final, arrow)

    return final
