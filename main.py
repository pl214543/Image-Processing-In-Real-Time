# LINES NOTATED

# import required libraries
import cv2
import numpy as num

# import functions
from reader import rectangle, crop
from optimization import optimize
from lineDraw import contours

# video capture
video = cv2.VideoCapture("Timeline+1.mov")

# test case
print(video.isOpened())

# check if the video is opened
while video.isOpened():

    # gets the frame from the video to draw on
    booleanReady, frame = video.read()

    # retrieves the height and width of the frame for masking
    height, width = frame.shape[:2]

    pt_A = [-300, 600]
    pt_B = [-300, 900]
    pt_C = [-1080, 600]
    pt_D = [-1080, 900]

    # Here, I have used L2 norm. You can use L1 also.
    width_AD = num.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = num.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = num.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = num.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    # Locate points of the documents
    # or object which you want to transform

    inputpoints = num.float32([pt_A, pt_B, pt_C, pt_D])
    outputpoints = num.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])

# https://theailearner.com/tag/cv2-getperspectivetransform/

    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(inputpoints, outputpoints)
    result = cv2.warpPerspective(frame, matrix, (500, 600))

    # # creates a rectangle with background of 0s for masking
    # zerosRectangle = num.zeros((height, width), dtype="uint8")
    #
    # # defines coordinates for masking as percentage of camera view so compatible with any camera
    # topLeftX = int(width * 0.175)
    # topLeftY = int(height * 0.725)
    # bottomRightX = int(width * 0.85)
    # bottomRightY = int(height * 0.25)
    #
    # # creates list of those coordinates for drawing the rectangles and masking later, easy to use as parameters
    # varList = [topLeftX, topLeftY, bottomRightX, bottomRightY]

    # grayscale
    # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # uses the functions from the other files to optimize, make a rectangle, and crop
    # optimize will employ Gaussian Blur and Canny Edge Detection
    # optimized = optimize(grey)

    # cropped will create the mask
    # cropped = crop(optimized, zerosRectangle, varList)

    # calls the contours function to detect contours and draw
    # contoured = contours(cropped, frame)

    # addRectangle will draw the rectangle around the mask. put after others so contours doesn't detect the rectangle
    # addRectangle = rectangle(contoured, varList)

    # display the frame, creating a video
    cv2.imshow('Frame', frame)
    cv2.imshow('Transform', result)

    # closes the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# releases the video capture
video.release()

# closes the video
cv2.destroyAllWindows()
