# import required libraries
import cv2
import numpy as num
from parallelCheck import findParallels


# function for drawing the straight lines
def drawing(frame, lines):
    # sets up the variables for drawing mid line
    averageStartX = 0
    averageStartY = 0
    averageEndX = 0
    averageEndY = 0

    # checks if anything is in the list of lines (do any lines exist) so no errors appear
    if lines is not None:

        # gets all parallel lines
        parallels, nonparallels = findParallels(lines)

        # checks to make sure that parallels are at least detected
        if len(parallels) != 0:

            # iterates through
            for parLines in parallels:

                # iterates through
                for line in parLines:
                    # adds x and y to the average storer
                    averageStartX += line[0][0]
                    averageEndX += line[0][2]
                    averageStartY += line[0][1]
                    averageEndY += line[0][3]

                    # gets the average
                    averageStartX = averageStartX / 2
                    averageEndX = averageEndX / 2
                    averageStartY = averageStartY / 2
                    averageEndY = averageEndY / 2

                    averageStartX = int(averageStartX)
                    averageEndX = int(averageEndX)
                    averageStartY = int(averageStartY)
                    averageEndY = int(averageEndY)

        # iterates through parallels
        for parLines in parallels:
            for line in parLines:
                # unpacks
                x1, y1, x2, y2 = line[0]

                # draws lines
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.line(frame, (averageStartX, averageStartY), (averageEndX, averageEndY), (0, 255, 0), 2)

        # draws nonparallels - MAY REMOVE
        for nonParLines in nonparallels:
            for line in nonParLines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # return frame
    return frame
