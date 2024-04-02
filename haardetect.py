import cv2 as cv2
import numpy as np

# sets up the reference images
template1 = cv2.imread('6549486799_4c60e10e5c_b.jpg')
template2 = cv2.imread('gfddgfgfd.jpg')

# gets their shapes
h1, w1 = template1[:2]
h2, w2 = template2[:2]

def haar_detect1(frame):
    # gets required methods
    methods = [cv2.TM_CCOEFF, cv2.TM_CCORR_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF,
               cv2.TM_SQDIFF_NORMED]
    # iteration
    for method in methods:
        # searches for the template
        result = cv2.matchTemplate(frame, template1, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        bottom_right = (location[0] + w1, location[1] + h1)
        cv2.rectangle(frame, location, bottom_right, 255)

        return frame


def haar_detect2(frame):
    # gets required methods
    methods = [cv2.TM_CCOEFF, cv2.TM_CCORR_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF,
               cv2.TM_SQDIFF_NORMED]
    # iteration
    for method in methods:
        # looks for the provided arrows
        result = cv2.matchTemplate(frame, template1, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            location = min_loc
        else:
            location = max_loc

        bottom_right = (location[0] + w2, location[1] + h2)
        cv2.rectangle(frame, location, bottom_right, 255)

        return frame
