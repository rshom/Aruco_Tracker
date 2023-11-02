'''Rotation speed measurement

Look for a single aruco target and track angle and angular rate
'''

import numpy as np
import cv2 as cv
from cv2 import aruco

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


if __name__=='__main__':
    
    cap = cv.VideoCapture(0)
