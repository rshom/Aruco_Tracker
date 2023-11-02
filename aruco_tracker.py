"""
Framework   : OpenCV Aruco
Description : Calibration of camera and using that for finding pose of multiple markers
Status      : Working
References  :
    1) https://docs.opencv.org/3.4.0/d5/dae/tutorial_aruco_detection.html
    2) https://docs.opencv.org/3.4.3/dc/dbb/tutorial_py_calibration.html
    3) https://docs.opencv.org/3.1.0/d5/dae/tutorial_aruco_detection.html
"""

import numpy as np
import cv2
from cv2 import aruco
import glob
import math

FONT = cv2.FONT_HERSHEY_SIMPLEX
ARUCO_ID = 1

def calibrate_from_images(fpath):
    ####---------------------- CALIBRATION ---------------------------
    # termination criteria for the iterative algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    # checkerboard of size (7 x 6) is used
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # iterating through all calibration images
    # in the folder
    images = glob.glob(fpath)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # find the chess board (calibration pattern) corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # if calibration pattern is found, add object points,
        # image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            # Refine the corners of the detected corners
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)


    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx,dist

def show_aruco_markers(N=20, size=400) -> None:
    for id in range(20):
        marker_image = aruco.generateImageMarker(aruco_dict, id, size)
        cv2.imshow("marker", marker_image)
        cv2.waitKey(0)


mtx,dist = calibrate_from_images(fpath='calib_images/checkerboard/*.jpg')
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
objPoints = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
# TODO: check if objPoints are right, it works but probably not for scale

parameters = aruco.DetectorParameters()
parameters.adaptiveThreshConstant = 10 # TODO: check constant
detector = aruco.ArucoDetector(aruco_dict, parameters)

def find_marker(frame, detector=detector, id=ARUCO_ID):
    '''Find the marker on the frame'''
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:     
        cv2.putText(frame, "No Ids", (5,64), FONT, 1, (0,255,0),2,cv2.LINE_AA)
        return frame, None
    
    strg = ''
    for i in range(0, ids.size):
        strg += str(ids[i][0])+', '
    cv2.putText(frame, "ids: " + strg, (5,64), FONT, 1, (0,255,0),2,cv2.LINE_AA)

    if ids[0][0] != 1:
        aruco.drawDetectedMarkers(frame, corners)
    
    return frame, corners[0]

def marker_angle(corner):
    valid, rvec, tvec = cv2.solvePnP(objPoints, corner, mtx, dist)
    cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, 1)

    rmat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rmat,tvec))
    euler_angles_degrees = -cv2.decomposeProjectionMatrix(P)[6]
    roll = euler_angles_degrees[2][0]
    return roll


from time import time
t = [time()]
roll = [0]

###------------------ ARUCO TRACKER ---------------------------
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    frame,corner = find_marker(frame)
    if corner is not None:        
        roll.append(marker_angle(corner))
        t.append(time())
        rate = (roll[-1]-roll[-2])/(t[-1]-t[-2])
        cv2.putText(frame, f"Roll: {roll[-1]:.2f} deg", 
                    (5,96), FONT, 1, (0,255,0),2,cv2.LINE_AA)
        cv2.putText(frame, f"Rate: {rate/360:.2f} rps", 
                    (5,96+32), FONT, 1, (0,255,0),2,cv2.LINE_AA)


    # display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


