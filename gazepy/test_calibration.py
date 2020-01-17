from ctypes import *
import glob
import cv2
import csv
import numpy as np
import pyautogui as pag
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def set_gaze_estimation_lib():
    for lib in glob.glob("/usr/local/lib/libopencv*.so"):
        CDLL(lib, mode= RTLD_GLOBAL)
    libgaze = CDLL("libgaze_estimation_demo.so")
    libgaze.get_gaze_x_py.restype = c_float
    libgaze.get_gaze_y_py.restype = c_float
    libgaze.get_gaze_z_py.restype = c_float
    return libgaze

if __name__ == "__main__":
    cv2.namedWindow("WindowImg")
    screenSize = pag.size()
    circleRadius = 10
    circleColor = (0,0,255)


    libgaze = set_gaze_estimation_lib()
    gazeclass = libgaze.gazeClass_py()
    poly = PolynomialFeatures(2,include_bias=True, interaction_only=False)
    poly.fit(np.array([[0.5,0.5,0.5]]))
    regX = LinearRegression()
    regY = LinearRegression()
    itemX = None
    biasX = 0.0
    with open("test.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                itemX = [float(a) for a in row[0:-1]]
                itemX = np.array(itemX)
                biasX = float(row[-1])
            if line_count == 1:
                itemY = [float(a) for a in row[0:-1]]
                itemY = np.array(itemY)
                biasY = float(row[-1])
            line_count += 1
    regX.coef_ = itemX
    regY.coef_ = itemY
    regX.intercept_ = biasX
    regY.intercept_ = biasY
    while True:
        libgaze.exeEsti_py(gazeclass)
        gazeVec = [libgaze.get_gaze_x_py(gazeclass),libgaze.get_gaze_y_py(gazeclass),libgaze.get_gaze_z_py(gazeclass)]
        tmpX = regX.predict(poly.transform([gazeVec]))
        tmpY = regY.predict(poly.transform([gazeVec]))
        tmpX = np.clip(tmpX,0,screenSize[0])
        tmpY = np.clip(tmpY,0,screenSize[1])
        screenImg = np.zeros((screenSize[1],screenSize[0],3), np.uint8)
        screenImg.fill(255)
        screenImg = cv2.circle(screenImg, (tmpX,tmpY), circleRadius, circleColor, -1)
        cv2.imshow("WindowImg", screenImg)
        print("x: {}, y: {}".format(tmpX[0],tmpY[0]))
        tmpKb = cv2.waitKey(1)
        # q = 113, s = 115, space = 32
        if tmpKb == 113:
            break 

