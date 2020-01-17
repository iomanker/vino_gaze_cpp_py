from ctypes import *
import glob
import cv2
import csv
import numpy as np
import pyautogui as pag # To get screen size
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def set_gaze_estimation_lib():
    for lib in glob.glob("/usr/local/lib/libopencv*.so"):
        CDLL(lib, mode= RTLD_GLOBAL)
    libgaze = CDLL("libgaze_estimation_demo.so")
    libgaze.get_gaze_x_py.restype = c_float
    libgaze.get_gaze_y_py.restype = c_float
    libgaze.get_gaze_z_py.restype = c_float
    return libgaze

class calibration():
    def __init__(self):
        super(calibration, self).__init__()
        self.poly = PolynomialFeatures(2,include_bias=True, interaction_only=False)
        self.regX = LinearRegression()
        self.regY = LinearRegression()


    def get_params(self,reg):
        return reg.coef_
    def get_bias(self,reg):
        return reg.intercept_


    def set_poly(self,X):
        self.polyFeats = self.poly.fit_transform(X)
    def fit(self,x_Y,y_Y):
        self.regX.fit(self.polyFeats,x_Y)
        self.regY.fit(self.polyFeats,y_Y)
    def transform(self,Y):
        return self.reg.transform(Y)


    def save_as_csv(self,filename):
        with open(filename, mode='w') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            # for item in np.hstack((self.paramsY.reshape(9,1),self.params.reshape(9,10))):
            #     spamwriter.writerow(list(item))
            paramsX = list(self.get_params(self.regX))
            paramsX.append(self.get_bias(self.regX))
            paramsY = list(self.get_params(self.regY))
            paramsY.append(self.get_bias(self.regY))
            spamwriter.writerow(paramsX)
            spamwriter.writerow(paramsY)


if __name__ == "__main__":
    cv2.namedWindow("WindowImg")
    libgaze = set_gaze_estimation_lib()
    gazeclass = libgaze.gazeClass_py()

    # Width,Height
    screenSize = pag.size()

    circleRadius = 10
    circleColor = (0,0,255)

    gazePoints = []
    gazeScreenX = []
    gazeScreenY = []
    # 9 points
    tmpKb = 0
    for screen_y in [circleRadius+40,int(screenSize[1]/2),screenSize[1] - circleRadius-40]:
        for screen_x in [circleRadius+40,int(screenSize[0]/2),screenSize[0] - circleRadius-40]:
            screenImg = np.zeros((screenSize[1],screenSize[0],3), np.uint8)
            screenImg.fill(255)
            screenImg = cv2.circle(screenImg, (screen_x,screen_y), circleRadius, circleColor, -1)
            cv2.imshow("WindowImg", screenImg)
            while True:
                libgaze.exeEsti_py(gazeclass)
                tmpKb = cv2.waitKey(20)
                # q = 113, s = 115, space = 32
                if tmpKb == 32: 
                    gazeScreenX.append(screen_x)
                    gazeScreenY.append(screen_y)
                    gazePoints.append([libgaze.get_gaze_x_py(gazeclass),
                                       libgaze.get_gaze_y_py(gazeclass),
                                       libgaze.get_gaze_z_py(gazeclass)])
                    break
                if tmpKb == 113:
                    break
            if tmpKb == 113:
                break
        if tmpKb == 113:
            break
    if len(gazePoints) == 9:
        gazeCali = calibration()
        X = np.array(gazePoints)
        screen_x = np.array(gazeScreenX)
        screen_y = np.array(gazeScreenY)
        gazeCali.set_poly(X)
        gazeCali.fit(screen_x,screen_y)
        gazeCali.save_as_csv("test.csv")
    else:
        print("Points is not enough.")
