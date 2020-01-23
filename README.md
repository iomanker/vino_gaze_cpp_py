# Gaze Calibration Tool: openVINO Gaze wraps from C++ to Python
This tool, which is built on Raspberry Pi 4, wraps C++ code of GazeEstimation on openVINO and can be called in Python.
## Hardware Requirement
* Raspberry Pi
* Intel Neural Compute Stick 2
* Webcam
## Installation
* [Install OpenVINO™ toolkit for Raspbian* OS](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html)
* Make sure your openCV which is installed by yourself before would not interrupt openCV package in openVINO: `/opt/intel/openvino/opencv/lib/lib*.so.*`
* Clone this repo (this repo is edited from [opencv/open_model_zoo](https://github.com/opencv/open_model_zoo))
* Change directory to this repo folder: `cd ./vino_gaze_cpp_py`
* (optional)Modify library path: `./gaze_estimation/gaze_estimation_demo/main.cpp`
* Build: `./build_demos.sh`
* This built library might be located in `/home/<USERNAME>/omz_demos_build`
* Copy gaze library file: `cp /home/<USERNAME>/omz_demos_build/armv7l/Release/lib/libgaze_estimation_demo.so ./gazepy`
* **IF YOU WANT TO SEE HOW TO USE FUNCTIONS, SCROLL TO THE BOTTOM.**
## Calibration
* Install `sklearn` and `pyautogui`
* Run: `python3 ./gazepy/vino_gaze.py`
* Eyes look at a red point and enter space at nine times.
* [Demo Video](https://www.youtube.com/watch?v=jpyN-7Mz3jc)
## Test the result of calibration
* Run: `python3 ./gazepy/test_calibration.py`
* [Demo Video](https://www.youtube.com/watch?v=Q8h-14pjda0)
## Do it with Kafka
For more detail in sending gaze data with Kafka, please [click here](https://github.com/jimmYA-1995/Real-time-vehicle-alarm-system).

## Function
### Defined name
* `set_gaze_estimation_lib()`: load gaze class library and bind to python.
* `GAZELIB.gazeClass_py()`: get gaze class.
* `GAZELIB.exeEsti_py(GAZECLASS)`: do a estimation.
* `GAZELIB.get_gaze_*_py(GAZECLASS)`: get {x,y,z} gaze direction. (* stands for {x,y,z})
### Simple Code
```python=
from ctypes import *
from vino_gaze import set_gaze_estimation_lib()
if __name__ == "__main__":
    libgaze = set_gaze_estimation_lib()
    gazeclass = libgaze.gazeClass_py()
    while True:
        libgaze.exeEsti_py(gazeclass)
        x = libgaze.get_gaze_x_py(gazeclass)
        y = libgaze.get_gaze_y_py(gazeclass)
        z = libgaze.get_gaze_z_py(gazeclass)
```