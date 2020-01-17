// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
* \brief The entry point for the Inference Engine gaze_estimation_demo application
* \file gaze_estimation_demo/main.cpp
* \example gaze_estimation_demo/main.cpp
*/
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core.hpp>

#include <inference_engine.hpp>

#include <samples/ocv_common.hpp>

#include "gaze_estimation_demo.hpp"

#include "face_inference_results.hpp"

#include "face_detector.hpp"

#include "base_estimator.hpp"
#include "head_pose_estimator.hpp"
#include "landmarks_estimator.hpp"
#include "gaze_estimator.hpp"

#include "utils.hpp"

#include <ie_iextension.h>
#ifdef WITH_EXTENSIONS
#include <ext_list.hpp>
#endif

using namespace InferenceEngine;
using namespace gaze_estimation;

// Estimation Struct
typedef struct GazeStruct{
    float gaze_x = 0.0f;
    float gaze_y = 0.0f;
    float gaze_z = 0.0f;
}GazeStruct;

class gazeClass{
    public:
    cv::VideoCapture cap;
    InferenceEngine::Core ie;

    std::string FLAGS_i = "cam";
    std::string FLAGS_m = "./lib/gaze-estimation-adas-0002.xml";
    std::string FLAGS_m_fd = "./lib/face-detection-retail-0004.xml";
    std::string FLAGS_m_hp = "./lib/head-pose-estimation-adas-0001.xml";
    std::string FLAGS_m_lm = "./lib/facial-landmarks-35-adas-0002.xml";
    std::string FLAGS_d = "MYRIAD";
    std::string FLAGS_d_fd = "MYRIAD";
    std::string FLAGS_d_hp = "MYRIAD";
    std::string FLAGS_d_lm = "MYRIAD";
    bool FLAGS_fd_reshape = false;
    double FLAGS_t = 0.5;

    FaceDetector faceDetector;

    HeadPoseEstimator headPoseEstimator;
    LandmarksEstimator landmarksEstimator;
    GazeEstimator gazeEstimator;
    
    std::vector<BaseEstimator*> estimators;
    std::vector<FaceInferenceResults> inferenceResults;
    cv::Mat frame;
    float gaze_x = 0.0, gaze_y = 0.0, gaze_z = 0.0;
    
    gazeClass():
        faceDetector(ie, FLAGS_m_fd, FLAGS_d_fd, FLAGS_t, FLAGS_fd_reshape),
        headPoseEstimator(ie, FLAGS_m_hp, FLAGS_d_hp),
        landmarksEstimator(ie, FLAGS_m_lm, FLAGS_d_lm),
        gazeEstimator(ie, FLAGS_m, FLAGS_d){
        estimators = {&headPoseEstimator, &landmarksEstimator, &gazeEstimator};
        if(!(FLAGS_i == "cam" ? cap.open(0) : cap.open(FLAGS_i))){
            printf("Cannot open input file or camera: %s", FLAGS_i.c_str());
        }
        if (!cap.read(frame)){
            printf("Failed to get frame from cv::VideoCapture");
        }
        std::vector<std::pair<std::string, std::string>> cmdOptions = {
            {FLAGS_d, FLAGS_m}, {FLAGS_d_fd, FLAGS_m_fd},
            {FLAGS_d_hp, FLAGS_m_hp}, {FLAGS_d_lm, FLAGS_m_lm}
        };
        initializeIEObject(ie, cmdOptions);
    }
    void exeEsti(){
        cap.read(frame);
        cv::flip(frame,frame,1);
        auto inferenceResults = faceDetector.detect(frame);
        for(auto& inferenceResult: inferenceResults){
            for(auto estimator: estimators){
                estimator->estimate(frame,inferenceResult);
            }
        }
        gaze_x = -2.0; gaze_y = -2.0; gaze_z = -2.0;
        for(auto& inferenceResult: inferenceResults){
            gaze_x = inferenceResult.gazeVector.x;
            gaze_y = inferenceResult.gazeVector.y;
            gaze_z = inferenceResult.gazeVector.z;
            break;
        }
    }
};

extern "C"{
    gazeClass* gazeClass_py(){return new gazeClass();}
    void exeEsti_py(gazeClass* myClass){
        myClass->exeEsti();
    }
    float get_gaze_x_py(gazeClass* myClass){
        return myClass->gaze_x;
    }
    float get_gaze_y_py(gazeClass* myClass){
        return myClass->gaze_y;
    }
    float get_gaze_z_py(gazeClass* myClass){
        return myClass->gaze_z;
    }
}


