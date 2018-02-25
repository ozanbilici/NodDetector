/* Includes ------------------------------------------------------------------ */
#include <iostream>
#include <stdio.h>

/* OpenCV Includes ----------------------------------------------------------- */ 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

/* Heder Includes ------------------------------------------------------------ */

#include "nod_detector.hpp"

/* Public function declarations ---------------------------------------------- */

void PrintNodDetection(NodType type);

/* Main ---------------------------------------------------------------------- */

/**
 * @brief Main program
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char* argv[])
{
    std::string FACE_CASCADE_NAME = "data/haarcascade_frontalface.xml";
    cv::CascadeClassifier face_classifier;

    if(!face_classifier.load(FACE_CASCADE_NAME))
    {
    	std::cout << "Classifier couldn't be loaded" << std::endl;
    	return -1;
    };


    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cout << "Camera couldn't be opeend" << std::endl;
        return -1;
    }

    cv::Mat frame;
    NodDetector nod_detector(face_classifier);

    while(true)
    {
        if (!cap.read(frame))
            break;

        auto type = nod_detector.DetectNode(frame);
        PrintNodDetection(type);

        cv::imshow("window", frame);
        char key = cvWaitKey(50);
        
        if (key == 27)
        {
            break;
        }
    }

    return 0;
}

/* Public function definitions ----------------------------------------------- */

void PrintNodDetection(NodType type)
{
    if(type == NodType::YES)
    {
        std::cout << "Yes" << std::endl;
    } else if(type == NodType::NO)
    {
        std::cout << "No" << std::endl;
    }
}
