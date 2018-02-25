/* Header Includes ----------------------------------------------------------- */ 

#include "nod_detector.hpp"

/* Public member functions --------------------------------------------------- */ 

/**
 * @brief Detect node
 * 
 * First try to detect a single face(multiple face detection is not supported), 
 * then when there is enough data, calculate variance. When the variance is greater
 * than threshold values, interpret it as node detection. 
 * 
 * @param frame 
 * @return NodeType 
 */
NodType NodDetector::DetectNod(cv::Mat& frame)
{
    NodType type = NodType::NONE;

    DetectSingleFace(frame);
    
    if(_counter == CAPTURE_LENGTH)
    {
        type = CheckVariance();
        ClearData();
    }

    return type;
}

/* Private member functions -------------------------------------------------- */ 

/**
 * @brief Find single face by using OpenCV predefined classifier
 * 
 * @param frame 
 * @return true 
 * @return false 
 */
bool NodDetector::DetectSingleFace(cv::Mat& frame)
{
    std::vector<cv::Rect> faces;
    cv::Mat frame_gray;

    cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(frame_gray, frame_gray);

    _classifier.detectMultiScale(frame_gray, faces, 1.1, 2, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(24, 24));

    if(faces.size() == 1)
    {
        cv::rectangle(frame, faces[0], cv::Scalar(0,0,255));
        _coordinates[_counter++] = {faces[0].x, faces[0].y};

        return true;
    }

    return false;
}

/**
 * @brief Check whether variance in X and Y axises are greater than 
 * threshold values or not.
 * 
 * @return NodeType 
 */
NodType NodDetector::CheckVariance()
{
    Vector mean = CalculateMean();
    Vector variance = CalculateVariance(mean);

    if(variance.x > THRESHOLD_X)
    {
        return NodType::NO;
    }

    if(variance.y > THRESHOLD_Y)
    {
        return NodType::YES;
    }

    return NodType::NONE;
}

/**
 * @brief Calculate mean of captured data
 * 
 * @return Vector 
 */
Vector NodDetector::CalculateMean()
{
    Vector mean;

    for(int i = 0; i < CAPTURE_LENGTH; ++i)
    {
        mean.x += _coordinates[i].x;
        mean.y += _coordinates[i].y ;
    }

    mean.x /= CAPTURE_LENGTH;
    mean.y /= CAPTURE_LENGTH;

    return mean;
}

/**
 * @brief Calculate variance of captured data
 * 
 * @param mean 
 * @return Vector 
 */
Vector NodDetector::CalculateVariance(Vector& mean)
{
    Vector number;
    Vector variance;

    for(int i = 0; i < CAPTURE_LENGTH; ++i)
    {
        variance.x += (_coordinates[i].x-mean.x) * (_coordinates[i].x - mean.x);
        variance.y += (_coordinates[i].y-mean.y) * (_coordinates[i].y - mean.y);
    }

    variance.x /= (CAPTURE_LENGTH-1);
    variance.y /= (CAPTURE_LENGTH-1);

    return variance;
}

/**
 * @brief Clear all the captured data
 */
void NodDetector::ClearData()
{
    for(int i = 0; i < CAPTURE_LENGTH; ++i)
    {
        _coordinates[i].x = 0;
        _coordinates[i].y = 0;
    }

    _counter = 0;
}
