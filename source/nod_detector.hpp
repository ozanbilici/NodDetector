/* Include guard ------------------------------------------------------------- */

#ifndef NOD_DETECTOR_HPP
#define NOD_DETECTOR_HPP

/* OpenCV Includes ----------------------------------------------------------- */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

/* Structure definitions ----------------------------------------------------- */

struct Coordinate
{
    int32_t x;
    int32_t y;
};

struct Vector
{
    double x;
    double y;
};

/* Enum class definitions ---------------------------------------------------- */

enum class NodType
{
    NONE,
    YES,
    NO
};

/* Class definitions --------------------------------------------------------- */

class NodDetector
{
    using Classifier = cv::CascadeClassifier;
public:
    NodDetector(cv::CascadeClassifier& classifier) :
        _classifier(classifier),
        _counter(0)
    { }

    NodType DetectNod(cv::Mat& frame);

private:
    bool DetectSingleFace(cv::Mat& frame);
    NodType CheckVariance();

    Vector CalculateMean();
    Vector CalculateVariance(Vector& mean);

    void ClearData();

    static constexpr uint32_t CAPTURE_LENGTH = 10;

    static constexpr uint32_t THRESHOLD_X = 400;
    static constexpr uint32_t THRESHOLD_Y = 300;

    Classifier& _classifier;
    Coordinate _coordinates[CAPTURE_LENGTH];

    uint32_t _counter;
};

#endif // NOD_DETECTOR_HPP
