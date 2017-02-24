#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

std::string remove_extension(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

int main() {
	std::string filename = "qrcode_gradient_edited.png";

    cv::Mat src = cv::imread(filename);
	cv::Mat hsv;
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	cv::Mat binary;
	cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(190, 190, 190), binary);
	binary = ~binary;

	imshow("Binary image", binary);
	cv::RNG rng;
	cv::imwrite(remove_extension(filename) + "_thresholded.jpg", binary);
	cv::waitKey(0);
}
