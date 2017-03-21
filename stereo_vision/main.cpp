#include <stdio.h>
#include <opencv2\opencv.hpp>

using namespace cv;

int main() {
	// Task 1

	/*Mat img(500, 500, CV_8UC3, cv::Scalar(0, 255, 0));
	imshow("test", img);
	waitKey(0);*/
	
	// Task 2

	// Reading left and right images
	Mat left = imread("img/tsukuba_left.png");
	Mat right = imread("img/tsukuba_right.png");
	
	// Calculating the absolute difference
	Mat diff;
	absdiff(left, right, diff);

	// Rendering the absolute difference to screen
	imshow("Tsukuba: Absolute difference", diff);

	waitKey(0);

	return 0;
}