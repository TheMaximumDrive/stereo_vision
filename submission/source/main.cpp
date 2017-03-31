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


	string left_path = "img/tsukuba_left.png";
	string right_path = "img/tsukuba_right.png";


	Mat left = imread(left_path);
	Mat right = imread(right_path);
	
	// Calculating the absolute difference
	Mat diff;
	absdiff(left, right, diff);

	// Rendering the absolute difference to screen
	imshow("Tsukuba: Absolute difference", diff);

	waitKey(0);

	return 0;
}