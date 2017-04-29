
#include "main.h"


using namespace cv;

int main() {

	//return ex_1();

	return ex_2();



}

int ex_1(){

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

int ex_2(){


	string left_path = "img/tsukuba_left.png";
	string right_path = "img/tsukuba_right.png";


	Mat left = imread(left_path); // channels are BGR
	Mat right = imread(right_path);

	std::vector<Mat> costVolumeLeft;
	std::vector<Mat> costVolumeRight;
	int windowSize = 5;
	int disp = 0;
	int maxDisp = 15;

	for (disp; disp <= maxDisp; disp++)
	{
		computeCostVolume(left, right, costVolumeLeft, costVolumeRight, windowSize, disp);
	}

	// Create empty gray-scale images
	Mat dispLeft(left.rows, left.cols, CV_8UC1);
	Mat dispRight(right.rows, right.cols, CV_8UC1);

	selectDisparity(dispLeft, dispRight, costVolumeLeft, costVolumeRight);
	
	// display disparity maps
	imshow("dispLeft", dispLeft);
	waitKey(0);
	imshow("dispRight", dispRight);
	waitKey(0);

	return 0;


}

void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight,
	int windowSize, int disp){

	int max_rows = imgLeft.rows;
	int max_cols = imgLeft.cols;

	Mat leftVolume(max_rows, max_cols, CV_16UC1, 0.0);
	Mat rightVolume(max_rows, max_cols, CV_16UC1, 0.0);

	for (int rows = 0; rows < max_rows; rows++)
	{
		for (int cols = 0; cols < max_cols; cols++)
		{

			compute_cost(leftVolume, imgLeft, imgRight, rows, cols, max_rows, max_cols, windowSize, disp);
			compute_cost(rightVolume, imgRight, imgLeft, rows, cols, max_rows, max_cols, windowSize, disp);

		}
	}

	// doesnt visualize well
	// bright patches are areas with data > 255 ( i think )

	Mat leftVolume_vis(max_rows, max_cols, CV_8UC1, 0.0);
	Mat rightVolume_vis(max_rows, max_cols, CV_8UC1, 0.0);

	/*convertScaleAbs(leftVolume, leftVolume_vis);
	imshow("Left Volume", leftVolume_vis);
	waitKey(0);



	convertScaleAbs(rightVolume, rightVolume_vis);
	imshow("Right Volume", rightVolume_vis);
	waitKey(0);*/

	costVolumeLeft.push_back(leftVolume);
	costVolumeRight.push_back(rightVolume);

}

void compute_cost(cv::Mat &target, const cv::Mat &imgLeft, const cv::Mat &imgRight, int r, int c, int max_rows, int max_cols, int windowSize, int disp){

	int window_off = windowSize / 2;
	int channels = imgLeft.channels();

	Scalar left_s = cv::Scalar(0, 0, 0,0);
	Scalar right_s = cv::Scalar(0, 0, 0,0);

	unsigned int cost = 0;

	for (int q_r = 0; q_r < windowSize; q_r++)
	{
		for (int q_c = 0; q_c < windowSize; q_c++)
		{

			// Calculate the logical index of the pixel and offset it according to the disparity arg
			int idx = q_c + q_r * windowSize;
			idx -= disp;

			int q_c_disp = idx % windowSize;
			int q_r_disp = floor(idx / windowSize);

			// Apply window offset i.e. window element 0 maps to a -2 offset
			int sample_r = r + (q_r - window_off);
			int sample_c = c + (q_c - window_off);

			int sample_r_with_disp = r + (q_r_disp - window_off);
			int sample_c_with_disp = c + (q_c_disp - window_off);

			if (sample_r >= 0 && sample_r < max_rows && sample_c >= 0 && sample_c < max_cols)
				left_s = imgLeft.at<uchar>(sample_r, sample_c);

			if (sample_c_with_disp >= 0 && sample_c_with_disp < max_cols && sample_r_with_disp >= 0 && sample_r_with_disp < max_rows)
				right_s = imgRight.at<uchar>(sample_r_with_disp, sample_c_with_disp);

			for (int channel = 0; channel < channels; channel++)
			{

				cost += abs(left_s[channel] - right_s[channel]);
			
			}

		}
	}

	target.at<unsigned short>(r, c) = cost;
	
	// Sanity Check
	//Scalar t = target.at<unsigned short>(r, c);
	//int v = t[0];

}

void selectDisparity(Mat &dispLeft, Mat &dispRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight){
	
	int disparityScale = 16;
	int disparityLeft = 0;
	int disparityRight = 0;
	float disparityLevelLeft = 255;
	float disparityLevelRight = 255;
	float costVolumeLeftXY = 0;
	float costVolumeRightXY = 0;

	// loop through pixels
	for (int x = 0; x<dispLeft.rows; ++x) {
		for (int y = 0; y<dispLeft.cols; ++y) {

			// loop through disparity values
			for (int i = 0; i<costVolumeRight.size(); i++) {

				unsigned short valueLeft = costVolumeLeft.at(i).at<unsigned short>(x,y);
				costVolumeLeftXY = static_cast<float>(valueLeft);
				unsigned short valueRight = costVolumeRight.at(i).at<unsigned short>(x,y);
				costVolumeRightXY = static_cast<float>(valueRight);

				// minimize cost volumes
				if (costVolumeLeftXY < disparityLevelLeft) {
					disparityLevelLeft = costVolumeLeftXY;
					disparityLeft = i;
				}
				if (costVolumeRightXY < disparityLevelRight) {
					disparityLevelRight = costVolumeRightXY;
					disparityRight = i;
				}
			}

			dispLeft.at<uchar>(x,y) = disparityLeft*disparityScale;			//set pixel in desparity map
			dispRight.at<uchar>(x,y) = disparityRight*disparityScale;			//set pixel in desparity map
			
			// reset comparison values for next pixel
			disparityLeft = 0;
			disparityRight = 0;
			disparityLevelLeft = 255;
			disparityLevelRight = 255;
		}
	}
}