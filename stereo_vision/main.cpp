
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

	//imshow("Left", left);
	//imshow("Right", right);
	//waitKey(0);

	std::vector<Mat> costVolumeLeft;
	std::vector<Mat> costVolumeRight;
	int windowSize = 2;
	int maxDisp = 8;

	for (int disp = 0; disp <= maxDisp; disp++)
	{
		computeCostVolume(left, right, costVolumeLeft, costVolumeRight, windowSize, disp);
	}

	// Create empty gray-scale images
	Mat dispLeft(left.rows, left.cols, CV_16UC1, 0.0);
	Mat dispRight(right.rows, right.cols, CV_16UC1, 0.0);

	Mat dispLeft_vis(left.rows, left.cols, CV_8UC1, 0.0);
	Mat dispRight_vis(right.rows, right.cols, CV_8UC1, 0.0);

	selectDisparity(dispLeft, dispRight, costVolumeLeft, costVolumeRight);

	//convertScaleAbs(dispLeft, dispLeft_vis);
	//convertScaleAbs(dispRight, dispRight_vis);

	double min, max;
	minMaxLoc(dispLeft, &min, &max);
	dispLeft.convertTo(dispLeft_vis, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));
	minMaxLoc(dispRight, &min, &max);
	dispRight.convertTo(dispRight_vis, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));
	
	// display disparity maps
	imshow("dispLeft", dispLeft_vis);
	imshow("dispRight", dispRight_vis);
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

	//Mat dst_left; 
	//leftVolume.convertTo(dst_left, CV_8U);
	//imshow("Left Volume", dst_left);

	//Mat dst_right;
	//rightVolume.convertTo(dst_right, CV_8U);
	//imshow("Right Volume", dst_right);
	//waitKey(0);

	costVolumeLeft.push_back(leftVolume);
	costVolumeRight.push_back(rightVolume);

}

void compute_cost(cv::Mat &target, const cv::Mat &imgLeft, const cv::Mat &imgRight, int r, int c, int max_rows, int max_cols, int windowSize, int disp){

	int window_off = windowSize / 2;
	int channels = imgLeft.channels();
	
	Vec3b left_s = cv::Vec3b(0, 0, 0);
	Vec3b right_s = cv::Vec3b(0, 0, 0);

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
			int sample_c = c +(q_c - window_off);

			int sample_r_with_disp = r + (q_r_disp - window_off);
			int sample_c_with_disp = c + (q_c_disp - window_off);

			if (sample_r >= 0 && sample_r < max_rows && sample_c >= 0 && sample_c < max_cols)
				left_s = imgLeft.at<Vec3b>(sample_r, sample_c);

			if (sample_c_with_disp >= 0 && sample_c_with_disp < max_cols && sample_r_with_disp >= 0 && sample_r_with_disp < max_rows)
				right_s = imgRight.at<Vec3b>(sample_r_with_disp, sample_c_with_disp);

			for (int channel = 0; channel < channels; channel++)
			{

				cost += abs(left_s[channel] - right_s[channel]);
				//cost = left_s[channel];

			}

		}
	}

	target.at<unsigned short>(r, c) = cost;
	
	// Sanity Check
	//Vec3s t = target.at<Vec3s>(r, c);
	//int v = t[0];

}

void selectDisparity(Mat &dispLeft, Mat &dispRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight){
	
	int disparityScale = 9;
	//int disparityScale = 32; // good vor visualization
	const unsigned short MAX_INIT = 1000;
	unsigned short disparityPLeft = MAX_INIT; // cost valume has entries > 255
	unsigned short disparityPRight = MAX_INIT;
	unsigned short  costVolumeLeftXY = 0;
	unsigned short  costVolumeRightXY = 0;

	// loop through pixels
	for (int x = 0; x<dispLeft.rows; ++x) {
		for (int y = 0; y<dispLeft.cols; ++y) {

			// loop through disparity values
			for (int i = 0; i<costVolumeRight.size(); i++) {

				unsigned short valueLeft = costVolumeLeft.at(i).at<unsigned short>(x,y);
				//costVolumeLeftXY = static_cast<float>(valueLeft);
				costVolumeLeftXY = valueLeft;
				unsigned short valueRight = costVolumeRight.at(i).at<unsigned short>(x,y);
				//costVolumeRightXY = static_cast<float>(valueRight);
				costVolumeRightXY = valueRight;

				// minimize cost volumes
				if (costVolumeLeftXY < disparityPLeft) {
					disparityPLeft = costVolumeLeftXY;
				}
				if (costVolumeRightXY < disparityPRight) {
					disparityPRight = costVolumeRightXY;
				}
			}

			dispLeft.at<unsigned short>(x, y) = disparityPLeft*disparityScale;			//set pixel in desparity map
			dispRight.at<unsigned short>(x, y) = disparityPRight*disparityScale;			//set pixel in desparity map
			
			// reset comparison values for next pixel
			disparityPLeft = MAX_INIT;
			disparityPRight = MAX_INIT;
		}
	}
}