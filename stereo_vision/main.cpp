
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

	Mat left_lab;
	Mat right_lab;

	cvtColor(left, left_lab, CV_RGB2Lab);
	cvtColor(right, right_lab, CV_RGB2Lab);

	//imshow("Left", left);
	//imshow("Right", right);
	//waitKey(0);

	std::vector<Mat> costVolumeLeft;
	std::vector<Mat> costVolumeRight;
	int windowSize = 6; // was 2 for submission ex2
	int maxDisp = 5; // was 8 for submission ex2
	//int scaleDispFactor = 9;
	int scaleDispFactor = 1; // good for visualization

	for (int disp = 0; disp <= maxDisp; disp++)
	{
		computeCostVolume(left_lab, right_lab,left,right, costVolumeLeft, costVolumeRight, windowSize, disp);
	}

	// Create empty gray-scale images
	//Mat dispLeft(left.rows, left.cols, CV_16UC1, 0.0);
	//Mat dispRight(right.rows, right.cols, CV_16UC1, 0.0);

	// EX3
	Mat dispLeft(left.rows, left.cols, CV_32FC1, 0.0);
	Mat dispRight(right.rows, right.cols, CV_32FC1, 0.0);

	Mat dispLeft_vis;
	Mat dispRight_vis;

	//selectDisparity(dispLeft, dispRight, costVolumeLeft, costVolumeRight, scaleDispFactor);
	// EX3
	selectDisparity_v2(dispLeft, dispRight, costVolumeLeft, costVolumeRight, scaleDispFactor);
	//refineDisparity(dispLeft, dispRight, scaleDispFactor);

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

void computeCostVolume(const Mat &imgLeft, const Mat &imgRight, const Mat &imgLeft_RGB, const Mat &imgRight_RGB, std::vector<Mat> &costVolumeLeft, std::vector<Mat> &costVolumeRight,
	int windowSize, int disp){

	int max_rows = imgLeft.rows;
	int max_cols = imgLeft.cols;

	//Mat leftVolume(max_rows, max_cols, CV_16UC1, 0.0);
	//Mat rightVolume(max_rows, max_cols, CV_16UC1, 0.0);

	// EX3
	Mat leftVolume(max_rows, max_cols, CV_32FC1, 1.0);
	Mat rightVolume(max_rows, max_cols, CV_32FC1, 1.0);

	for (int rows = 0; rows < max_rows; rows++)
	{
		for (int cols = 0; cols < max_cols; cols++)
		{

			compute_cost(leftVolume, imgLeft, imgRight, imgLeft_RGB, imgRight_RGB, rows, cols, max_rows, max_cols, windowSize, disp);
			compute_cost(rightVolume, imgRight, imgLeft,imgRight_RGB,imgLeft_RGB, rows, cols, max_rows, max_cols, windowSize, disp);

		}
	}

	// doesnt visualize well
	// bright patches are areas with data > 255 ( i think )

	//Mat dst_left; 
	//double min, max;
	//minMaxLoc(leftVolume, &min, &max);
	//leftVolume.convertTo(dst_left, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));
	//imshow("Left Volume", dst_left);

	//Mat dst_right;
	//minMaxLoc(rightVolume, &min, &max);
	//rightVolume.convertTo(dst_right, CV_8U, 255.0 / (max - min), -min * 255.0 / (max - min));
	//imshow("Right Volume", dst_right);
	//waitKey(0);

	costVolumeLeft.push_back(leftVolume);
	costVolumeRight.push_back(rightVolume);

}

float weight(cv::Vec3b color_l, cv::Vec3b color_r, int sample_r, int sample_c, int sample_r_with_disp, int sample_c_with_disp){

	float gamma_c = 7; // from paper
	float gamme_p = 20; 
	float k = 5;

	float delta_c = sqrt(pow((color_l[0] - color_r[0]), 2) + pow((color_l[1] - color_r[1]), 2) + pow((color_l[2] - color_r[2]), 2));
	float delta_g = sqrt(pow(sample_r - sample_r_with_disp, 2) + pow(sample_c - sample_c_with_disp, 2));

	float inner = -1*( (delta_c / gamma_c ) + (delta_g / gamme_p));

	float weight = k*expf(inner);

	return weight;


}

void compute_cost(cv::Mat &target, const cv::Mat &imgLeft, const cv::Mat &imgRight, const cv::Mat &imgLeft_RGB, const cv::Mat &imgRight_RGB, int r, int c, int max_rows, int max_cols, int windowSize, int disp){

	int window_off = windowSize / 2;
	int channels = imgLeft.channels();
	
	Vec3b left_p = cv::Vec3b(0, 0, 0);
	Vec3b right_q = cv::Vec3b(0, 0, 0);

	Vec3b left_q = cv::Vec3b(0, 0, 0);
	Vec3b right_p = cv::Vec3b(0, 0, 0);



	unsigned int cost = 0;
	float energy_top = 0;
	float energy_bottom = 0;

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
				left_p = imgLeft.at<Vec3b>(sample_r, sample_c);

			if (sample_r_with_disp >= 0 && sample_r_with_disp < max_rows && sample_c_with_disp >= 0 && sample_c_with_disp < max_cols)
				left_q = imgLeft.at<Vec3b>(sample_r_with_disp, sample_c_with_disp);

			if (sample_c >= 0 && sample_c < max_cols && sample_r >= 0 && sample_r < max_rows)
				right_p = imgRight.at<Vec3b>(sample_r, sample_c);

			if (sample_c_with_disp >= 0 && sample_c_with_disp < max_cols && sample_r_with_disp >= 0 && sample_r_with_disp < max_rows)
				right_q = imgRight.at<Vec3b>(sample_r_with_disp, sample_c_with_disp);

			for (int channel = 0; channel < channels; channel++)
			{
				cost += abs(left_p[channel] - right_q[channel]);
			}


			float w1 = weight(left_p, left_q, sample_r, sample_c, sample_r_with_disp, sample_c_with_disp);
			float w2 = weight(right_p, right_q, sample_r, sample_c, sample_r_with_disp, sample_c_with_disp);
			float e = 0;
			if (sample_r_with_disp >= 0 && sample_r_with_disp < max_rows && sample_c_with_disp >= 0 && sample_c_with_disp < max_cols){
				for (int channel = 0; channel < channels; channel++)
				{

					float v = imgLeft_RGB.at<Vec3b>(sample_r_with_disp, sample_c_with_disp)[channel]-imgRight_RGB.at<Vec3b>(sample_r_with_disp, sample_c_with_disp)[channel];
					e += v*v;
				}

			}

			e = sqrt(e);

			energy_top += w1*w2*e;
			
			energy_bottom += w1*w2;


		}
	}

	// EX3
	float e = energy_top / energy_bottom;
	target.at<float>(r, c) = e;

	//target.at<unsigned short>(r, c) = cost;

	
	// Sanity Check
	//Vec3s t = target.at<Vec3s>(r, c);
	//int v = t[0];

}

void selectDisparity(Mat &dispLeft, Mat &dispRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight, int scaleDispFactor){
	
	const unsigned short MAX_INIT = 1000;
	unsigned short disparityPLeft = MAX_INIT; // cost volume has entries > 255
	unsigned short disparityPRight = MAX_INIT;
	unsigned short  costVolumeLeftXY = 0;
	unsigned short  costVolumeRightXY = 0;

	// loop through pixels
	for (int x = 0; x<dispLeft.rows; ++x) {
		for (int y = 0; y<dispLeft.cols; ++y) {

			// loop through disparity values
			for (int i = 0; i<costVolumeRight.size(); i++) {

				unsigned short valueLeft = costVolumeLeft.at(i).at<unsigned short>(x,y);
				costVolumeLeftXY = valueLeft;
				unsigned short valueRight = costVolumeRight.at(i).at<unsigned short>(x,y);
				costVolumeRightXY = valueRight;

				// minimize cost volumes
				if (costVolumeLeftXY < disparityPLeft) {
					disparityPLeft = costVolumeLeftXY;
				}
				if (costVolumeRightXY < disparityPRight) {
					disparityPRight = costVolumeRightXY;
				}
			}

			dispLeft.at<unsigned short>(x, y) = disparityPLeft*scaleDispFactor;			//set pixel in desparity map
			dispRight.at<unsigned short>(x, y) = disparityPRight*scaleDispFactor;			//set pixel in desparity map
			
			// reset comparison values for next pixel
			disparityPLeft = MAX_INIT;
			disparityPRight = MAX_INIT;
		}
	}
}

void selectDisparity_v2(Mat &dispLeft, Mat &dispRight, vector<Mat> &costVolumeLeft, vector<Mat> &costVolumeRight, int scaleDispFactor){

	const float MAX_INIT = 2;
	float disparityCostLeft = MAX_INIT; // cost volume has entries > 255
	float disparityCostRight = MAX_INIT;
	int disparityLeft = 0;
	int disparityRight = 0;
	float costVolumeLeftXY = 0;
	float costVolumeRightXY = 0;

	// loop through pixels
	for (int x = 0; x<dispLeft.rows; ++x) {
		for (int y = 0; y<dispLeft.cols; ++y) {

			// loop through disparity values
			for (int i = 0; i<costVolumeRight.size(); i++) {

				float costVolumeLeftXY = costVolumeLeft.at(i).at<float>(x, y);
				float costVolumeRightXY = costVolumeRight.at(i).at<float>(x, y);

				// minimize cost volumes
				if (costVolumeLeftXY < disparityCostLeft) {
					disparityCostLeft = costVolumeLeft.at(i).at<float>(x, y);
					disparityLeft = i;
				}
				if (costVolumeRightXY < disparityCostRight) {
					disparityCostRight = costVolumeRight.at(i).at<float>(x, y);
					disparityRight = i;
				}
			}

			dispLeft.at<float>(x, y) = disparityLeft*scaleDispFactor;			//set pixel in desparity map
			dispRight.at<float>(x, y) = disparityRight*scaleDispFactor;			//set pixel in desparity map

			// reset comparison values for next pixel
			disparityCostLeft = MAX_INIT;
			disparityCostRight = MAX_INIT;
			disparityLeft = 0;
			disparityRight = 0;
		}
	}
}

void refineDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, int scaleDispFactor) {

	Mat dispLeftCopy(dispLeft.rows, dispLeft.cols, CV_32FC1, 0.0);
	Mat dispRightCopy(dispRight.rows, dispRight.cols, CV_32FC1, 0.0);
	
	// Mark inconsistent pixels
	for (int x = 0; x<dispLeft.rows; ++x) {
		for (int y = 0; y<dispLeft.cols; ++y) {
			dispLeftCopy.at<float>(x, y) = dispLeft.at<float>(x, y) / scaleDispFactor;			//set pixel in desparity map
			dispRightCopy.at<float>(x, y) = dispRight.at<float>(x, y) / scaleDispFactor;			//set pixel in desparity map

			if (abs(dispLeftCopy.at<float>(x, y) - dispRightCopy.at<float>(x, y)) > 1) {
				dispLeftCopy.at<float>(x, y) = FLT_MAX;
				dispRightCopy.at<float>(x, y) = FLT_MAX;
			}
		}
	}

	for (int x = 0; x < dispLeft.rows; ++x) {
		for (int y = 0; y < dispLeft.cols; ++y) {
			if (dispLeftCopy.at<float>(x, y) == FLT_MAX) {
				if (x == 0) {
					int i = x; 
					while(i < dispLeft.rows) {
						if (dispLeftCopy.at<float>(i, y) != FLT_MAX) {
							dispLeft.at<float>(x, y) = dispLeftCopy.at<float>(i, y) * scaleDispFactor;
							dispRight.at<float>(x, y) = dispRightCopy.at<float>(i, y) * scaleDispFactor;
							break;
						}
						i++;
					}
				}
				else if (x == dispLeft.rows) {
					int i = x;
					while (i >= 0) {
						if (dispLeftCopy.at<float>(i, y) != FLT_MAX) {
							dispLeft.at<float>(x, y) = dispLeftCopy.at<float>(i, y) * scaleDispFactor;
							dispRight.at<float>(x, y) = dispRightCopy.at<float>(i, y) * scaleDispFactor;
							break;
						}
						i--;
					}
				}
				else {
					int leftNeighbor;
					int rightNeighbor;
					int i = x;

					// Finding closest valid left and right neighbor's disparity
					while (i < dispLeft.rows) {
						if (dispLeftCopy.at<float>(i, y) != FLT_MAX) {
							rightNeighbor = i;
							break;
						}
						i++;
					}
					i = x;
					while (i >= 0) {
						if (dispLeftCopy.at<float>(i, y) != FLT_MAX) {
							leftNeighbor = i;
							break;
						}
						i--;
					}

					// Compare disparities and fill left disparity map
					if (dispLeftCopy.at<float>(leftNeighbor, y) < dispLeftCopy.at<float>(rightNeighbor, y)) {
						dispLeft.at<float>(x, y) = dispLeftCopy.at<float>(leftNeighbor, y) * scaleDispFactor;
					}
					else {
						dispLeft.at<float>(x, y) = dispLeftCopy.at<float>(rightNeighbor, y) * scaleDispFactor;
					}

					// Compare disparities and fill right disparity map
					if (dispRightCopy.at<float>(leftNeighbor, y) < dispRightCopy.at<float>(rightNeighbor, y)) {
						dispRight.at<float>(x, y) = dispRightCopy.at<float>(leftNeighbor, y) * scaleDispFactor;
					}
					else {
						dispRight.at<float>(x, y) = dispRightCopy.at<float>(rightNeighbor, y) * scaleDispFactor;
					}
				}
			}
		}
	}
}