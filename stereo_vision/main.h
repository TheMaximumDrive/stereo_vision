#include <stdio.h>
#include <opencv2\opencv.hpp>


int ex_1();
int ex_2();

void computeCostVolume(const cv::Mat &imgLeft, const cv::Mat &imgRight, const cv::Mat &imgLeft_RGB, const cv::Mat &imgRight_RGB, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight,
	int windowSize, int disp);

void compute_cost(cv::Mat &target, const cv::Mat &imgLeft, const cv::Mat &imgRight, const cv::Mat &imgLeft_RGB, const cv::Mat &imgRight_RGB, int r, int c, int max_rows, int max_cols, int windowSize, int disp);

void selectDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int scaleDispFactor);

void selectDisparity_v2(cv::Mat &dispLeft, cv::Mat &dispRight, std::vector<cv::Mat> &costVolumeLeft, std::vector<cv::Mat> &costVolumeRight, int scaleDispFactor);

void refineDisparity(cv::Mat &dispLeft, cv::Mat &dispRight, int scaleDispFactor);