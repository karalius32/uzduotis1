#pragma once
#include <torch/script.h> 
#include <torch/cuda.h>
#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using torch::indexing::Slice;
using torch::indexing::None;
using std::cout;
using std::endl;

class helper_new
{
public:
	static float generate_scale(cv::Mat& image, const std::vector<int>& target_size);
	static float letterbox(cv::Mat& input_image, cv::Mat& output_image, const std::vector<int>& target_size);
	static torch::Tensor xyxy2xywh(const torch::Tensor& x);
	static torch::Tensor xywh2xyxy(const torch::Tensor& x);
	static torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);
	static torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.25, float iou_thres = 0.45, int max_det = 300);
	static torch::Tensor clip_boxes(torch::Tensor& boxes, const std::vector<int>& shape);
	static torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape);
};

