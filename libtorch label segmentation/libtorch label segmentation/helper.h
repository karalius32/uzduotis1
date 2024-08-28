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

class helper
{
public:
	static torch::Tensor nms(const std::vector<std::vector<float>>& bounding_boxes, float threshold);
	static std::vector<torch::Tensor> NonMaxSuppression(torch::Tensor output0, torch::Tensor prototypes, float threshold_detection = 0.5, float threshold_iou = 0.5);
	static torch::Tensor CropImage(const torch::Tensor& image, const torch::Tensor& box);
	static torch::Tensor ThresholdImage(const torch::Tensor& image, float threshold = 0.1f);
	static std::tuple<std::vector<std::vector<torch::Tensor>>, std::vector<std::vector<int>>, std::vector<std::vector<float>>> GetSegmentationMasks(std::vector<torch::Tensor> l_class_NMS, torch::Tensor prototypes, std::vector<int64_t> imageSize);
	static void DrawRectangles(std::vector<torch::Tensor> l_class_NMS);
	static void DrawRectangles(std::vector<std::vector<std::vector<float>>> l_class);
	static void DrawResults(cv::Mat image, std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<torch::Tensor>>> results);
};

