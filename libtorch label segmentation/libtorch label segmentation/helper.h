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

class helper
{
public:
	static torch::Tensor nms(const std::vector<std::vector<float>>& bounding_boxes, float threshold);
};

