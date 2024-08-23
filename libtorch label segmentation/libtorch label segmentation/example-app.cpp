#include <torch/script.h> 
#include <torch/cuda.h>
#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "helper.h"

std::vector<cv::Mat> ReadImages(std::string folderPath);
std::vector<torch::Tensor> PreprocessImages(std::vector<cv::Mat> images, cv::Size modelInputSize, torch::DeviceType deviceType, int ch);
void Predict(std::vector<torch::Tensor> images, torch::jit::script::Module model);

int main() 
{
	const char* MODEL_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\best.torchscript";
	const char* IMAGES_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\images";

	std::vector<cv::Mat> images = ReadImages(IMAGES_PATH);
	std::vector<torch::Tensor> imageTensorsRGB = PreprocessImages(images, cv::Size(960, 960), at::kCUDA, 3);
	std::vector<torch::Tensor> imageTensorsGRAY = PreprocessImages(images, cv::Size(960, 960), at::kCUDA, 1);

	torch::jit::script::Module model;
	model = torch::jit::load(MODEL_PATH, at::kCUDA);
	model.eval();

	Predict(imageTensorsRGB, model);

	return 0;
}

std::vector<cv::Mat> ReadImages(std::string folderPath)
{
	std::vector<std::string> imageNames;
	cv::glob(folderPath + "/*.*", imageNames);

	std::vector<cv::Mat> images;
	for (std::string imageName : imageNames)
	{
		cv::Mat inputImage = cv::imread(imageName);
		images.push_back(inputImage);
	}

	return images;
}

std::vector<torch::Tensor> PreprocessImages(std::vector<cv::Mat> images, cv::Size modelInputSize, torch::DeviceType deviceType, int ch)
{
	int colorConversion = cv::COLOR_BGR2GRAY;
	int colorType = CV_32FC1;
	if (ch == 3)
	{
		colorConversion = cv::COLOR_BGR2RGB;
		colorType = CV_32FC3;
	}

	// Preparing images for model use
	std::vector<torch::Tensor> imageTensors;
	for (cv::Mat image : images)
	{
		cv::resize(image, image, modelInputSize); // resizing
		cv::cvtColor(image, image, colorConversion); // converting color
		image.convertTo(image, colorType, 1.0f / 255.0f); // scaling
		torch::Tensor imageTensor = torch::from_blob(image.data, { 1, modelInputSize.width, modelInputSize.height, ch }).to(deviceType); // converting to tensor
		imageTensor = imageTensor.permute({ 0, 3, 1, 2 }).contiguous();
		std::cout << imageTensor.sizes() << std::endl;
		imageTensors.push_back(imageTensor);
	}

	return imageTensors;
}

void PreprocessOneOutput(torch::Tensor output0, torch::Tensor prototypes)
{
	int nb_class = output0.size(0) - 4 - prototypes.size(0);
	std::vector<std::vector<std::vector<float>>> l_class(nb_class);
	torch::Tensor output_0_T = output0.transpose(0, 1);
	float threshold_detection = 0.8;
	float threshold_iou = 0.5;

	for (int i = 0; i < output_0_T.size(0); i++) 
	{
		torch::Tensor detection = output_0_T[i];

		torch::Tensor conf = detection.slice(0, 4, nb_class + 4);
		torch::Tensor max_conv = torch::max(conf);
		int argmax_conv = torch::argmax(conf).item<int>();

		if (max_conv.item<float>() > threshold_detection) 
		{
			std::vector<float> combined;
			auto detection_head = detection.slice(0, 0, 4).to(torch::kCPU).data_ptr<float>();
			auto max_conv_val = max_conv.item<float>();
			auto detection_tail = detection.slice(0, 4 + nb_class).to(torch::kCPU).data_ptr<float>();

			combined.insert(combined.end(), detection_head, detection_head + 4);
			combined.push_back(max_conv_val);
			combined.insert(combined.end(), detection_tail, detection_tail + detection.size(0) - (4 + nb_class));

			l_class[argmax_conv].emplace_back(combined);
		}
	}

	std::vector<torch::Tensor> l_class_NMS;
	for (const auto& clas : l_class) {
		l_class_NMS.push_back(helper::nms(clas, threshold_iou));
	}

	std::cout << l_class_NMS.size() << std::endl;
	std::cout << l_class_NMS[0].sizes() << std::endl;
}

void Predict(std::vector<torch::Tensor> imageTensors, torch::jit::script::Module model)
{
	// Prepare the tensors for model input
	torch::Tensor input = torch::cat(imageTensors, 0);
	std::cout << input.sizes() << std::endl;
	std::vector<torch::jit::IValue> inputs{ input };
	
	// Forward pass
	torch::jit::IValue output = model.forward(inputs);

	// Process the output
	torch::Tensor output_0_all = output.toTuple()->elements()[0].toTensor().to(at::kCPU);
	torch::Tensor prototypes_all = output.toTuple()->elements()[1].toTensor().to(at::kCPU);

	PreprocessOneOutput(output_0_all[0], prototypes_all[1]);
}

