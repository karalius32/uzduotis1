#include <torch/script.h> 
#include <torch/cuda.h>
#include <torch/torch.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <chrono>
#include <string>
#include "helper.h"
#include "helper_new.h"

#define YOLO 0
#define UNET 1
#define DEEPLAB 2

using std::cout;
using std::endl;


void TestModel(const char* path, std::vector<cv::Mat> images, std::string name, int modelType);
std::vector<cv::Mat> ReadImages(std::string folderPath, cv::Size modelInputSize);
std::vector<torch::Tensor> PreprocessImages(std::vector<cv::Mat> images, torch::DeviceType deviceType, int ch);
void Predict_YOLO(std::vector<cv::Mat> images, torch::jit::script::Module model);
void Predict_UNET(std::vector<cv::Mat> images, torch::jit::script::Module model);
void Predict_Deeplab(std::vector<cv::Mat> images, torch::jit::script::Module model);
std::vector<std::vector<cv::Mat>> TorchUnet_PredictMasks(std::vector<cv::Mat> images);
void Predict_Deeplab(std::vector<cv::Mat> images, torch::jit::script::Module model);

int main() 
{
	const char* MODEL_YOLOV8N_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\best_n.torchscript";
	const char* MODEL_YOLOV8S_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\best_s.torchscript";
	const char* MODEL_UNET_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\unet_exported.torchscript";
	const char* MODEL_DEEPLAB_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\deeplab_exported.torchscript";
	const char* IMAGES_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\images";

	std::vector<cv::Mat> images = ReadImages(IMAGES_PATH, cv::Size(960, 960));

	//TestModel(MODEL_YOLOV8N_PATH, images, "yolov8n (3.4M params): ", YOLO);
	//TestModel(MODEL_YOLOV8S_PATH, images, "yolov8s (11.8M params): ", YOLO);
	TestModel(MODEL_UNET_PATH, images, "UNET (1M params): ", UNET);
	TestModel(MODEL_DEEPLAB_PATH, images, "DeeplabV3 (11M params): ", DEEPLAB);


	return 0;
}

void TestModel(const char* path, std::vector<cv::Mat> images, std::string name, int modelType)
{
	torch::jit::script::Module model;
	model = torch::jit::load(path, at::kCUDA);
	model.eval();

	cout << name << endl;
	for (int i = 0; i < 10; i++)
	{
		cout << i + 1 << ": " << endl;
		switch (modelType) 
		{
			case YOLO:
				Predict_YOLO(images, model);
				break;
			case UNET:
				Predict_UNET(images, model);
				break;
			case DEEPLAB:
				Predict_Deeplab(images, model);
				break;
		}
		cout << "--------------------" << endl;
	}

	cout << endl;

	model.to(at::kCPU);
}

std::vector<cv::Mat> ReadImages(std::string folderPath, cv::Size modelInputSize)
{
	std::vector<std::string> imageNames;
	cv::glob(folderPath + "/*.*", imageNames);

	std::vector<cv::Mat> images;
	for (std::string imageName : imageNames)
	{
		cv::Mat inputImage = cv::imread(imageName);
		cv::resize(inputImage, inputImage, modelInputSize); // resizing
		images.push_back(inputImage);
	}

	return images;
}

std::vector<torch::Tensor> PreprocessImages(std::vector<cv::Mat> images, torch::DeviceType deviceType, int ch)
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
		cv::cvtColor(image, image, colorConversion); // converting color
		torch::Tensor imageTensor = torch::from_blob(image.data, { image.rows, image.cols, ch }, torch::kByte).to(deviceType); // converting to tensor
		imageTensor = imageTensor.toType(torch::kFloat32).div(255);
		imageTensor = imageTensor.permute({ 2, 0, 1 });
		imageTensor = imageTensor.unsqueeze(0);
		imageTensor = imageTensor.contiguous();
		imageTensors.push_back(imageTensor);
	}

	return imageTensors;
}

std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<torch::Tensor>>> ProcessOneOutput(torch::Tensor output0, torch::Tensor prototypes)
{
	// Perform non-maximum suppression (NMS)
	std::vector<torch::Tensor> l_class_NMS = helper::NonMaxSuppression(output0, prototypes, 0.5, 0.5);
	// Get segmentation masks
	std::tuple<std::vector<std::vector<torch::Tensor>>, std::vector<std::vector<int>>, std::vector<std::vector<float>>> segmentationOutput = helper::GetSegmentationMasks(l_class_NMS, prototypes, { 960, 960 });
	std::vector<std::vector<torch::Tensor>> masks = std::get<0>(segmentationOutput);

	return { l_class_NMS, masks };
}

void Predict_YOLO(std::vector<cv::Mat> images, torch::jit::script::Module model)
{
	std::vector<torch::Tensor> imageTensors = PreprocessImages(images, at::kCUDA, 3);

	// Prepare the tensors for model input
	torch::Tensor input = torch::cat(imageTensors, 0);
	std::vector<torch::jit::IValue> inputs{ input };
	
	auto t1 = std::chrono::high_resolution_clock::now();
	// Forward pass
	torch::jit::IValue output = model.forward(inputs);

	// Process the output
	torch::Tensor output_0_all = output.toTuple()->elements()[0].toTensor().to(at::kCPU);
	torch::Tensor prototypes_all = output.toTuple()->elements()[1].toTensor().to(at::kCPU);

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	cout << "Inference duration: " << duration << "ms" << endl;

	int imgIndex = 0;
	std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<torch::Tensor>>> results = ProcessOneOutput(output_0_all[imgIndex], prototypes_all[imgIndex]);

	auto t3 = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
	cout << "Postprocessing duration: " << duration << "ms" << endl;

	//helper::DrawResults(images[imgIndex], results);
}

void Predict_UNET(std::vector<cv::Mat> images, torch::jit::script::Module model)
{
	std::vector<torch::Tensor> imageTensors = PreprocessImages(images, at::kCUDA, 1);

	// Prepare the tensors for model input
	torch::Tensor input = torch::cat(imageTensors, 0);
	std::vector<torch::jit::IValue> inputs{ input };

	// Forward pass
	auto t1 = std::chrono::high_resolution_clock::now();

	torch::Tensor output = model.forward(inputs).toTensor().to(at::kCPU);

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	cout << "Inference duration: " << duration << "ms" << endl;

	// Process the output
}

void Predict_Deeplab(std::vector<cv::Mat> images, torch::jit::script::Module model)
{
	std::vector<torch::Tensor> imageTensors = PreprocessImages(images, at::kCUDA, 1);

	// Prepare the tensors for model input
	torch::Tensor input = torch::cat(imageTensors, 0);
	std::vector<torch::jit::IValue> inputs{ input };

	// Forward pass
	auto t1 = std::chrono::high_resolution_clock::now();

	torch::Tensor output = model.forward(inputs).toGenericDict().at("out").toTensor().to(at::kCPU);

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	cout << "Inference duration: " << duration << "ms" << endl;

	// Process the output
}

std::vector<std::vector<cv::Mat>> TorchUnet_PredictMasks(std::vector<cv::Mat> images)
{
	const char* MODEL_UNET_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\unet_exported.torchscript";
	torch::jit::script::Module net;
	net = torch::jit::load(MODEL_UNET_PATH, at::kCUDA);
	net.eval();

	std::vector<std::vector<cv::Mat>> output;

	std::vector<torch::Tensor> imageTensors;
	for (const auto& image : images) 
	{
		cv::Mat resized;
		cv::resize(image, resized, cv::Size(960, 960));
		cv::cvtColor(resized, resized, cv::COLOR_BGR2GRAY);

		torch::Tensor tensorImage = torch::from_blob(resized.data, { 1, resized.rows, resized.cols, resized.channels() }, torch::kByte).to(at::kCUDA);
		tensorImage = tensorImage.permute({ 0, 3, 1, 2 });
		tensorImage = tensorImage.toType(torch::kFloat32).div(255);
		imageTensors.push_back(tensorImage);
	}

	torch::Tensor input = torch::cat(imageTensors, 0);
	std::vector<torch::jit::IValue> inputs{ input };

	auto t1 = std::chrono::high_resolution_clock::now();

	auto pred = net.forward(inputs).toTensor().to(torch::kCPU);

	auto predAccessor = pred.accessor<float, 4>();
	int batches = predAccessor.size(0);
	int height = predAccessor.size(2);
	int width = predAccessor.size(3);
	int classes = predAccessor.size(1);
	

	for (int i = 0; i < batches; ++i) {
		std::vector<cv::Mat> prediction;
		cv::Mat probImage(cv::Size(width, height), CV_32FC(classes), predAccessor[i].data());
		std::vector<cv::Mat> splitProbImages;
		cv::split(probImage, splitProbImages);
		for (int j = 0; j < classes; ++j) {
			cv::Mat normPrediction;
			splitProbImages[j].convertTo(normPrediction, CV_8UC1, 255.0);
			prediction.push_back(normPrediction);
		}
		output.push_back(prediction);
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	cout << "Inference duration: " << duration << "ms" << endl;

	return output;
}