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
#include <fstream>

using std::cout;
using std::endl;


// Function for calculating median 
double Median(std::vector<int> v, int n)
{
	// Sort the vector 
	sort(v.begin(), v.end());

	// Check if the number of elements is odd 
	if (n % 2 != 0)
		return (double)v[n / 2];

	// If the number of elements is even, return the average 
	// of the two middle elements 
	return (double)(v[(n - 1) / 2] + v[n / 2]) / 2.0;
}


double TestModel(torch::jit::script::Module model, std::vector<cv::Mat> images, std::string name, int modelType);
std::vector<cv::Mat> ReadImages(std::string folderPath, cv::Size modelInputSize);
std::vector<torch::Tensor> PreprocessImages(std::vector<cv::Mat> images, torch::DeviceType deviceType, int ch);

int Predict(std::vector<cv::Mat> images, torch::jit::script::Module model);
torch::Tensor PredictMasks(torch::jit::script::Module* net, std::vector<cv::Mat> images);
std::vector<cv::Mat> GetDrawableMasksFromTensor(torch::Tensor predictions);

int main() 
{
	const char* MODEL_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\models\\deeplabv3plus_mobilevitv2_050_300.torchscript";

	const char* IMAGES_PATH = "C:\\git\\darbas\\libtorch label segmentation\\libtorch label segmentation\\images_igel";

	torch::jit::script::Module model = torch::jit::load(MODEL_PATH, at::kCUDA); // load model
	model.eval();

	std::vector<cv::Mat> images = ReadImages(IMAGES_PATH, cv::Size(320, 320)); // load images

	/// /// /// Run model several times (because first 2-3 times are slower than the rest)
	std::vector<torch::Tensor> imageTensors = PreprocessImages(images, at::kCUDA, 1); // images to tensor
	torch::Tensor input = torch::cat(imageTensors, 0);
	std::vector<torch::jit::IValue> inputs{ input };
	model.forward(inputs);
	model.forward(inputs);
	model.forward(inputs);
	/// /// ///

	torch::Tensor outputs = PredictMasks(&model, images); // predict masks
	cout << outputs.sizes() << endl;

	//torch::save(outputs, "predictedMasks2.pt");

	std::vector<cv::Mat> masks = GetDrawableMasksFromTensor(outputs); // get masks as cv::Mat

	
	cv::Mat imageWithMask;
	cv::cvtColor(images[1], images[1], cv::COLOR_BGR2BGRA);
	cv::addWeighted(images[1], 0.9, masks[1], 0.25, 0, imageWithMask);

	cv::imshow("img1", imageWithMask);
	cv::waitKey(0);

	return 0;
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
		imageTensor = imageTensor.toType(torch::kFloat32);
		imageTensor = imageTensor.toType(torch::kFloat32).div(255);
		imageTensor = imageTensor.permute({ 2, 0, 1 });
		imageTensor = imageTensor.unsqueeze(0);
		imageTensor = imageTensor.contiguous();
		imageTensors.push_back(imageTensor);
	}

	return imageTensors;
}


int Predict(std::vector<cv::Mat> images, torch::jit::script::Module model)
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
	//cout << "Inference duration: " << duration << "ms" << endl;

	return duration;
	// Process the output
}

torch::Tensor PredictMasks(torch::jit::script::Module* net, std::vector<cv::Mat> images)
{
	float treshold = 0.2;

	std::vector<std::vector<cv::Mat>> output;

	std::vector<torch::Tensor> imageTensors = PreprocessImages(images, at::kCUDA, 1); // images to tensor

	torch::Tensor input = torch::cat(imageTensors, 0);
	std::vector<torch::jit::IValue> inputs{ input };

	auto t1 = std::chrono::high_resolution_clock::now();

	auto pred = net->forward(inputs).toTensor().to(torch::kCPU);

	auto maxValues = std::get<0>(torch::max(pred, 1));
	auto condition = maxValues > treshold;
	auto indices = torch::argmax(pred, 1) + 1;
	auto predictions = torch::where(condition, indices, torch::zeros_like(indices));

	/*auto predAccessor = pred.accessor<float, 4>();
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
	}*/

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
	cout << "Inference duration: " << duration << "ms" << endl;

	return predictions;
}

std::vector<cv::Mat> GetDrawableMasksFromTensor(torch::Tensor predictions)
{
	int classes_n = 4;
	std::vector<cv::Mat> coloredMats;

	auto predAccessor = predictions.accessor<int64, 3>();
	int batches = predAccessor.size(0);
	int height = predAccessor.size(1);
	int width = predAccessor.size(2);

	int a = 255;
	std::vector<std::vector<int>> colors = {
		{0, 0, 0, 0},
		{0, 0, 255, a},
		{0, 255, 0, a},
		{255, 0, 0, a}
	};

	for (int i = 0; i < batches; ++i)
	{
		torch::Tensor coloredTensor = torch::zeros({ 4, height, width }).to(torch::kUInt8);
		auto slice = predictions[i];

		for (int c = 0; c < classes_n; ++c)
		{
			auto mask = (slice == c).expand({ 4, height, width }).to(torch::kUInt8);
			auto color = torch::tensor(colors[c], torch::kUInt8).view({ 4, 1, 1 });
			coloredTensor += mask * color;
		}

		coloredTensor = coloredTensor.permute({ 1, 2, 0 }).contiguous();
		cv::Mat coloredMat(height, width, CV_8UC4, coloredTensor.data_ptr());
		coloredMats.push_back(coloredMat.clone());
	}

	return coloredMats;
}