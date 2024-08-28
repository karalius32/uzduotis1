#include "helper.h"

// Function to convert std::vector<std::vector<float>> to torch::Tensor
torch::Tensor ConvertToTensor(const std::vector<std::vector<float>>& vector2d) {
    // Get the number of boxes and the number of elements per box
    auto num_boxes = vector2d.size();
    auto num_elements_per_box = vector2d[0].size();

    // Flatten the 2D vector into a 1D vector
    std::vector<float> flattened;
    flattened.reserve(num_boxes * num_elements_per_box);
    for (const auto& box : vector2d) {
        flattened.insert(flattened.end(), box.begin(), box.end());
    }

    // Create a torch::Tensor from the flattened data
    torch::Tensor boxes = torch::tensor(flattened).view({ static_cast<int64_t>(num_boxes), static_cast<int64_t>(num_elements_per_box) });

    return boxes;
}

torch::Tensor helper::nms(const std::vector<std::vector<float>>& bounding_boxes, float threshold) 
{
    // If no bounding boxes, return empty tensor
    if (bounding_boxes.empty()) {
        return torch::empty({ 0 });
    }

    // Convert bounding boxes to a torch::Tensor
    torch::Tensor boxes = ConvertToTensor(bounding_boxes);

    // Extracting coordinates
    torch::Tensor x_mid = boxes.select(1, 0);
    torch::Tensor y_mid = boxes.select(1, 1);
    torch::Tensor width = boxes.select(1, 2);
    torch::Tensor height = boxes.select(1, 3);
    torch::Tensor score = boxes.select(1, 4);

    // Calculating start and end coordinates
    torch::Tensor start_x = x_mid - width / 2;
    torch::Tensor start_y = y_mid - height / 2;
    torch::Tensor end_x = x_mid + width / 2;
    torch::Tensor end_y = y_mid + height / 2;

    // Picked bounding boxes
    std::vector<torch::Tensor> picked_boxes;

    // Compute areas of bounding boxes
    torch::Tensor areas = (end_x - start_x + 1) * (end_y - start_y + 1);

    // Sort by confidence score of bounding boxes
    torch::Tensor order = std::get<1>(score.sort(0, /*descending=*/false));

    // Iterate bounding boxes
    while (order.size(0) > 0) {
        // The index of the largest confidence score
        int64_t index = order[-1].item<int64_t>();

        // Pick the bounding box with the largest confidence score
        picked_boxes.push_back(boxes.index({ index }));

        //if (order.size(0) == 1) break;

        // Compute ordinates of intersection-over-union (IoU)
        torch::Tensor x1 = torch::max(start_x.index({ index }), start_x.index({ order.slice(0, 0, -1) }));
        torch::Tensor x2 = torch::min(end_x.index({ index }), end_x.index({ order.slice(0, 0, -1) }));
        torch::Tensor y1 = torch::max(start_y.index({ index }), start_y.index({ order.slice(0, 0, -1) }));
        torch::Tensor y2 = torch::min(end_y.index({ index }), end_y.index({ order.slice(0, 0, -1) }));

        // Compute areas of intersection-over-union
        torch::Tensor w = torch::max(torch::tensor(0.0), x2 - x1 + 1);
        torch::Tensor h = torch::max(torch::tensor(0.0), y2 - y1 + 1);
        torch::Tensor intersection = w * h;

        // Compute the ratio between intersection and union
        torch::Tensor ratio = intersection / (areas.index({ index }) + areas.index({ order.slice(0, 0, -1) }) - intersection);

        // Mask to select the boxes with IoU less than the threshold
        torch::Tensor left = ratio < threshold;

        // Update order to only include boxes that are not suppressed

        try
        {
            order = order.masked_select(torch::cat({ left, torch::tensor({false}) }, 0));
        }
        catch (at::Error e)
        {
            int a = 0;
        }
    }

    // Stack the picked boxes into a single tensor
    return torch::stack(picked_boxes);
}

std::vector<torch::Tensor> helper::NonMaxSuppression(torch::Tensor output0, torch::Tensor prototypes, float threshold_detection, float threshold_iou)
{
    int nb_class = output0.size(0) - 4 - prototypes.size(0);
    std::vector<std::vector<std::vector<float>>> l_class(nb_class);
    torch::Tensor output_0_T = output0.transpose(0, 1).contiguous();

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
            auto detection_tail = detection.slice(0, 4 + nb_class, None).to(torch::kCPU).data_ptr<float>();

            combined.insert(combined.end(), detection_head, detection_head + 4);
            combined.push_back(max_conv_val);
            combined.insert(combined.end(), detection_tail, detection_tail + detection.size(0) - (4 + nb_class));

            l_class[argmax_conv].emplace_back(combined);
        }
    }

    //helper::DrawRectangles(l_class);

    std::vector<torch::Tensor> l_class_NMS;
    for (const auto& clas : l_class) {
        l_class_NMS.push_back(helper::nms(clas, threshold_iou));
    }

    return l_class_NMS;
}

torch::Tensor helper::CropImage(const torch::Tensor& image, const torch::Tensor& box) 
{
    int x = box[0].item<int>();
    int y = box[1].item<int>();
    int w = box[2].item<int>();
    int h = box[3].item<int>();

    // Ensure box coordinates are within image boundaries
    int x1 = std::max(0, x - w / 2);
    int y1 = std::max(0, y - h / 2);
    int x2 = std::min(int(image.size(0)), x + w / 2); 
    int y2 = std::min(int(image.size(1)), y + h / 2); 

    // Create a mask to zero out areas outside the box
    torch::Tensor mask = torch::zeros_like(image, torch::kFloat);
    mask.index_put_({ torch::indexing::Slice(y1, y2), torch::indexing::Slice(x1, x2) }, 1.0);

    // Apply the mask to the original image
    torch::Tensor croppedImage = image.to(torch::kFloat32) * mask;

    return croppedImage;
    
}

torch::Tensor helper::ThresholdImage(const torch::Tensor& image, float threshold) 
{
    torch::Tensor thresholdedImage = torch::where(image > threshold, torch::tensor(255.0f), torch::tensor(0.0f));
    return thresholdedImage.to(torch::kU8); // Convert to uint8 for further processing
}

std::tuple<std::vector<std::vector<torch::Tensor>>, std::vector<std::vector<int>>, std::vector<std::vector<float>>> helper::GetSegmentationMasks(std::vector<torch::Tensor> l_class_NMS, torch::Tensor prototypes, std::vector<int64_t> imageSize)
{
    std::vector<std::vector<torch::Tensor>> l_mask;
    std::vector<std::vector<int>> l_class;
    std::vector<std::vector<float>> l_conf;
    
    for (int k = 0; k < l_class_NMS.size(); k++)
    {
        std::vector<torch::Tensor> l_mask_;
        std::vector<int> l_class_;
        std::vector<float> l_conf_;

        for (int j = 0; j < l_class_NMS[k].size(0); j++)
        {
            torch::Tensor detection = l_class_NMS[k][j];
            torch::Tensor coeff = detection.slice(0, 5, None);
            torch::Tensor mask = prototypes * coeff.reshape({ prototypes.size(0), 1, 1 });

            torch::Tensor resized_mask = torch::empty({ mask.size(0), imageSize[0], imageSize[1] });

            for (int i = 0; i < mask.size(0); i++)
            {
                resized_mask[i] = torch::nn::functional::interpolate(
                    mask[i].unsqueeze(0).unsqueeze(0),
                    torch::nn::functional::InterpolateFuncOptions().size(imageSize).mode(torch::kBilinear).align_corners(false)
                ).squeeze(0).squeeze(0);
            }

            torch::Tensor cropped = CropImage(torch::mean(resized_mask, 0), detection.slice(0, None, 4));
            l_mask_.push_back(ThresholdImage(cropped));
			l_class_.push_back(k);
			l_conf_.push_back(detection[4].item<float>());
        }

        l_mask.push_back(l_mask_);
        l_class.push_back(l_class_);
        l_conf.push_back(l_conf_);
    }

	return std::make_tuple(l_mask, l_class, l_conf);
}

void helper::DrawRectangles(std::vector<torch::Tensor> l_class_NMS)
{
    cv::Mat image = cv::Mat::zeros(960, 960, CV_8UC3);

    // Draw each rectangle
    for (int i = 0; i < l_class_NMS[0].sizes()[0]; i++)
    {
        int x_center = l_class_NMS[0][i][0].item<int>();
        int y_center = l_class_NMS[0][i][1].item<int>();
        int width = l_class_NMS[0][i][2].item<int>();
        int height = l_class_NMS[0][i][3].item<int>();

        // Calculate top-left and bottom-right points of the rectangle
        cv::Point top_left(x_center - width / 2, y_center - height / 2);
        cv::Point bottom_right(x_center + width / 2, y_center + height / 2);

        // Draw the rectangle on the image (color is white, thickness is 2)
        cv::rectangle(image, top_left, bottom_right, cv::Scalar(255, 255, 255), 2);
    }

    // Display the image
    cv::imshow("Image", image);
    cv::waitKey(0);
}

void helper::DrawRectangles(std::vector<std::vector<std::vector<float>>> l_class)
{
    cv::Mat image = cv::Mat::zeros(960, 960, CV_8UC3);

    // Draw each rectangle
    for (int i = 0; i < l_class[0].size(); i++)
    {
        int x_center = l_class[0][i][0];
        int y_center = l_class[0][i][1];
        int width = l_class[0][i][2];
        int height = l_class[0][i][3];

        // Calculate top-left and bottom-right points of the rectangle
        cv::Point top_left(x_center - width / 2, y_center - height / 2);
        cv::Point bottom_right(x_center + width / 2, y_center + height / 2);

        // Draw the rectangle on the image (color is white, thickness is 2)
        cv::rectangle(image, top_left, bottom_right, cv::Scalar(255, 255, 255), 2);
    }

    // Display the image
    cv::imshow("Image", image);
    cv::waitKey(0);
}

// draws only first class results
void helper::DrawResults(cv::Mat image, std::tuple<std::vector<torch::Tensor>, std::vector<std::vector<torch::Tensor>>> results)
{
    cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    std::vector<torch::Tensor> boxes = std::get<0>(results);
    std::vector<std::vector<torch::Tensor>> masks = std::get<1>(results);

    // Draw each rectangle
    for (int i = 0; i < boxes[0].size(0); i++)
    {
        int x_center = boxes[0][i][0].item<int>();
        int y_center = boxes[0][i][1].item<int>();
        int width = boxes[0][i][2].item<int>();
        int height = boxes[0][i][3].item<int>();

        // Calculate top-left and bottom-right points of the rectangle
        cv::Point top_left(x_center - width / 2, y_center - height / 2);
        cv::Point bottom_right(x_center + width / 2, y_center + height / 2);

        // Draw mask
        cv::Mat mask_mat = cv::Mat(masks[0][i].size(0), masks[0][i].size(1), CV_8UC1, masks[0][i].data_ptr());
        cv::Scalar color(255, 0, 255);
        cv::Mat color_mask;
        cv::Mat overlay;
        cv::cvtColor(mask_mat, color_mask, cv::COLOR_GRAY2BGR);
        color_mask = color_mask.mul(cv::Scalar(color));
        cv::addWeighted(image, 0.5, color_mask, 0.5, 0, overlay);
        overlay.copyTo(image, mask_mat);

        // Draw the rectangle on the image (color is white, thickness is 2)
        cv::rectangle(image, top_left, bottom_right, cv::Scalar(255, 0, 255), 2);
    }

    // Display the image
    cv::imshow("Image", image);
    cv::waitKey(0);

}