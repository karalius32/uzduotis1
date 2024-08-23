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

    std::cout << boxes.sizes() << std::endl;

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

        if (order.size(0) == 1) break;

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

        std::cout << std::endl << order << std::endl << left << std::endl;
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