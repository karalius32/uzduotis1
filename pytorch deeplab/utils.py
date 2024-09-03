import torch


def evaluate_model(model, dataloader, loss_fn, device):
    model.eval()
    running_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device).squeeze(1)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()

            # Calculate IoU
            preds = torch.argmax(outputs, dim=1)
            iou = (preds & masks).sum() / (preds | masks).sum()
            total_iou += iou.item()
    
    avg_loss = running_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return avg_loss, avg_iou