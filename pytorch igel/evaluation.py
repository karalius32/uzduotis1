import torch


def evaluate_model(model, dataloader, loss_fn, device, class_n):
    model.eval()
    running_loss = 0
    total_iou = 0
    class_n = class_n + 1

    precisions = [0 for _ in range(class_n)]
    recalls = [0 for _ in range(class_n)]
    ious = [0 for _ in range(class_n)]
    dices = [0 for _ in range(class_n)]
    classes_counts = [0 for _ in range(class_n)]

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device).squeeze(1)

            # Forward pass
            model.dont_slice = True
            outputs = model(images)
            model.dont_slice = False
            loss = loss_fn(outputs, masks)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            # Total iou
            total_iou += (torch.where(preds & masks > 0, 1, 0).sum() / torch.where(preds | masks > 0, 1, 0).sum()).item()

            # precision, recall and iou for each class
            for c in range(1, class_n):
                tp = torch.sum(preds[masks == c] == c).item()
                fn = torch.sum(preds[masks == c] != c).item()
                fp = torch.sum(masks[preds == c] != c).item()

                precision = tp / (tp + fp + 1e-6)
                recall = tp / (tp + fn + 1e-6)
                iou = (((preds == c) & (masks == c)).sum() / (((preds == c) | (masks == c)).sum() + 1e-6)).item()
                dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

                precisions[c] += precision
                recalls[c] += recall
                ious[c] += iou
                dices[c] += dice

                if (torch.sum(masks == c) > 0):
                    classes_counts[c] += 1

    for c in range(1, class_n):
        precisions[c] = precisions[c] / classes_counts[c]
        recalls[c] = recalls[c] / classes_counts[c]
        ious[c] = ious[c] / classes_counts[c]
        dices[c] = dices[c] / classes_counts[c]
    avg_loss = running_loss / len(dataloader)
    avg_iou = total_iou / len(dataloader)

    return precisions, recalls, ious, dices, avg_loss, avg_iou
