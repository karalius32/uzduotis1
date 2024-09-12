import numpy as np
from matplotlib import pyplot as plt
import torch
import cv2

output_0 = torch.load("output_0.pt").state_dict()["0"][0]
prototypes = torch.load("prototypes.pt").state_dict()["0"][0]

nb_class = output_0.shape[0]-4-prototypes.shape[0] # extract total number of classes
l_class = [[] for k in range(nb_class)]
output_0_T = output_0.T # so it become shape (8400 116)
threshold_detection = 0.5 #threshold to filter irrelevent detection
theshold_iou = 0.5 #threshold for NMS algo
for detection in output_0_T:
    conf = detection[4:nb_class+4] #extract all class confidence values for one detecton
    max_conv = torch.max(conf) #maximum confidence value
    argmax_conv = torch.argmax(conf)#class of the maximum confidence value
    if max_conv > threshold_detection:
        l_class[argmax_conv].append(np.concatenate((detection[:4], np.array([max_conv]), detection[4+nb_class:])))

image = np.zeros((960, 960, 3), dtype=np.uint8)
# Draw each rectangle
for rect in l_class[0]:
    x_center, y_center, width, height = rect[:4]
    
    # Calculate top-left and bottom-right points of the rectangle
    top_left = (int(x_center - width / 2), int(y_center - height / 2))
    bottom_right = (int(x_center + width / 2), int(y_center + height / 2))
    
    print(f"points: {top_left} | {bottom_right}")

    # Draw the rectangle on the image (color is white)
    image = cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)

plt.imshow(image)
plt.show()