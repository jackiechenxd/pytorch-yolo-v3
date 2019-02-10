from __future__ import division

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from preprocess import *
from darknet import Darknet


class GenerateFinalDetections():
    def __init__(self):
        self.cfg = './cfg/ap_yolov3-tiny_obj.cfg'
        self.weightsfile = './weights/ap_yolov3-tiny_obj_10000.weights'
        print("Loading network.....")
        self.model = Darknet(self.cfg)
        self.model.load_weights(self.weightsfile)
        print("Network successfully loaded.")
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            print('GPU is available.')
        else:
            print('GPU is NOT available.')
        self.classes = ['marker']

    def get_test_input(self, input_dim, CUDA):
        img = cv2.imread("dog-cycle-car.png")
        img = cv2.resize(img, (input_dim, input_dim))
        img_ = img[:, :, ::-1].transpose((2, 0, 1))
        img_ = img_[np.newaxis, :, :, :] / 255.0
        img_ = torch.from_numpy(img_).float()
        img_ = Variable(img_)

        if CUDA:
            img_ = img_.cuda()
        return img_

    def detect_single(self, img):
        confidence = 0.5
        nms_thesh = 0.4
        input_dim = 416
        num_classes = 1

        if self.cuda:
            self.model.cuda()

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_, orig_img, dim = prep_image_content(img, input_dim)
        if self.cuda:
            img_ = img_.cuda()

        self.model.eval()

        with torch.no_grad():
            prediction = self.model(Variable(img_), self.cuda)

        prediction = write_results(prediction, confidence, num_classes, nms=True, nms_conf=nms_thesh)
        output = prediction

        objs = [self.classes[int(x[-1])] for x in output if int(x[0]) == 0]
        print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
        print("----------------------------------------------------------")

        dim_torch = torch.index_select(torch.FloatTensor(dim).repeat(1, 2).cuda(), 0, output[:, 0].long())
        scaling_factor = torch.min(input_dim / dim_torch, 1)[0].view(-1, 1)
        output[:, [1, 3]] -= (input_dim - scaling_factor * dim_torch[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (input_dim - scaling_factor * dim_torch[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, dim_torch[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, dim_torch[i, 1])

        return output[0, 1:5].int().cpu().numpy()

    def predict(self, imageRGB):
        bbox = self.detect_single(imageRGB)
        x, y, x2, y2 = bbox
        image = cv2.cvtColor(imageRGB, cv2.COLOR_RGB2BGR)
        crop_img = image[y:y2, x:x2]
        crop_img = cv2.GaussianBlur(crop_img, (15, 15), 0)
        equ = cv2.equalizeHist(crop_img[:, :, 1])
        # cv2.imshow("NormImage", equ)
        # cv2.waitKey(0)

        # _, mask_s = cv2.threshold(equ, 0, 255, type=cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # cv2.imshow("BinaryImage", mask_s)
        # cv2.waitKey(0)

        hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        # working well wit blue color but not purple
        lower_blue = np.array([100, 0, 0])
        upper_blue = np.array([180, 255, 255])

        # lower_blue = np.array([100, 100, 20])
        # upper_blue = np.array([180, 255, 255])
        binary = 255 - cv2.inRange(hsv, lower_blue, upper_blue)
        binary = cv2.GaussianBlur(binary, (5, 5), 0)

        im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        index = np.argmax([cv2.contourArea(cnt) for cnt in contours])
        rect_cnt = contours[index]
        epsilon = 0.1 * cv2.arcLength(rect_cnt, True)
        approx = cv2.approxPolyDP(rect_cnt, epsilon, True)
        shift_loc = approx + [x, y]

        cv2.drawContours(image, [shift_loc], -1, (0, 0, 255), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        return shift_loc.squeeze()


if __name__ == '__main__':
    detector = GenerateFinalDetections()
    image = cv2.imread('./imgs/IMG_7403.JPG')
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.predict(imageRGB)
    print(result)
