import argparse
import os
from PIL import Image
from custom_transforms import *
from cvtorchvision import cvtransforms	#clone to the directory from https://github.com/hongchu098/opencv_torchvision_transforms
import numpy as np
import torch.utils.data
import torch
import time
from data import DatasetFullImages
import cv2
from time import sleep



# Main to generate images
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="checkpoint location", required=True)
    parser.add_argument("--data_root", help="data root", required=True)
    parser.add_argument("--dir_input", help="dir input", required=True)
    parser.add_argument("--dir_x1", help="dir extra 1", required=False)
    parser.add_argument("--dir_x2", help="dir extra 2", required=False)
    parser.add_argument("--dir_x3", help="dir extra 3", required=False)
    parser.add_argument("--outdir", help="output directory", required=True)
    parser.add_argument("--device", help="device", required=True)
    parser.add_argument("--resolution", type=int, nargs=2, metavar=('width', 'height'), default=(480, 640))
    args = parser.parse_args()

    generator = (torch.load(args.checkpoint, map_location=lambda storage, loc: storage))
    generator.eval()

    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    device = args.device
    print("device: " + device, flush=True)

    generator = generator.to(device)
    if device.lower() != "cpu":
        generator = generator.type(torch.half)
    transform = build_transform()
    cap = cv2.VideoCapture(0)
    width, height = args.resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    while(True):
        ret, test = cap.read()
        res = int(min(test.shape[0], test.shape[1])/2)
        x = int(test.shape[0] / 2)
        y = int(test.shape[1] / 2)
        #print(x, y, res)
        test = test[x-res:x+res,y-res:y+res, :]
        test2 = cv2.resize(test, (256, 256)) #test 480x640 to square - change if your webcam outputs a different resolution
        net_in2 = cvtransforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(cvtransforms.ToTensor()(cv2.cvtColor(test2, cv2.COLOR_BGR2RGB))).to(args.device).unsqueeze(0)
        if device.lower() != "cpu":
            net_in2 = net_in2.type(torch.half)
        net_out = generator(net_in2)
        im = ((net_out.clamp(-1, 1) + 1) * 127.5).permute((0, 2, 3, 1)).cpu().data.numpy().astype(np.uint8)
        im2 = cv2.resize(im[0], (2*res, 2*res))
        cv2.imshow("image",np.concatenate((test,cv2.cvtColor(im2,cv2.COLOR_RGB2BGR)), axis=1))
        #cv2.imshow("image", cv2.cvtColor(im[0],cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()
            
            
