from vision.ssd.mb_tiny_fd import  create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

from vision.ssd.config.fd_config import define_img_size
from vision.utils.misc import Timer

import torch
import cv2

import webbrowser
import time

import argparse

parser = argparse.ArgumentParser(description='Back Mirror')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--net_size', default="640", type=str,
                    help='The network architecture pretrain size ,optional: 640 or 320')
parser.add_argument('--input_size', default=480, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.8, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,
                    help='nms candidate size')
parser.add_argument('--test_device', default="cpu", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--site', default="http://python.org", type=str,
                    help='open web site url')
parser.add_argument('--detect', default=2, type=int,
                    help='This is the number of people Backmirror detects. If set to 2, a website will be displayed when two or more people are detected on the camera')
parser.add_argument('--delay', default=10, type=int,
                    help='Back Mirror delay time, If you set this value small, the frequency with which the website appears may increase.')
parser.add_argument('--monitor', default=True, type=int,
                    help="if you don't wnat to Monitoring ,Set False")

args = parser.parse_args()



capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# model_path = "weight/version-slim-640.pth"

define_img_size(args.input_size)
net_type = args.net_type
net_size = args.net_size
test_device = args.test_device
candidate_size = args.candidate_size
web_site = args.site
delay_time = args.delay
detect_num = args.detect
detect_num = max(detect_num,1)
monitor_p = args.monitor

if net_type == 'slim':
    if net_size == '320':
        model_path = "weight/version-slim-320.pth"
    else:
        model_path = "weight/version-slim-640.pth"
    net = create_mb_tiny_fd(2, is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    if net_size == '320':
        model_path = "weight/version-RFB-320.pth"
    else:
        model_path = "weight/version-RFB-640.pth"
    net = create_Mb_Tiny_RFB_fd(2, is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
if test_device == 'cpu':
    net.load_state_dict(torch.load(model_path,map_location= torch.device('cpu')))
else:
    net.load(model_path)

timer = Timer()
while True:
    ret, frame = capture.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 1000 / 2, 0.8)
    interval = timer.end()
    # print('Time: {:.6f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))
    if labels.size(0) >= detect_num:
        webbrowser.open(web_site, new=1)
        time.sleep(delay_time)

    for i in range(boxes.size(0)):
        box = list(map(int,boxes[i, :]))#boxes[i, :].type(int)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 4)
    orig_image = cv2.resize(image, None, None, fx=0.8, fy=0.8)
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    if monitor_p:
        cv2.imshow('Back mirror', orig_image)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
capture.release()
cv2.destroyAllWindows()