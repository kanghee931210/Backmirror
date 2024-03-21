# Backmirror
check your This is a rearview mirror that automatically checks what's behind you using deep learning and a webcam.

## install

+ git clone https://github.com/kanghee931210/Backmirror.git

+ cd Backmirror

+ pip install -r requirements.txt
- - - 
## Run

+ python Backmirror.py 

If you want to quit the app, press ESC

### 
## Parameter Info

### The rearview mirror currently has a total of 10 parameters.

+ net_type
  + net_type represents the model architecture. Currently RFB(defalut) and slim are available.
+ net_size
  + net_size represents train imz size . Currently 640(defalut) and 320 are available.
+ input_size
  + input_size represents inference imz size . Currently 480(defalut) and 128/160/320/480/640/1280 are available.
+ threshold
  + threshold represents the detector's threshold, As the value gets smaller, more object ​​are found. Currently 0.8(defalut) and 0~1(float) are available.
+ candidate_size
  + candidate_size is a parameter used in nms. As the value increases, more candidates are found in nms. Currently 1000(defalut) and anything int value are available.
+ test_device
  + test_device indicates whether the model will run on CPU or GPU. Currently cpu(defalut) and cuda:0 are available.
+ site
  + site refers to the website that will be displayed on the monitor when the Back Mirror detects a person behind it. The current default website is http://python.org
+ detect
  + detects represents the threshold for Backmirror. If it is set to 2, a site will be displayed if two or more people are detected on the cam. The current default is 2
+ delay
  + Delay refers to the time the app is delayed after launching the website. The current default is 10 seconds.
+ monitor
  + monitor is a parameter that determines whether to display the webcam on the screen. If False, the webcam will not be output. The current default is True.
 
- - - 

## If you have any questions, please email me anytime. kanghe931210@gmail.com


### ref 

https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/tree/master
