"""
文件功能：OpenMV 端实时图像分类主程序，用于加载 TFLite 模型并在摄像头画面上显示识别结果。
代码分布：前半部分初始化摄像头、LCD 和模型，主循环中采集图像、执行分类、绘制 FPS 与检测框。
整理思路：该文件面向 OpenMV 固件运行，依赖 sensor、image、tf、pyb 等 OpenMV 模块，不在普通 Python 环境执行。
使用方法：将本文件、openmv_my_LCD.py、.tflite 模型和 labels.txt 拷贝到 OpenMV 设备后运行。
"""

"""
 By: 28014 - 周五 3月 3 2023
@author: Zn
OpenMV脱机运行TFlite模型
"""
# Edge Impulse - OpenMV Image Classification Example

import sensor, image, time, os, tf, uos, gc
import openmv_my_LCD
def find_max(blobs):
    max_size=0
    for blob in blobs:
         if blob.pixels() > max_size:
             maxblob = blob
             max_size = blob.pixels()
    return maxblob
sensor.reset()                         # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)    # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
#sensor.set_windowing((240, 240))       # Set 240x240 window.屏幕显示要求320*240
sensor.skip_frames(time=2000)          # Let the camera adjust.
openmv_my_LCD.LCDinit() # Initialize the lcd screen.

net = None
labels = None

try:
    # load the model, alloc the model file on the heap if we have at least 64K free after loading
    net = tf.load("smd_150rgb_softmax_onehot.tflite", load_to_fb=uos.stat('smd_150rgb_softmax_onehot.tflite')[6] > (gc.mem_free() - (64*1024)))
except Exception as e:
    print(e)
    raise Exception('Failed to load "trained.tflite", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

try:
    labels = [line.rstrip('\n') for line in open("labels.txt")]
except Exception as e:
    raise Exception('Failed to load "labels.txt", did you copy the .tflite and labels.txt file onto the mass-storage device? (' + str(e) + ')')

clock = time.clock()
while(True):
    clock.tick()

    img = sensor.snapshot().lens_corr(1.8)
    blobs = img.find_blobs([(5, 100, 11, 57, -78, -24)], pixels_threshold=200, area_threshold=200, margin=3)
    # default settings just do one detection... change them to search the image...
    for obj in net.classify(img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
        print("**********\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
        # This combines the labels and confidence values into a list of tuples
        predictions_list = list(zip(labels, obj.output()))

        for i in range(len(predictions_list)):
            print("%s = %f" % (predictions_list[i][0], predictions_list[i][1]))
        if predictions_list[0][1]>predictions_list[1][1]:
             img.draw_string(5,40,predictions_list[0][0],scale=3, color=(255,0,0))
        else:img.draw_string(5,40,predictions_list[1][0],scale=3, color=(255,0,0))
    print(clock.fps(), "fps")
    fps="%.2ffps"%clock.fps()

    img.draw_string(5,5, fps,scale=3, color=(255,0,0))
    print(blobs)
    if blobs != []:
        blob = find_max(blobs)
        # These values depend on the blob not being circular - otherwise they will be shaky.
        img.draw_rectangle(blob.rect(),color=(255,0,0))
    #img.draw_string(20, , fps, color=White)
    openmv_my_LCD.display(img) # Take a picture and display the image.
