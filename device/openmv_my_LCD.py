"""
文件功能：OpenMV 外接 LCD 屏幕驱动，用于初始化屏幕并显示摄像头图像。
代码分布：前半部分定义引脚、屏幕方向和颜色常量，中间封装 SPI 写命令/数据，后半部分提供 LCDinit 和 display。
整理思路：保持为设备端独立驱动文件，供 openmv_image_classification.py 直接 import。
使用方法：与 OpenMV 主程序放在同一目录，主程序调用 LCDinit() 和 display(img)。
"""

"""
Created on Thu Dec  1 16:48:44 2022
@author: Zn
OpenMV lcd模块 驱动自己的LCD屏
openmv main文件 import
分辨率 240*320 QVGA 2.2寸
"""

import time
from pyb import Pin, SPI
fbuf = bytearray(320)#建立帧缓冲区，对于每个RGB565像素，帧缓冲区都需2字节

# IO接线：
# SDO/MISO  ---> P1(MISO)   		如果不读LCD颜色，可不接
# LED       ---> P6 (背光控制) 		可直接 接VCC，开机亮屏
# SCK       ---> P2(SCLK)			SCLK 时钟线
# SDI/MOSI  ---> P0(MOSI)			SDA 数据线
# DO/RS     ---> P8					数据命令选择线
# RESET     ---> P7					复位线
# CS        ---> P3					片选线
# GND       ---> GND
# VCC       ---> 3.3V

cs  = Pin("P3", Pin.OUT_OD)
rst = Pin("P7", Pin.OUT_PP)
rs  = Pin("P8", Pin.OUT_PP)
bl  = Pin("P6", Pin.OUT_PP)     # 背光控制

# 定义横屏/竖屏
USE_HORIZONTAL  = True         # 定义横屏
IMAGE_INVER     = True         # 旋转180°

# TFT resolution 240*320
X_MIN_PIXEL = 0
Y_MIN_PIXEL = 0
if USE_HORIZONTAL:
    X_MAX_PIXEL = 320               # 定义屏幕宽度
    Y_MAX_PIXEL = 240               # 定义屏幕高度
else:
    X_MAX_PIXEL = 240               # 定义屏幕宽度
    Y_MAX_PIXEL = 320               # 定义屏幕高度

# 常用颜色表 						#可去除
RED     = 0XF800
GREEN   = 0X07E0
BLUE    = 0X001F
BLACK   = 0X0000
YELLOW  = 0XFFE0
WHITE   = 0XFFFF

CYAN    = 0X07FF
BRIGHT_RED = 0XF810
GRAY1   = 0X8410
GRAY2   = 0X4208

# OpenMV  SPI2 总线  8位数据模式
spi = SPI(2, SPI.MASTER, baudrate=int(10000000000/66*0.06), polarity=0, phase=0,bits=8)

# SPI 写命令
def write_command_byte(c):
    cs.low()
    rs.low()
    spi.send(c)
    cs.high()

# SPI 写数据
def write_data_byte(c):
    cs.low()
    rs.high()
    spi.send(c)
    cs.high()

def write_command(c, *data): #命令数据一起写，先写命令 第二个开始为数据位。如果只写一个，则代表不写数据只写命令。
    write_command_byte(c)
    if data:
        for d in data:
            if d > 255:
                write_data_byte(d >> 8)
                write_data_byte(d&0xFF)
            else:
                write_data_byte(d)

def write_image(img):
    cs.low()
    rs.high()
    #spi.send(img)
#修改代码的核心##########################################################################
    if(True): #全发
        for m in img: 					#把一帧图像的对象取出来，放到帧缓存区中
            fbuf=m
            for i in range(0,320):		#每行每行的发送
                spi.send(fbuf[i]>>8)	#先发第N行的第I个数据的高八位
                spi.send(fbuf[i]&0xFF)	#再发低八位
            #print(fbuf)
    else:#只发一行固定的数据
        for i in range(0,320):
            spi.send(fbuf[i])
            spi.send(fbuf[i+1]&0xFF)
##########################################################################################
    cs.high()

def SetXY(xpos, ypos):
    write_command(0x2A, xpos>>8, xpos&0XFF)
    write_command(0x2B, ypos>>8, ypos&0XFF)
    write_command(0x2C)

def SetRegion(xStar, yStar, xEnd, yEnd):
    write_command(0x2A, xStar>>8, xStar&0XFF, xEnd>>8, xEnd&0XFF)
    write_command(0x2B, yStar>>8, yStar&0XFF, yEnd>>8, yEnd&0XFF)
    write_command(0x2C)

# 在指定位置绘制一个点
def DrawPoint(x, y, Color):
    SetXY(x, y)
    write_data_byte(Color >> 8)
    write_data_byte(Color&0XFF)

def ReadPoint(x, y):
    data = 0
    SetXY(x, y)
    write_data_byte(data)
    return data

def Clear(Color):
    global X_MAX_PIXEL, Y_MAX_PIXEL
    SetRegion(0, 0, X_MAX_PIXEL-1 , Y_MAX_PIXEL-1 )
    #cs.low()
    #rs.high()
    for i in range(0, Y_MAX_PIXEL):
        for m in range(0, X_MAX_PIXEL):
            write_data_byte(Color >> 8)
            write_data_byte(Color&0xFF)

def LCDinit():
    rst.low()
    time.sleep_ms(100)
    rst.high()
    time.sleep_ms(100)

    write_command(0xCB, 0x39, 0x2c, 0x00, 0x34, 0x02)
    write_command(0xCF, 0x00, 0XAA, 0XE0)
    write_command(0xE8, 0x85, 0x11, 0x78)
    write_command(0xEA, 0x00, 0x00)
    write_command(0xED, 0x67, 0x03, 0X12, 0X81)
    write_command(0xF7, 0x20)
    write_command(0xC0, 0x21)       # Power control, VRH[5:0]
    write_command(0xC1, 0x11)       # Power control, SAP[2:0];BT[3:0]
    write_command(0xC5, 0x24, 0x3C) # VCM control, 对比度调节
    write_command(0xC7, 0xB7)       # VCM control2, --

    # Memory Data Access Control
    if USE_HORIZONTAL:      # //C8   //48 68竖屏//28 E8 横屏
        if IMAGE_INVER:
            write_command(0x36, 0xE8)   # 从右到左 e8/68
        else:
            write_command(0x36, 0x28)   # 从左到右 28
    else:
        if IMAGE_INVER:
            write_command(0x36, 0xC8)   # 从下到上刷
        else:
            write_command(0x36, 0x48)   # 从上到下刷

    global X_MAX_PIXEL, Y_MAX_PIXEL
    SetRegion(0, 0, X_MAX_PIXEL - 1, Y_MAX_PIXEL - 1)

    # Interface Pixel Format
    write_command(0x3A, 0x55)

    write_command(0xB1, 0x00, 0x18)
    write_command(0xB6, 0x08, 0x82, 0x27)   # Display Function Control
    write_command(0xF2, 0x00)               # 3Gamma Function Disable
    write_command(0x26, 0x01)               # Gamma curve selected
    write_command(0xB7, 0x06)

    # set gamma
    write_command(0xE0, 0x0F, 0x31, 0x2B, 0x0C, 0x0E, 0x08, 0x4E, 0xF1, 0x37, 0x07, 0x10, 0x03, 0x0E, 0x09, 0x00)
    write_command(0XE1, 0x00, 0x0E, 0x14, 0x03, 0x11, 0x07, 0x31, 0xC1, 0x48, 0x08, 0x0F, 0x0C, 0x31, 0x36, 0x0F)

    write_command(0x11) # sleep_ms Exit
    time.sleep_ms(120)

    # Display On
    write_command(0x29)
    write_command(0x2C)
    bl.high()   # 拉背光

def display(img):
    write_command(0x2C)
    write_image(img)

#import sensor, image
#sensor.reset() # Initialize the camera sensor.
#sensor.set_pixformat(sensor.RGB565) # or sensor.GRAYSCALE
#sensor.set_framesize(sensor.QVGA) # Special 128x160 framesize for LCD Shield.
#LCDinit() # Initialize the lcd screen.
#Clear(BLUE)
#DrawPoint(50, 50, RED)
#display(sensor.snapshot())
#while(True):
    #pass
    #display(sensor.snapshot()) # Take a picture and display the image.
    #print(sensor.snapshot())




