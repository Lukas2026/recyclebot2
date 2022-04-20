import os.path
import RPi.GPIO as GPIO
import time
"""
Testing the creation/modification date of a text file
"""
categories = ["paper","plastic","metal","cardboard","glass"]

GPIO.setmode(GPIO.BCM)

GPIO.setup(4,GPIO.OUT)
GPIO.setup(0,GPIO.OUT)
GPIO.setup(1,GPIO.OUT)
GPIO.setup(2,GPIO.OUT)
GPIO.setup(3,GPIO.OUT)

def turn_off():
        for i in range(5):
                GPIO.output(i,False)
                

file_name = '/home/pi/shared/result.txt'
print("last modified {}".format(time.ctime(os.path.getmtime(file_name))))
print("last created {}".format(time.ctime(os.path.getctime(file_name))))

time_stamp = time.ctime(os.path.getctime(file_name))

while True:
    time_stamp_2 = time.ctime(os.path.getctime(file_name))
    if time_stamp_2 == time_stamp:
        continue
    else:
        print("new_file", time.strftime("%H:%M:%S"))
        time_stamp = time.ctime(os.path.getctime(file_name))
        with open(file_name) as my_file:
                lines = my_file.readlines()
        try:
                result = lines[0].strip()
                print(result)
                turn_off()
                GPIO.output(categories.index(result), True)
        except IndexError:
                pass
                

