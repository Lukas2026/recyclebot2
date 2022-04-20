from picamera import PiCamera
from time import sleep

camera = PiCamera()

camera.start_preview()
sleep(5)
while True:
	camera.capture('/home/pi/shared/sample.jpg')
	print('image captured')
	sleep(2)
camera.stop_preview()
