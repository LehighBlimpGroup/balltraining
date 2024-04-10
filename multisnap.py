# This work is licensed under the MIT license.
# Copyright (c) 2013-2023 OpenMV LLC. All rights reserved.
# https://github.com/openmv/openmv/blob/master/LICENSE
#
# Snapshot Example
#
# Note: You will need an SD card to run this example.
# You can use your OpenMV Cam to save image files.

import sensor
import time
import machine

sensor.reset()  # Reset and initialize the sensor.
sensor.set_pixformat(sensor.RGB565)  # Set pixel format to RGB565 (or GRAYSCALE)
sensor.set_framesize(sensor.QVGA)  # Set frame size to QVGA (320x240)
sensor.skip_frames(time=2000)  # Wait for settings take effect.

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test1.jpg")  # or "example.bmp" (or others)
print("image1 taken")

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test2.jpg")  # or "example.bmp" (or others)
print("image2 taken")

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test3.jpg")  # or "example.bmp" (or others)
print("image3 taken")

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test4.jpg")  # or "example.bmp" (or others)
print("image4 taken")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

led = machine.LED("LED_BLUE")

img = sensor.snapshot()
img.save("test5.jpg")  # or "example.bmp" (or others)
print("image5 taken")

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test6.jpg")  # or "example.bmp" (or others)
print("image6 taken")

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test7.jpg")  # or "example.bmp" (or others)
print("image7 taken")

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test8.jpg")  # or "example.bmp" (or others)
print("image8 taken")

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test9.jpg")  # or "example.bmp" (or others)
print("image9 taken")

led = machine.LED("LED_BLUE")

start = time.ticks_ms()
while time.ticks_diff(time.ticks_ms(), start) < 10000:
    sensor.snapshot()
    led.toggle()

led.off()

img = sensor.snapshot()
img.save("test10.jpg")  # or "example.bmp" (or others)
print("image10 taken")

img = sensor.snapshot()
img.save("test10.jpg")  # or "example.bmp" (or others)
print("image10 taken")


raise (Exception("Please reset the camera to see the new file."))
