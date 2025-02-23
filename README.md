# balltraining

## prerequisite:
1. Install the [Openmv IDE](https://openmv.io/pages/download)
2. Clone the repository: [Blob-detection-and-Tracking](https://github.com/LehighBlimpGroup/Blob-detection-and-Tracking) onto your device
3. Clone this repository onto your local device
4. Python is required to run these programs
5. This program was done with the Arduino Nicla Vision

## Steps for obtaining LAB color space information
1. Connect your Nicla Vision to your device. Open the Openmv IDE and connect your device with the IDE using the connect button on the bottom left.
2. Navigate to [Blob-detection-and-Tracking](https://github.com/LehighBlimpGroup/Blob-detection-and-Tracking) and open the get_gains.py file on the Openmv IDE.
3. Run the get_gains.py program that is located in the root of the folder. In the serial terminal, the RGB Gains value should be printed. This should look something like: [64, 64, 64].
4. Copy that value onto your clipboard. Open the file perception-subsystem.py, and past the rgb gain values on this portion of the code on line 51:  
```
elif board == NICLA:
    R_GAIN, G_GAIN, B_GAIN = [64, 64, 64]
```
5. Go futher down to line 70 and change PRINT_CORNERS to True:
```
PRINT_CORNER = False
```
6. Run perception-subsystem.py and direct the camera so that the target color encompasses the entirety of the 2x2 grid displayed on the screen. In the serial terminal, you can see the tuple values printed. Hold the camera in place for 30 seconds to 2 minutes based on desired accuracy.
7. Once complete, copy the tuple values into one of the .txt files in the [balltraining](https://github.com/LehighBlimpGroup/balltraining/tree/main). To obtain better accuracy, it is recommended to collect at least 1500 lines of data.
8. Run the gaussian_manual_multi.py file. The resulting graph should show the color space of the desired color in the LAB space.
