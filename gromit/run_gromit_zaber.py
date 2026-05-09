'''
A script to run Gromit with the Zaber stages and save images. 
A Basler pylon camera is used for image acquisition.

'''

from pypylon import pylon
import numpy as np
from time import sleep
from zaber_motion import Units
from zaber_motion.ascii import Connection
from imageio import imwrite
import os


def main():

    # ================= USER SETTINGS =================
    PORT = "COM5"
    NMEAS = 24
    PSG_START = 32.85
    PSA_START = 8.3

    EXPOSURE_TIME = 200.0   # microseconds
    GAIN = 0.0              # dB
    SAVE_IMAGES = True
    SAVE_DIR = "gromit_data_reduction/data/3_26_26_dichroic225_data/"
    # =================================================
    
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ================= CAMERA SETUP ==================
    print("Initializing camera...")
    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice()
    )
    camera.Open()

    # Disable auto settings
    camera.ExposureAuto.SetValue("Off")
    camera.GainAuto.SetValue("Off")

    camera.ExposureTime.SetValue(EXPOSURE_TIME)
    camera.Gain.SetValue(GAIN)

    # Force 8-bit output
    camera.PixelFormat.SetValue("Mono8")

    # Enable software trigger mode
    camera.TriggerSelector.SetValue("FrameStart")
    camera.TriggerMode.SetValue("On")
    camera.TriggerSource.SetValue("Software")

    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)


    # ================= STAGE SETUP ===================
    print("Initializing stages...")
    connection = Connection.open_serial_port(PORT)

    device_list = connection.detect_devices()
    print("Found {} devices".format(len(device_list)))


    device1 = device_list[0]
    device2 = device_list[1]
    axis1 = device1.get_axis(1)
    # home rotation stages and move to starting positions
    axis1.home()
    axis1.move_absolute(PSG_START, units=Units.ANGLE_DEGREES, velocity=5.2, velocity_unit=Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
    print("zaber1 set")
    axis2 = device2.get_axis(1)
    axis2.home()
    axis2.move_absolute(PSA_START, units=Units.ANGLE_DEGREES, velocity=5.2, velocity_unit=Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
    print("zaber2 set")

    psg_angles = np.linspace(PSG_START, PSG_START + 180, NMEAS+1)
    psa_angles = np.linspace(PSA_START, PSA_START + 900, NMEAS+1)
    print("PSG angles:", psg_angles)
    print("PSA angles:", psa_angles)

    image_list = []

    print("Starting acquisition loop...")

    for i in range(NMEAS+1):

        if i != 0:

            # Move PSG
            axis2.move_absolute(psg_angles[i], units=Units.ANGLE_DEGREES, velocity=5.2, velocity_unit=Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)
            # Move PSA
            axis1.move_absolute(psa_angles[i], units=Units.ANGLE_DEGREES, velocity=5.2, velocity_unit=Units.ANGULAR_VELOCITY_DEGREES_PER_SECOND)

        # -------- software triggered image --------
        camera.ExecuteSoftwareTrigger()

        grabResult = camera.RetrieveResult(
            5000, pylon.TimeoutHandling_ThrowException
        )

        if grabResult.GrabSucceeded():
            img = grabResult.Array  # uint8

            print(
                f"Frame {i+1}/{NMEAS+1}: mean={np.mean(img):.2f}, "
                f"min={img.min()}, max={img.max()}"
            )

            image_list.append(img.copy())

            if SAVE_IMAGES:
                filename = f"{i+1}.tiff"
                imwrite(os.path.join(SAVE_DIR, filename), img)

        grabResult.Release()


    #Cleanup
    camera.StopGrabbing()
    camera.Close()
    connection.close()
    print("Acquisition complete.")


if __name__ == "__main__":
    main()