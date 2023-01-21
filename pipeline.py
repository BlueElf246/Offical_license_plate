import glob
import os
import cv2
os.chdir("/Users/datle/Desktop/Official_license_plate")
from Training_license_plate_detection.run_sliding_window1 import run as detect_plate
from Training_vehicle_detection.run_sliding_window1 import run as detect_vehicle
def show(img):
    cv2.imshow('i', img)
    cv2.waitKey(0)
def filter_vehicle(img, bbox):
    imgs=[]
    for x in bbox:
        if (x[3]- x[1]) <30 and (x[2]- x[0]) <30:
            continue
        imgs.append(img[x[1]+5:x[3]-5,x[0]+5:x[2]-5,:])
    return imgs
def filter_plate(img, bbox):
    imgs=[]
    for x in bbox:
        if (x[3]- x[1]) <30 and (x[2]- x[0]) <30:
            continue
        imgs.append(img[x[1]+5:x[3]-5,x[0]+5:x[2]-5,:])
    return imgs
def run():
    img_input = sorted(glob.glob("/Users/datle/Desktop/Official_license_plate/images/*.png"))
    # os.chdir("/Users/datle/Desktop/Official/image_vehicle")
    for x,img in enumerate(img_input):
        result, bbox= detect_vehicle(img, debug=False)
        imgs=filter_vehicle(result, bbox)
        for img in imgs:
            cv2.imwrite(f'/Users/datle/Desktop/Official_license_plate/image_vehicle/{x}.jpg', img)

    img_vehicle= sorted(glob.glob("/Users/datle/Desktop/Official_license_plate/image_vehicle/*.jpg"))
    for x,img in enumerate(img_vehicle):
        result, bbox= detect_plate(img, debug=False)
        imgs=filter_vehicle(result, bbox)
        # os.chdir("/Users/datle/Desktop/Official/image_plate")
        for img in imgs:
            cv2.imwrite(f'/Users/datle/Desktop/Official_license_plate/image_plate/{x}.jpg', img)

run()





