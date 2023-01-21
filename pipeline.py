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
        imgs.append(img[x[1]+2:x[3]-2,x[0]+2:x[2]-2,:])
    return imgs
def filter_plate(img, bbox):
    imgs=[]
    for x in bbox:
        # if (x[3]- x[1]) <50 and (x[2]- x[0]) <50:
        #     continue
        imgs.append(img[x[1]+2:x[3]-2,x[0]+2:x[2]-2,:])
    return imgs
def run():
    img_input = sorted(glob.glob("/Users/datle/Desktop/Official_license_plate/images/*.png"))
    # os.chdir("/Users/datle/Desktop/Official/image_vehicle")
    print(img_input)
    for x,img in enumerate(img_input):
        result, bbox= detect_vehicle(img, debug=False)
        if result is None and bbox is None:
            continue
        imgs=filter_vehicle(result, bbox)
        if len(imgs)==0:
            continue
        for y,img in enumerate(imgs):
            if img.shape[0]==0 or img.shape[1]==0:
                continue
            cv2.imwrite(f'/Users/datle/Desktop/Official_license_plate/image_vehicle/{x}_{y}.jpg', img)

    img_vehicle= sorted(glob.glob("/Users/datle/Desktop/Official_license_plate/image_vehicle/*.jpg"))
    for x,img in enumerate(img_vehicle):
        result, bbox= detect_plate(img, debug=False)
        if result is None and bbox is None:
            continue
        imgs=filter_plate(result, bbox)
        # os.chdir("/Users/datle/Desktop/Official/image_plate")
        if len(imgs)==0:
            continue
        for y,img1 in enumerate(imgs):
            if img1.shape[0] == 0 or img1.shape[1] == 0:
                continue
            cv2.imwrite(f'/Users/datle/Desktop/Official_license_plate/image_plate/{x}_{y}.jpg', img1)

run()





