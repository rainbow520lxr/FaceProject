from RetinaFace.retinaface import RetinaFace
import numpy as np
import cv2
import os
from common.face_align import norm_crop
import shutil
import time
import tqdm
import glob

def face_detect_and_aligin():
    # detector = MtcnnDetector('/home/shtf/PycharmProjects/insightface_20190529/deploy/mtcnn-model')
    detector = RetinaFace('./models/RetinaFace_Face_Detector/R50', 0, 0, 'net3')

    #100110685
    base_dir = '/home/lxr/ssd_dataset/raf/basic/original'
    to_base_dir = '/home/lxr/ssd_dataset/raf/basic/images'

    images = os.listdir(base_dir)

    count = 0
    detected = 0
    detecting = 0
    current_generate = 0
    for image in images:
        prefix, back = image.split('.')
        dest_flatten = os.path.join(to_base_dir, prefix+'*')
        if glob.glob(dest_flatten):
            # print glob.glob(dest_flatten)
            detected += 1
            # print image + 'is over!'
        else:
            image_path = os.path.join(base_dir, image)
            img = cv2.imread(image_path)
            # result = detector.detect_face(img)
            # print(image_path)
            if type(img) != np.ndarray:
                print(image_path)
                continue

            result = detector.detect(img, 0.997, scales=[1], do_flip=False)

            one, two = result
            if len(one) == 0:
                continue
            lands = np.asarray(two, np.int32)
            b = 0
            for land, bbox in zip(lands, one):
                crop_img = norm_crop(img, land)
                if np.sum(crop_img) == 0:
                    continue
                b += 1
                dest_path = os.path.join(to_base_dir, prefix+'-'+str(b)+'.'+back)
                # print dest_path
                cv2.imwrite(dest_path, crop_img)
                current_generate += 1
            detecting += 1
        count += 1
        print 'count: %d | detected: %d | detecting: %d | current_generate: %d' % (count, detected, detecting, current_generate)


if __name__ == '__main__':
    face_detect_and_aligin()