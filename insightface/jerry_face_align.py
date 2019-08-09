#coding=utf-8
# from deploy.mtcnn_detector import MtcnnDetector

from RetinaFace.retinaface import RetinaFace
import numpy as np
import cv2
import os
from common.face_align import norm_crop
import shutil

#用拉普拉斯算子计算模糊读，实际测试中对与我们112*112的人脸效果很差，没有采用
def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

#人脸对齐并将人脸抽取到文件夹中 处理智慧树给的数据
def face_detect_and_aligin():
    # detector = MtcnnDetector('/home/shtf/PycharmProjects/insightface_20190529/deploy/mtcnn-model')
    detector = RetinaFace('./models/RetinaFace_Face_Detector/resnet-50', 0, 0000, 'net3')
    #100110685
    base_dir = '/e/company_file/九江银行/九江银行-初始分类'
    to_base_dir = '/e/company_file/九江银行/face_images/align_faces'
    dirs = os.listdir(base_dir)
    count = 1
    for dir in dirs:
        if dir=='视频' or dir == '音频':
            continue
        print('go image dir:',dir.decode('utf-8'))
        from_dir = os.path.join(base_dir,dir)
        # to_dir = os.path.join(to_base_dir,dir)
        # if not os.path.exists(to_dir):
        #     os.mkdir(to_dir)
        images = os.listdir(from_dir)
        for image in images:
            image_path = os.path.join(from_dir,image)
            img = cv2.imread(image_path)
            # result = detector.detect_face(img)
            # print(image_path)
            if type(img)!=np.ndarray:
                print(image_path)
                continue
            result = detector.detect(img, 0.997, scales=[1], do_flip=False)
            one, two = result
            if len(one)==0:
                continue
            # one = np.asarray(one,np.int32)
            lands = np.asarray(two,np.int32)
            for land,bbox in zip(lands,one):
            # cv2.rectangle(img,(one[0][0],one[0][1]),(one[0][2],one[0][3]),(255,255,0))
            # cv2.circle(img,(two[0][0],two[0][5]),3,(255,0,0))

            # for i in range(5):
            #     cv2.circle(img,(two[0][i],two[0][5+i]),3,(255,0,0))
            #     land = np.reshape(land,(2,5))
                # land = np.transpose(land,(1,0))
                crop_img = norm_crop(img,land)
                if np.sum(crop_img)==0:
                    continue
                # cv2.putText(crop_img,'w:'+str(int(bbox[2]-bbox[0])),(0,30),2,1,(255,0,0))
                # cv2.putText(crop_img,'l:'+str(int(variance_of_laplacian(crop_img))),(0,90),2,1,(255,0,0))
                cv2.imwrite(os.path.join(to_base_dir, str(int(bbox[2]-bbox[0]))+'-'+str(count) + '.jpg'), crop_img)
                count += 1
                # cv2.imshow("b",crop_img)
                # cv2.putText(img,'%.3f'%(bbox[4]),(int(bbox[0]),int(bbox[1])),2,2,(255,0,0))
            # cv2.imshow("a",img)
            # cv2.waitKey(0)
            # cv2.imwrite(os.path.join(to_dir,str(count)+'.jpg'),img)


#重命名文件名，按序号排序
def imaga_rename():
    # base_path = '/e/company_file/bbtree/faces_width40'
    # dirs = os.listdir(base_path)
    # for dir in dirs:
    # print('go image dir:', dir)
    # item_dir = os.path.join(base_path, dir)
    item_dir = '/e/company_file/九江银行/face_images/align_faces'
    count = 1
    images = os.listdir(item_dir)
    for image in images:
        os.rename(os.path.join(item_dir, image), os.path.join(item_dir, str(count) + '.jpg'))
        count += 1

#将人脸宽度大于等于固定数值的图像筛选出来
def filter_image_witdh(width):
    base_dir_from = '/e/company_file/bbtree/faces'
    base_dir_to = '/e/company_file/bbtree/faces_width40'
    dirs = os.listdir(base_dir_from)
    for dir in dirs:
        if dir == '100110685':
            continue
        print('go image dir:', dir)
        from_dir = os.path.join(base_dir_from, dir)
        to_dir = os.path.join(base_dir_to, dir)
        if not os.path.exists(to_dir):
            os.mkdir(to_dir)
        images = os.listdir(from_dir)
        count = 1
        for image in images:
            if int(image.split('-')[0])>=width:
                shutil.copy(os.path.join(from_dir,image),os.path.join(to_dir,image))
# imaga_rename()
# filter_image_witdh(40)

face_detect_and_aligin()