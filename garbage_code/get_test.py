import cv2
import os
import datetime

saveimg = False  # 保存图片开关


def saveROIImg(img):  # 保存图片函数
    global saveimg
    timestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S').replace(" ", "").replace("-", "").replace(":", "")
    path = '../total_datasets/test/'
    name = path + timestr + ".jpg"
    cv2.imwrite(name, img)
    print("Saved img:", name)
    saveimg = False
    return saveimg


def save_figure():
    saveimg = False
    cap = cv2.VideoCapture(0)  # 打开摄像头

    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 3)
        frame = cv2.resize(frame, (456,456))

        if ret == True:
            if saveimg == True:
                saveimg = saveROIImg(frame)

        cv2.putText(frame,'push key s to save samples',(10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2,1)
        cv2.imshow('Original',frame)

        key = cv2.waitKey(5) & 0xff
        if key == 27:  # Esc键退出
            # cap.release()
            # cv2.destroyAllWindows()
            return
        elif key == ord('s'):
            saveimg = True

if __name__ =="__main__":
    save_figure()
