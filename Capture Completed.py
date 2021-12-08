import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
import easyocr
key = cv2.waitKey(1)
webcam = cv2.VideoCapture(0)
while True:
    try:
        check, frame = webcam.read()
        print(check) 
        print(frame) 
        cv2.imshow("Capturing",frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            cv2.imwrite(filename='saved_img.jpg', img=frame)
            webcam.release()
            img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
            img_new = cv2.imshow("Captured Image", img_new)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
            img = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            check = False
            bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
            edged = cv2.Canny(bfilter, 30, 200) 
            keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(keypoints)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            location = None
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 10, True)
                if len(approx) == 4:
                    location = approx
                    break
            mask = np.zeros(gray.shape, np.uint8)
            new_image = cv2.drawContours(mask, [location], 0,255, -1)
            new_image = cv2.bitwise_and(img, img, mask=mask)
            (x,y) = np.where(mask==255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2+1, y1:y2+1]
            reader = easyocr.Reader(['en'])
            result = reader.readtext(cropped_image)
            print(result)
            text = result[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)
            break
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break

    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

cv2.imwrite(filename='saved_img1.jpg', img=cv2.cvtColor(res, cv2.COLOR_BGR2RGB))