import cv2
import numpy as np
import math
from convert_to_tensor import predict_on_frames
from rnn_predictor import predictor

cap = cv2.VideoCapture(0)

saved_frames = []

while(1):
        
    try:  #an error comes if it does not find anything in window as it cannot find contour of max area
          #therefore this try error statement
          
        ret, frame = cap.read()
        frame=cv2.flip(frame,1)
        kernel = np.ones((3,3),np.uint8)
        
        x1,y1 = 100,100
        x2,y2 = 300,300

        #define region of interest
        roi=frame[x1:x2, y1:y2]
        
        
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),0)    
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # define range of skin color in HSV
        lower_skin = np.array([0,20,70], dtype=np.uint8)
        upper_skin = np.array([20,255,255], dtype=np.uint8)
        
     #extract skin colur imagw  
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        res = cv2.bitwise_and(roi,roi, mask= mask) 
        # res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        image = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
        color = tuple(reversed((0,0,0)))
        image[:] = color
        
        h1, w1 = image.shape[:2]
        h2, w2 = res.shape[:2]

        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

        #combine 2 images
        vis[:h1, :w1,:3] = image
        vis[y1:y2, x1:x2,:3] = res

        # res = np.concatenate((image, res), axis=0)
        # res = cv2.resize(res,(299,299))
        vis = cv2.cvtColor(cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow('roi',roi)
        cv2.imshow('res',vis) # Training frame 
        
        saved_frames.append(vis)
        if len(saved_frames) == 101:
            # res.reshape()
            # print(np.array(saved_frames).shape,res.shape)
            image_weights = predict_on_frames(np.array(saved_frames))
            predictor(image_weights)
            break
            saved_frames = []

        cv2.putText(frame,'',(50,50), font, 2, (0,0,255), 3, cv2.LINE_AA)
        test = []
        #show the windows
        cv2.imshow('frame',frame)
    except Exception as e:
        print("error",e)
        break
        
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
cv2.destroyAllWindows()
cap.release()    