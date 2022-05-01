import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time

mp_drawing=mp.solutions.drawing_utils
mp_pose=mp.solutions.pose




cap = cv2.VideoCapture('KneeBendVideo.mp4')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
result = cv2.VideoWriter('vidop1.mp4', fourcc, 20.0, (frame_width,frame_height))
# ret, frame = cap.read()
# print(type(frame))
# plt.imshow(frame)
# plt.show()
# yy1=input()


#Calculating the angle
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle >180.0:
        angle = 360-angle
    return angle 

framecount=0
secs=0
startflag1=0
secrep=0
repcount=0
faultstart1=0
faulttime=0
relflag=0



with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    anglearr1=np.array([])
    while cap.isOpened():
        ret, frame = cap.read()
        if(ret):
            framecount+=1
            if(framecount%25==0):
                secs+=1
            # print(frame.shape)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            results = pose.process(image)
            
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                if landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].z<landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].z:
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]#23
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]#25
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]#27
                else:
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]#23
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]#25
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]#27
                # Calculate angle
                angle = calculate_angle(hip, knee, ankle)
                anglearr1=np.append(anglearr1,angle)
                if relflag==1:
                    if angle>140:
                        reltime=secs-secrel
                        if reltime>1:
                            relflag=0
                if angle<140 and relflag==0:
                    if startflag1==0:
                        startflag1=1
                        secrep=0
                        currsec=secs
                    else:
                        if faulttime<0:
                            faulttime=0
                        secrep=secs-currsec-faulttime
                        faultstart1=0
                        faulttime=0
                    if secrep>1:
                        # relflag=0
                        faultstart1=0
                        faulttime=0
                    if secrep>=8:
                        repcount+=1
                        # t_end = time.time() + 5
                        # while time.time() < t_end:
                        startflag1=0
                        secrep=0
                        relflag=1
                        secrel=secs
                if angle>140 and secrep<8 and relflag==0 and secrep>1:
                    # cv2.putText(image, 'Keep your knee bent', (150,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                    if faultstart1==0:
                        currfault=secs
                        faulttime=0
                        faultstart1=1 
                    else:
                        faulttime=secs-currfault
                        # cv2.putText(image, 'faulttime', (400,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                        # cv2.putText(image,str(faulttime), (750,300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
                    if faulttime>1:
                        cv2.putText(image, 'Keep your knee bent', (150,350), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,255), 2, cv2.LINE_AA)
                cv2.putText(image, 'current second in rep', (400,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(image, str(secrep), (750,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
                if relflag==1:
                    cv2.putText(image, 'stretch leg straight', (150,250), cv2.FONT_HERSHEY_SIMPLEX,2, (0,0,200), 2, cv2.LINE_AA)
                    
                # else:
                #     startflag1=0


            except:
                pass
                # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                        ) 
            cv2.putText(image, 'REPS', (200,100), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(image, str(repcount), 
                        (370,100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 2, cv2.LINE_AA)
                        
            cv2.imshow('Mediapipe Feed', image)
            result.write(image)
            # plt.plot(anglearr1)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
    plt.plot(anglearr1)
    plt.show()
    np.save('angledata1.npy',anglearr1)
    cap.release()
#     cv2.destroyAllWindows()
    result.release()
result.release()


        
    