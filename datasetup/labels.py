import cv2
import pandas as pd
import numpy as np
import mediapipe

mpPose = mediapipe.solutions.pose
pose = mpPose.Pose()
def getPose(img):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = pose.process(imgRGB).pose_landmarks.landmark

    #landmarks = [results[15],results[16],results[30],results[31]] + [results[x] for x in range(23,26)]

    inputs = [[x.x,x.y,x.z] for x in results]

    return inputs

labels = []
categories = []
df = pd.read_csv(r"C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed\datalabels.csv")
i = 0

path = df.at[5, "videopath"]
print(path)
vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()
count = 0

while success:
    print(count)
    count += 1
    #try:
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    imS = cv2.resize(image, (1500, 1500))
    cv2.imshow("frame", imS)

    try:
        landmarkdata = getPose(image)
        #landmarkdata = [landmarkdata[0], landmarkdata[1], landmarkdata[2]]

        k = chr(cv2.waitKey())
        if k == "p":
            #np.append([getPose(image), 0], labels)
            labels.append([landmarkdata])
            categories.append(0)
        elif k == "a":
            #np.append(labels, np.array([getPose(image), 1],dtype=object))
            labels.append([landmarkdata])
            categories.append(1)
        elif k == "m":
            break
        
        success,image = vidcap.read()


    except:
        success,image = vidcap.read()

    
    #except:
    #    break
output = np.array(labels)
categories = np.array(categories)
print(categories)
np.save("/".join(path.split("/")[0:-1]) + "/posedata", output)
np.save("/".join(path.split("/")[0:-1]) + "/categorydata", categories)