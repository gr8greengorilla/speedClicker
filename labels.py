import os
import cv2
import pandas as pd


labels = []
df = pd.read_csv("datalabels.csv")
i = 0
while True:
    try:
        fname = "frames/v2frame" + str(i) + ".jpg"
        print(fname)
        img = cv2.imread(fname)
        cv2.imshow("frame", img)

        k = chr(cv2.waitKey()) 
        if k == "p":
            df.loc[len(df)] = [fname, 0]
        elif k == "a":
            df.loc[len(df)] = [fname, 1]
        
        i += 1
    except:
        break

df.to_csv("datalabels2.csv", index=False)
cv2.destroyAllWindows()