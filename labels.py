import os
import cv2
import pandas as pd


labels = []
df = pd.DataFrame(columns=["dest","clickp"])
for i in range(len(os.listdir("frames"))):
    fname = "frames/v1frame" + str(i) + ".jpg"
    print(fname)
    img = cv2.imread(fname)
    cv2.imshow("frame", img)

    k = chr(cv2.waitKey()) 
    if k == "p":
        df.loc[len(df)] = [fname, 0]
    elif k == "a":
        df.loc[len(df)] = [fname, 1]

#df.to_csv("datalabels.csv", index=False)
cv2.destroyAllWindows()
