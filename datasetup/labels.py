import os
import cv2
import pandas as pd


labels = []
df = pd.read_csv(r"C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed\datalabels.csv")
i = 0
input("Did you change the digit number?")
while True:
    try:
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        fname = "frames/v5frame" + str(i) + ".jpg"
        print(fname)
        img = cv2.imread(fname)
        imS = cv2.resize(img, (1500, 1500))
        cv2.imshow("frame", imS)

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