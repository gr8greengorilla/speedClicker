import numpy as np

path = r"C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed\trainingdata\train10\posedata.npy"

t = np.load(path)

#t = t[:,0, :, :]

print(t.shape)

#t = np.save(path, t)