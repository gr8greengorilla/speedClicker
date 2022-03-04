import numpy as np

t = np.load(r"trainingdata\train2\posedata.npy", allow_pickle=True)

#t = t[:,0,:,:]
print(t.shape)

#np.save(r"trainingdata\train2\posedata.npy", t)