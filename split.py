import cv2
vidcap = cv2.VideoCapture('speed.mp4')
success,image = vidcap.read()
count = 0
while success:
  #cv2.imwrite("C:\\Users\\erik\\OneDrive\\Cross-Code\\VSCode\\PythonCode\\Starting_Code\\Machine Learning\\pytorchlearning\\Speed\\Frames\\v1frame%d.jpg" % count, image)     # save frame as JPEG file      
  cv2.imwrite("frames/v2frame%d.jpg" % count, image)     # save frame as JPEG file      

  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1