import cv2
vidcap = cv2.VideoCapture(r'C:\Users\erik\OneDrive\Cross-Code\VSCode\PythonCode\Starting_Code\Machine Learning\pytorchlearning\Speed\needtolabel2.mp4')
success,image = vidcap.read()
count = 0
input("Did you change the digit number?")
while success:
  #cv2.imwrite("C:\\Users\\erik\\OneDrive\\Cross-Code\\VSCode\\PythonCode\\Starting_Code\\Machine Learning\\pytorchlearning\\Speed\\Frames\\v1frame%d.jpg" % count, image)     # save frame as JPEG file      
  cv2.imwrite("frames/v6frame%d.jpg" % count, image)     # save frame as JPEG file      

  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1