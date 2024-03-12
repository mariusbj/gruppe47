import cv2

video = cv2.VideoCapture("testvid.mov")

frame_nr = 1

while(True):
   ret,frame = video.read()
   if ret:
      name = str(frame_nr) + '.jpg'
      print (name)
      cv2.imwrite(name, frame)
      frame_nr += 1
   else:
      break

video.release()
cv2.destroyAllWindows()