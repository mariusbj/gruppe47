import cv2

video = cv2.VideoCapture("ENML_ENLK_Part2_RRTC_RTM92_HUD_hud_1.mp4")

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