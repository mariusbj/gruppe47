import cv2

# Specify the video file
video = cv2.VideoCapture("ENML_ENLK_Part2_RRTC_RTM92_HUD_hud_1.mp4")

# Specify the start timestamp in seconds (e.g., start from 10 seconds)
start_time_in_seconds = 10

# Calculate the start time in milliseconds
start_time_in_milliseconds = start_time_in_seconds * 1000

# Set the starting position of the video
video.set(cv2.CAP_PROP_POS_MSEC, start_time_in_milliseconds)

frame_nr = 1

while True:
    ret, frame = video.read()
    if ret:
        name = str(frame_nr) + '.jpg'
        print(name)
        cv2.imwrite(name, frame)
        frame_nr += 1
    else:
        break

video.release()
cv2.destroyAllWindows()
