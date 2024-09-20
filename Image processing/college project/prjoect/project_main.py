# relevant imports
from tkinter import W
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import sys
import time


from transform import *
from threshold import *
from lane_detection import *
from Drawings import *


def rescaleFrame(frame, scale=0.75):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

# apply the function on videos
# capture is instance of the videocapture class that contains the video given

# input_path = 'project_video_Trim.mp4'
# output_path = "filename_002.mp4"


input_path = sys.argv[1]
output_path = sys.argv[2]

# debugging: 1 for debug mode
debugging = int(sys.argv[3])

capture = cv2.VideoCapture(input_path)

# Video Duration
fps = capture.get(cv2.CAP_PROP_FPS)
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps


isTrue, frame = capture.read()

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

writer = cv2.VideoWriter(output_path, fourcc, fps,
                         (frame.shape[1], frame.shape[0]))

output = binarization_choice2(frame)
BC2 = threshold(frame)
warped, m, minv = per_transform(output)

first_time, left_eqn, right_eqn = fit_polynomial(warped)
rectangle = draw_rectangle(frame, left_eqn, right_eqn)

i = 0
save = i
t1 = time.time()

while i < frame_count:
    isTrue, frame = capture.read()
    if not isTrue:
        print(
            f"\n the output video have been saved to {output_path} successfully")
        break

    # warped = transform(frame,m)
    # ------------------------------
    # apply the function on videos

    output = binarization_choice2(frame)
    BC2 = threshold(frame)

    warped = transform(output, m)
    output_2, left_eqn, right_eqn, ploty = search_around_poly(warped, left_eqn, right_eqn)

    curve = measure_curvature_pixels(left_eqn, right_eqn)
    first_time, left_eqn, right_eqn = fit_polynomial(warped)
    rectangle = draw_rectangle(frame, left_eqn, right_eqn)
    transformed_back = transform(output_2, minv)
    correct_rectangle = transform(rectangle, minv)
    cv.putText(frame, "curvature: {} m".format(curve), (255, 255), cv.FONT_ITALIC, 1.0, (255, 255, 255), 2)
    write_frame = cv.addWeighted(correct_rectangle, 0.5, frame, 1, 0)

    if debugging:
        first_time = rescaleFrame(first_time, 0.5)
        output = rescaleFrame(output, 0.5)
        warped = rescaleFrame(warped, 0.5)
        warped = cv.cvtColor(warped, cv.COLOR_GRAY2BGR)

        BC2 = rescaleFrame(BC2, 0.5)
        BC2 = cv.cvtColor(BC2, cv.COLOR_GRAY2BGR) * 255
        output = cv.cvtColor(output, cv.COLOR_GRAY2BGR) * 255

        write_frame = rescaleFrame(write_frame, 0.5)

        wider_frame = frame * 0
        wider_frame[0:wider_frame.shape[0] // 2,
                    0: wider_frame.shape[1] // 2] = first_time
        # [0, 1]
        wider_frame[0:wider_frame.shape[0] // 2,
                    wider_frame.shape[1] // 2: wider_frame.shape[1]] = BC2
        # [1, 0]
        wider_frame[wider_frame.shape[0] // 2: wider_frame.shape[0],
                    0: wider_frame.shape[1] // 2] = output
        # [1, 1]
        wider_frame[wider_frame.shape[0] // 2: wider_frame.shape[0],
                    wider_frame.shape[1] // 2: wider_frame.shape[1]] = write_frame

        writer.write(wider_frame)
        pass
        # writer.write(The big frame contains the step, refer to the one I used in HP tuner selected)
    else:
        writer.write(write_frame)
    i += 1
    if i < save:
        i = save + 1
    save = i

    t2 = divmod(time.time() - t1, 60)

    mins = round(t2[0])
    if mins < 10:
        mins = "0" + str(mins)

    secs = round(t2[1])
    if secs < 10:
        secs = "0" + str(secs)

    percentage = round(((i * 100 / fps) / duration), 1)

    loading = ("■" * int(percentage / 2)) + ("□" * (50 - int(percentage)))

    sys.stdout.write(f"\r{percentage}% time:{mins}:{secs} {loading}")
    sys.stdout.flush()
    time.sleep(0.01)
    # cv2.imshow('Video',warped)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

capture.release()
writer.release()
cv2.destroyAllWindows()
