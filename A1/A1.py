import cv2, time

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://10.100.42.155:8080/video")

while(True):
    start_time = time.time()
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Part 4
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)
    cv2.circle(frame, max_loc, 10, (0, 255, 255), 2)

    #Part 5
    red_channel = frame[:, :, 2]
    min_val_red, max_val_red, min_loc_red, max_loc_red = cv2.minMaxLoc(red_channel)
    cv2.circle(frame, max_loc_red, 10, (0, 0, 255), 2)

    '''
    #Part 6
    brightest_val = 0
    brightest_loc = (0, 0)
    reddest_val = 0
    reddest_loc = (0, 0)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j] > brightest_val:
                brightest_val = gray[i, j]
                brightest_loc = (j, i)

            if red_channel[i, j] > reddest_val:
                reddest_val = red_channel[i, j]
                reddest_loc = (j, i)

    cv2.circle(frame, brightest_loc, 10, (0, 255, 255), 2)
    cv2.circle(frame, reddest_loc, 10, (0, 0, 255), 2)
    '''

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    elapsed_time = time.time() - start_time
    print(elapsed_time, ',')

cap.release()
cv2.destroyAllWindows()