import cv2

# Replace 'udp://@:5000' with your actual stream URL
cap = cv2.VideoCapture('udp://@192.168.1.24:5000')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Manipulate the frame with OpenCV here
    cv2.imshow('Video Stream', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()