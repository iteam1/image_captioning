import cv2 

cap = cv2.VideoCapture(0)

ret,frame = cap.read()

if __name__ == "__main__":
	
	while ret:

		ret,frame = cap.read()

		cv2.imshow('frame',frame)

		if cv2.waitKey(1) == 27:
			break

	cv2.destroyAllWindows()
	cap.release()