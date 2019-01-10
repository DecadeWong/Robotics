import cv2, os
from detector import myAlgorithm

folder = "testset"
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))

#img = cv2.imread('testset/003.png')
    blX, blY, trX, trY, d = myAlgorithm(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
