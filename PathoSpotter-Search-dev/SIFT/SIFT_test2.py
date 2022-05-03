import cv2
img1 = cv2.imread("PSAMILOIDOSE20210214-1.jpg",0)
img2 = cv2.imread("PSAMILOIDOSE20210214-2.jpg",0)
# check for similarities
sift = cv2.xfeatures2d.SIFT_create()
# check keypoints and descriptions of images
kp_1,desc_1 = sift.detectAndCompute(img1,None)
kp_2,desc_2 = sift.detectAndCompute(img2,None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_1, desc_2, k=2)
result = cv2.drawMatchesKnn(img1,kp_1,img2,kp_2,matches,None)
cv2.imshow("Correlation", result)
cv2.imshow("Image 1", img1)
cv2.imshow("Image 2", img2)