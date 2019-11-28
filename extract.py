#import the leabrary
import cv2 
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread("test5.jpg")

image = imutils.resize(img,500)

cv2.imshow("image",image)
#cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image GRAY",gray)
#cv2.waitKey(0)

gray = cv2.bilateralFilter(gray,11,17,17)
cv2.imshow("image GRAY bilateralFilter",gray)
#cv2.waitKey(0)

edges = cv2.Canny(gray,170,200)
cv2.imshow("Canny ",edges)

cnts,new = cv2.findContours(edges.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

img1 = image.copy()
cv2.drawContours(img1,cnts,-1,(0,255,0),3)

cv2.imshow("all contours ",img1)

cnts = sorted(cnts,key=cv2.contourArea,reverse = True)[:30]
NumberPlatecnt = None

#top 30 contours 
img2 = image.copy()
cv2.drawContours(img2,cnts,-1,(0,255,0),3)
cv2.imshow("top 30 contours",img2)




count = 0 
idx = 7

for c in cnts :
	peri =cv2.arcLength(c,True)
	approx = cv2.approxPolyDP(c,0.02 * peri,True)
	if len(approx) == 4:
		NumberPlatecnt = approx
		x, y,w,h = cv2.boundingRect(c)
		new_img = image[y:y +h,x:x +w]
		new_img = gray[y:y +h,x:x +w]
		cv2.imwrite('Cropped_Image_Text/'+str(idx)+'.png',new_img)
		idx+=1
		break

cv2.drawContours(image,[NumberPlatecnt],-1,(0,255,0),5)
cv2.imshow("finale image with plate detector",image)

Cropped_image_loc = 'Cropped_Image_Text/7.png'
cv2.imshow('Cropped Image',cv2.imread(Cropped_image_loc))

test = pytesseract.image_to_string(Cropped_image_loc,lang='eng')
print(test)
cv2.waitKey(0)