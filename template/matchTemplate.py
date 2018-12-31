import cv2
import numpy as np

# ? imread(filename[, flags]) -> retval
# * loads the templete image
template = cv2.imread('template.png', 0)
w, h = template.shape[::-1]

# ? imread(filename[, flags]) -> retval
# * loads the testing image
img_rgb = cv2.imread('test_image.png')

# ? cvtColor(src, code[, dst[, dstCn]]) -> dst
# * Converting testing image to grascale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# ? matchTemplate(image, templ, method[, result[, mask]]) -> result
# * The function slides through image , compares the overlapped patches of size
# * Fiding all sections of the image matching the templete
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
# ? Return elements, either from x or y, depending on condition
loc = np.where(res >= threshold)

# * draw rectanges for all occurace of the template
for point in zip(*loc[::-1]):
    # ? rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
    cv2.rectangle(img_rgb, point,
                  (point[0] + w, point[1] + h), (0, 0, 255), 2)

# ? imshow(winname, mat) -> None
# * displaying the modified testing image
cv2.imshow('Template Matching', img_rgb)

# * close all window when any buttong is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
