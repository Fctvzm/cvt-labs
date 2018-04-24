import cv2
import numpy as np
from random import randint, uniform
import string, random
import tools as tl


def addNoise(image):    
    row,col = image.shape
    s_vs_p = 0.4
    amount = 0.01
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out


def addLines(img):
    for i in range(randint(0,2)):
        y1 = randint(0, img.shape[0])
        y2 = randint(0, img.shape[0])
        cv2.line(img, (0, y1), (img.shape[1], y2), 0, 1)


def addBlur(img):
    kw = randint(3, 7)
    kh = randint(3, 7)

    return cv2.blur(img, (kw, kh))


def text_generator(size = 8, chars = string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def addText(img):
    font = randint(0, 5)
    size = uniform(2.5, 3.5)
    text = text_generator(randint(5, 10))
    line_size = randint(1, 3)

    cv2.putText(img, text, (10, img.shape[0] - 15), font, size, (0, 0, 255), line_size, cv2.LINE_AA)

    return text


def genSmallTextImg(lines = False):
    genImg = np.full((100, 800), 255, dtype= np.uint8)

    text = addText(genImg)

    if randint(0, 1):
        genImg = addNoise(genImg)
        
    if lines:
        addLines(genImg)

    if randint(0, 1):
        genImg = addBlur(genImg)


    return genImg, text


def getGradient(gray, x = 0, y = 0, useGradient = True):
    if useGradient:
        grad = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=x, dy=y, ksize=5)
        grad = np.absolute(grad)

        (minVal, maxVal) = (np.min(grad), np.max(grad)) 
        if maxVal - minVal > 0:
            grad = (255 * ((grad - minVal) / float(maxVal - minVal))).astype("uint8")
        else:
            grad  = np.zeros(gray.shape, dtype = "uint8")

    else:
        grad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

    return grad

def median(img, projection, drawedVerp):
    height = drawedVerp.shape[0]
    val = height - int(np.max(projection) * 0.3)
    sliceLine = drawedVerp[val:val + 3, :]
    cv2.imshow("Slice", sliceLine)
    im2, contours, hierarchy = cv2.findContours(cv2.cvtColor(sliceLine, cv2.COLOR_BGR2GRAY), 
                                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    width = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if w > 4:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
            width.append(w)
    

    median = int(np.median(np.asarray(width)) * 0.5)
    return median

def byThreshold(img):
    grad = getGradient(img, useGradient = False)
    im2, contours, hierarchy = cv2.findContours(grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 1)
    return tl.concat_ver((img, grad))

def byGradient(img):
    grad = getGradient(img, y = 1, useGradient = True)
    proj = np.sum(grad, axis = 0) / 255
    drawProj = tl.getDrawProjectionHor(img, proj)
    y = img.shape[0]
    #medVal = median(img, proj, drawProj)
    #print medVal
    bandP1ranges = []
    peaks = []
    height = drawProj.shape[0]
    limit = int(np.max(proj) * 0.25)
    while  np.max(proj) > limit:
        ybm = np.argmax(proj)
        c1 = 0.2
        c2 = 0.2
        yb0 = tl.findb0(proj,  ybm, 
                        c1 * proj[ybm])
        yb1 = tl.findb1(proj, ybm, 
                        c2 * proj[ybm])

        if yb1 - yb0 > 4:
            bandP1ranges.append((yb0,yb1))
            peaks.append((int(ybm), height - proj[ybm]))

        proj[yb0:yb1] = 0

    # draw peaks
    #for peak in peaks:
        #cv2.circle(drawProj, peak, 1, (0,0,255))

    images = []
    for band in bandP1ranges:
        x1, x2 = band
        crop = img[0:y, x1:x2]
        #crop = cv2.resize(crop, (128, 128), 0, 0, cv2.INTER_LINEAR)
        images.append(crop)
        #print(crop)
        #cv2.line(img, (x1, 0), (x1, 100), (0, 0, 255), 1)
        #cv2.line(img, (x2, 0), (x2, 100), (0, 0, 255), 1)

    return images

