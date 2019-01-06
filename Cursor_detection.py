import numpy as np
import cv2 as cv

# This funciton initialises the empty matrix with zero values in it


def initialise_matrix(row, col):
    matrix = [[0 for x in range(col)] for y in range(row)]
    return matrix

# This function is used for applying the sobel filter for given Image


def getSobel(Image):
    sobelx = cv.Sobel(Image, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(Image, cv.CV_64F, 0, 1, ksize=3)
    OP = initialise_matrix(len(sobelx), len(sobelx[0]))
    for i in range(len(sobelx)):
        for j in range(len(sobelx[0])):
            OP[i][j] = (((sobelx[i][j]**2)+(sobely[i][j]**2))**0.5)

    return OP


def setA():
    # this list gets input files
    listA = ["neg_1", "neg_2", "neg_3", "neg_4", "neg_5", "neg_6", "neg_8", "neg_9", "neg_10", "pos_1",
             "pos_2", "pos_3", "pos_4", "pos_5", "pos_6", "pos_7", "pos_8", "pos_9", "pos_10", "pos_11", "pos_12", "pos_13", "pos_14", "pos_15"]

    template = cv.imread("task3/template.png", cv.IMREAD_GRAYSCALE)
    resize_template = cv.resize(template, None, fx=0.55, fy=0.55)
    blur_template = cv.GaussianBlur(resize_template, (3, 3), 0)

    for filename in listA:
        file = "./task3/"+filename+".jpg"
        img = cv.imread(file)
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        w, h = resize_template.shape[::-1]
        blurredimage = cv.GaussianBlur(gray_img, (3, 3), 0)
        # cv.imshow("gray",gray_img)
        # cv.imshow("gauss_blur",blurredimage)
        # cv.imshow("gauss_blur_template",blur_template)

        laplacian = cv.Laplacian(blurredimage, 10)
        lap_template = cv.Laplacian(blur_template, 10)
        #cv.imshow("laplacian", laplacian)
        # cv.imshow("lap_template",lap_template)

        #sobelImage = getSobel(gray_img)
        #sobelTemplate = getSobel(blur_template)
        #cv.imwrite("t_sample.jpg", np.asarray(sobelImage))
        #cv.imwrite("template_sample.jpg", np.asarray(sobelTemplate))
        result = cv.matchTemplate(np.asarray(laplacian).astype(np.float32), np.asarray(
            lap_template).astype(np.float32), cv.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.55)
        #cv.imshow("result", result)

        for pt in zip(*loc[::-1]):
            cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)
        fileop = filename+"_op.jpg"
        cv.imwrite(fileop, img)


def setB():
    
    listB = ["neg_1", "neg_2", "neg_3", "neg_4", "neg_5", "neg_6", "neg_8", "neg_9", "neg_10","neg_11","neg_12", "t1_1",
             "t1_2","t1_3","t1_4","t1_5","t1_6","t2_1","t2_2","t2_3","t2_4","t2_5","t2_6","t3_1","t3_2","t3_3","t3_4","t3_5","t3_6"]

    template = cv.imread("task3/template.png", cv.IMREAD_GRAYSCALE)
    resize_template = cv.resize(template, None, fx=0.55, fy=0.55)
    blur_template = cv.GaussianBlur(resize_template, (3, 3), 0)

    template_compass = cv.imread("task3_bonus/compass.jpg", cv.IMREAD_GRAYSCALE)
    #resize_template_1 = cv.resize(template_compass, None, fx=0.70, fy=0.70)
    resize_template_1 = template_compass
    blur_template_1 = cv.GaussianBlur(resize_template_1, (3, 3), 0)
    
    template_hand = cv.imread("task3_bonus/handtool.jpg", cv.IMREAD_GRAYSCALE)
    #resize_template_1 = cv.resize(template_compass, None, fx=0.70, fy=0.70)
    resize_template_2 = template_hand
    blur_template_2 = cv.GaussianBlur(resize_template_2, (3, 3), 0)
    

    for filename in listB:
        file = "./task3_bonus/"+filename+".jpg"
        img = cv.imread(file)

        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        w, h = resize_template.shape[::-1]
        blurredimage = cv.GaussianBlur(gray_img, (3, 3), 0)

        w1, h1 = resize_template_1.shape[::-1]
        w2, h2 = resize_template_2.shape[::-1]
        # cv.imshow("gray",gray_img)
        # cv.imshow("gauss_blur",blurredimage)
        # cv.imshow("gauss_blur_template",blur_template)

        laplacian = cv.Laplacian(blurredimage, 10)
        lap_template = cv.Laplacian(blur_template, 10)
        lap_template_1 = cv.Laplacian(blur_template_1, 10)
        lap_template_2 = cv.Laplacian(blur_template_2, 10)
    #cv.imshow("laplacian", laplacian)
    # cv.imshow("lap_template",lap_template)

    #sobelImage = getSobel(gray_img)
    #sobelTemplate = getSobel(blur_template)
    #cv.imwrite("t_sample.jpg", np.asarray(sobelImage))
    #cv.imwrite("template_sample.jpg", np.asarray(sobelTemplate))
        result = cv.matchTemplate(np.asarray(laplacian).astype(np.float32), np.asarray(lap_template).astype(np.float32), cv.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.50)
        #cv.imshow("result", result)

        for pt in zip(*loc[::-1]):
            cv.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 3)

        result1 = cv.matchTemplate(np.asarray(laplacian).astype(np.float32), np.asarray(lap_template_1).astype(np.float32), cv.TM_CCOEFF_NORMED)
        loc_1 = np.where(result1 >= 0.50)
        
        for pt in zip(*loc_1[::-1]):
            cv.rectangle(img, pt, (pt[0] + w1, pt[1] + h1), (0, 255, 0), 3)

        result2 = cv.matchTemplate(np.asarray(laplacian).astype(np.float32), np.asarray(lap_template_2).astype(np.float32), cv.TM_CCOEFF_NORMED)
        loc_2 = np.where(result2 >= 0.60)
        
        for pt in zip(*loc_2[::-1]):
            cv.rectangle(img, pt, (pt[0] + w2, pt[1] + h2), (0, 255, 0), 3)
        
        fileop = filename+"_op.jpg"
        cv.imwrite(fileop, img)


setA()
setB()
