import numpy as np 
import pandas as pd
import cv2
from skimage.feature import graycomatrix,graycoprops
import xlsxwriter as xw
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import classification_report


book = xw.Workbook('feature.xlsx')
sheet = book.add_worksheet()
sheet.write(0,0,'Keterangan')

kolom = 1
#kolom feature glcm
glcm_feature = ['correlation','homogeneity','dissimilarity', 'contrast', 'energy', 'ASM']
sudut = ['0','45','90','135']
for i in glcm_feature :
    for j in sudut :
        sheet.write (0,kolom,i+' '+j)
        kolom+=1


#baris citra
dataset = os.listdir('dataset')
jenis_data = ['positif','negatif']

row = 1
kernel = cv2.getGaborKernel((8,8), sigma=3,theta=25, lambd=7.22, gamma=0.5, psi=0)
for i in dataset :
        column = 0
        file_name = 'dataset/'+i
        # print(file_name)
        if i.startswith('positif'):
            sheet.write(row,column,'positif')
        else :
            sheet.write(row,column,'negatif')
        column+=1

        img = cv2.imread(file_name)
        imgr = cv2.resize (img, (400,400))
        #grayscale gambar
        grayscale= cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
        #Pasang Filter Gabor
        fimg = cv2.filter2D(grayscale,-1, kernel)

        #glcm
        distances = [5]
        angles = [0,np.pi/4,np.pi/2,3*np.pi/4]
        levels = 256
        symetric = True
        normed = True

        glcm = graycomatrix (fimg, distances, angles, levels, symetric, normed)
        glcm_props = [propery for name in glcm_feature for propery in graycoprops(glcm,name)[0]]
        for item in glcm_props :
            sheet.write (row,column,item)
            column+=1

        row+=1





# data =pd.read_excel ('feature.xlsx', sheet_name='Sheet1')
# enc = LabelEncoder()

# data['Keterangan'] = enc.fit_transform (data['Keterangan'].values)
# atr_data = data.drop(columns='Keterangan')
# cls_data = data['Keterangan']

# model = SVC(kernel='linear')
# xtrain,xtest, ytrain, ytest = train_test_split (atr_data, cls_data, test_size = 0.2, random_state = 2)

# #Training data
# model.fit(xtrain,ytrain)

# #Hasil Prediksi
# hasilPrediksi = model.predict(xtest)
# prediksiBenar = (hasilPrediksi == ytest).sum()
# prediksiSalah = (hasilPrediksi != ytest).sum()
# print ("Prediksi yang benar :", prediksiBenar)
# print ("Prediksi Yang Salah :", prediksiSalah)
# print("akurasi: ", "%.2f"%(prediksiBenar/(prediksiBenar+prediksiSalah)*100),'%')
# print(classification_report(ytest, hasilPrediksi))


# # img = cv2.imread("positif 1.jpg")
# # grayscale= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # fimg = cv2.filter2D(grayscale,-1, kernel)

# # ret, img1 = cv2.threshold(fimg,230,255,cv2.THRESH_BINARY_INV)
# # img1 = cv2.dilate(img1.copy(), None, iterations=1)
# # img1 = cv2.erode (img1.copy(), None, iterations=1)
# # b,g,r = cv2.split(img)
# # rgba = [b,g,r,img1]
# # dst = cv2.merge (rgba,4 )


# cv2.imshow("img1",img) 
# cv2.imshow("Filter", fimg)
# cv2.waitKey()





