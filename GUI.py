import tkinter as tk
from tkinter import filedialog
import numpy as np 
import pandas as pd
import cv2
from skimage.feature import graycomatrix,graycoprops
import xlsxwriter as xw
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

pd.set_option('mode.chained_assignment', None)

window = tk.Tk()


window.configure (bg='#B0C4DE')
window.geometry("1150x600")
window.resizable(False,False)
window.title ("KLASIFIKASI DBD")

def openImage():
    global fileImage
    fileImage = filedialog.askopenfilename()
    kernel = cv2.getGaborKernel((8,8), sigma=3,theta=25, lambd=7.22, gamma=0.5, psi=0)
    img = cv2.cvtColor(cv2.imread(fileImage),cv2.COLOR_BGR2RGB)
    # grayscale = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2GRAY)
    imgr = cv2.resize (img, (400,400))
    #grayscale gambar
    grayscale= cv2.cvtColor(imgr,cv2.COLOR_RGB2GRAY)
    
    #Pasang Filter Gabor
    fimg = cv2.filter2D(grayscale,-1, kernel)
    ax1.imshow(img)
    canvas1.draw()
    ax2.imshow(grayscale, cmap="gray")
    canvas2.draw()
    ax3.imshow(fimg, cmap="gray")
    canvas3.draw()
    

def ekstrak_ciri():
    global hasil_klasifikasi
    book1 = xw.Workbook('datatraining.xlsx')
    sheet1 = book1.add_worksheet()
    sheet1.write(0,0,'Keterangan')

    book2 = xw.Workbook('datatesting.xlsx')
    sheet2 = book2.add_worksheet()
    sheet2.write(0,0,'Keterangan')


    kolom = 1
    #kolom feature glcm
    glcm_feature = ['correlation','homogeneity','dissimilarity', 'contrast', 'energy']
    sudut = ['0','45','90','135']
    for i in glcm_feature :
        for j in sudut :
            sheet1.write (0,kolom,i+' '+j)
            sheet2.write (0,kolom,i+' '+j)
            
            kolom+=1

    #baris citra
    data_training = os.listdir(filedataset+"\datatraining")
    data_testing = os.listdir(filedataset+"\datatesting")

    row = 1
    kernel = cv2.getGaborKernel((8,8), sigma=3,theta=25, lambd=7.22, gamma=0.5, psi=0)
    for i in data_training :
            column = 0
            file_name = 'dataset/datatraining/'+i
            # print(file_name)
            sheet1.write(row,column,i)
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
                sheet1.write (row,column,item)
                column+=1

            row+=1
    book1.close()

    row = 1
    for i in data_testing :
        column = 0
        file_name = 'dataset/datatesting/'+i
        # print(file_name)
        sheet2.write(row,column,i)
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
            sheet2.write (row,column,item)
            column+=1
        row+=1
    book2.close()

    dTraining = pd.read_excel ('datatraining.xlsx', sheet_name='Sheet1')
    dTesting  = pd.read_excel ('datatesting.xlsx', sheet_name='Sheet1')
    enc = LabelEncoder()

    x = len(dTraining["Keterangan"])
    y = len(dTesting["Keterangan"])

    for i in range(x):
        if (dTraining['Keterangan'][i]).startswith("negatif"):
                dTraining['Keterangan'][i] = "negatif"
        if (dTraining['Keterangan'][i]).startswith("positif"):
                dTraining['Keterangan'][i] = "positif"

    for i in range(y):
        if (dTesting['Keterangan'][i]).startswith("negatif"):
                 dTesting['Keterangan'][i] = "negatif"
        if (dTesting['Keterangan'][i]).startswith("positif"):
                dTesting['Keterangan'][i] = "positif"

    dTraining['Keterangan'] = enc.fit_transform (dTraining['Keterangan'].values)
    dTesting['Keterangan'] = enc.fit_transform (dTesting['Keterangan'].values)
    xtrain= dTraining.drop(columns='Keterangan')
    xtest = dTesting.drop(columns='Keterangan')
    ytrain = dTraining['Keterangan']
    ytest = dTesting['Keterangan']

    model = SVC(kernel='linear')

    #Training data
    model.fit(xtrain,ytrain)
    print(accuracy_score(ytest, model.predict(xtest)))
    
    img = cv2.imread(fileImage)
    imgr = cv2.resize (img, (400,400))
    #grayscale gambar
    grayscale= cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
    #Pasang Filter Gabor
    fimg = cv2.filter2D(grayscale,-1, kernel)

    distances = [5]
    angles = [0,np.pi/4,np.pi/2,3*np.pi/4]
    levels = 256
    symetric = True
    normed = True

    data_baru = xw.Workbook('test.xlsx')
    tambah = data_baru.add_worksheet()

    colom = 0
    #kolom feature glcm
    glcm_feature = ['correlation','homogeneity','dissimilarity', 'contrast', 'energy']
    sudut = ['0','45','90','135']
    for i in glcm_feature :
        for j in sudut :
            tambah.write (0,colom,i+' '+j)
            colom+=1

    kolom = 0
    glcm = graycomatrix (fimg, distances, angles, levels, symetric, normed)
    glcm_props = [propery for name in glcm_feature for propery in graycoprops(glcm,name)[0]]
    for item in glcm_props :
        tambah.write (1,kolom,item)
        kolom+=1
    
    data_baru.close()
    hasil_ekstrak = pd.read_excel('test.xlsx',sheet_name='Sheet1')
    hasil_klasifikasi = model.predict(hasil_ekstrak)
    

def openDataset():
    global filedataset
    filedataset = filedialog.askdirectory()

def HasilKlasifikasi() :

    if hasil_klasifikasi[0] == 0:
        Ouput = tk.Label(frame6, font=('Cambria Bold',12) ,text="Negatif", background='white', width= 23).place(x=562,y=100)
        Keterangan = tk.Label(frame6, font=('Cambria',10), fg="white",text="Tidak terinfeksi DBD", background='grey', width= 23).place(x=585,y=130)
    else:
        Ouput = tk.Label(frame6, font=('Cambria Bold',12) ,text="Positif", background='white', width= 22).place(x=562,y=100)
        Keterangan = tk.Label(frame6, font=('Cambria',10), fg="white",text="Terinfeksi DBD", background='grey', width= 23).place(x=585,y=130)




#
label = tk.Label (window, text="KLASIFIKASI SEL DARAH DEMAM BERDARAH DENGUE SVM", font=("Cambria Bold",12),fg="white", background="#778899")
label.place( x=380, y=20)
label = tk.Label (window, text="By.Aisyah", font=("Cambria Bold",9),fg="black", background="#B0C4DE")
label.place( x=550, y=580)

#frame dan button Datatraining dan testing
dataset=tk.Button (window, text="PILIH FOLDER DATASET", command= openDataset, width=26,height=10,highlightbackground="black").place(x=20, y=140)

#Frame gambar Normal
frame2 = tk.Frame (window,background='white',highlightbackground="black", highlightthickness="1")
frame2.place(x=220, y=70)
#Plot Fg1
fig1, ax1 = plt.subplots()
fig1.set_size_inches(w=3,h=3)
canvas1 = FigureCanvasTkAgg (fig1, master=frame2)
canvas1.get_tk_widget().pack()
ax1.set_title ("Normal")

#frame plot grayscale
frame3 = tk.Frame (window,background='white',highlightbackground="black", highlightthickness="1")
frame3.place(x=525, y=70)
#Plot Fg2
fig2, ax2 = plt.subplots()
fig2.set_size_inches(w=3,h=3)
canvas2 = FigureCanvasTkAgg (fig2, master=frame3)
canvas2.get_tk_widget().pack()
ax2.set_title ("Grayscale")

#frame plot Gabor
frame4 = tk.Frame (window, background='white',highlightbackground="black", highlightthickness="1",highlightcolor="black")
frame4.place(x=830, y=70)
#plot Fg3
fig3, ax3 = plt.subplots()
fig3.set_size_inches(w=3,h=3)
canvas3 = FigureCanvasTkAgg (fig3, master=frame4)
canvas3.get_tk_widget().pack()
ax3.set_title ("Gabor Filter")


#Frame Nama
frame5 = tk.Frame (window, width=193, height=100, highlightbackground="black", highlightthickness="1",background='white')
#frame5.place(x=20, y=430)
nama = tk.Label(frame5, font=('Cambria Bold',11) ,text="Aisyah Operadiva Arna", background="White").place(x=18,y=20)

#Frame Bawahhh
frame6 = tk.Frame (window, width=911, height=200, highlightbackground="black", highlightthickness="1",background='white')
frame6.place(x=221, y=380)

#Frame Hasil
frame_hasil = tk.Frame (frame6, width=300, height=150, background='grey').place(x = 525, y=20 )
tombol_klasifikasi = tk.Button(frame6, text="KLASIFIKASI SEL DARAH", command= HasilKlasifikasi).place(x=600, y=45)
Ouput = tk.Label(frame6, font=('Cambria Bold',11) ,text="", background='white', width= 22).place(x=562,y=100)
# Keterangan = tk.Label(frame6, font=('Cambria Bold',11) ,text="", background='grey', width= 22).place(x=562,y=140)

#Frame Open dan Esktrak
Frame_opes = tk.Frame (frame6, height=100, width=328, background="grey").place(x=90,y=45)
open_image = tk.Button (frame6, text="OPEN IMAGE", command = openImage, height=4,width=20).place(x=100,y=60)
ekstrak_citra= tk.Button (frame6, text="EKSTRAK CIRI", command= ekstrak_ciri, height=4,width=20).place (x=260, y=60)



window.mainloop()