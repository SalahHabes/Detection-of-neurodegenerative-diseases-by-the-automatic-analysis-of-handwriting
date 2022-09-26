from pyexpat import model
from tkinter import *
from PIL import Image;
from PIL import ImageTk;
from tkinter import filedialog as fd

import cv2 
import tensorflow
import pickle
from keras.models import load_model
import numpy as np
from sklearn.metrics import confusion_matrix

import seaborn
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import gc

app = Tk()
app.title('PD Detect')
app.geometry('800x450')
app.resizable(False,False)
app.iconbitmap(os.path.join(os.getcwd(), "icons\\pdd_ico.ico"))
folderico = Image.open(os.path.join(os.getcwd(), "icons\\folder_icon.ico")).resize((20, 20), Image.ANTIALIAS)
folderico = ImageTk.PhotoImage(folderico)
grayed_c = "#808B96"
active_c = "#000000"

imagesetpath_label = Label(app, text='PD Detection. Made by Habes Salah Eddine', font=('italic', 8), fg='#094cb8', anchor="w", width=70)
imagesetpath_label.grid(row=9, column=0, columnspan=7)
imagesetpath_label = Label(app, text='Detection of neurodegenerative diseases by the automatic analysis of handwriting', font=('italic', 7), fg='#094cb8', anchor="w", width=70, justify=LEFT)
imagesetpath_label.grid(row=10, column=1, columnspan=7)
imagesetpath_label = Label(app, text='Master II Systems and Multimedia (SYM) 2021/2022', font=('italic', 7), fg='#094cb8', anchor="w", width=70)
imagesetpath_label.grid(row=11, column=1, columnspan=7)
imagesetpath_label = Label(app, text='Larbi Tebessi University: Faculty of exact sciences and natural and life sciences', font=('italic', 7), fg='#094cb8', anchor="w", width=70)
imagesetpath_label.grid(row=12, column=1, columnspan=7)
imagesetpath_label = Label(app, text='Department of Math and computer science', font=('italic', 7), fg='#094cb8', anchor="w", width=70)
imagesetpath_label.grid(row=13, column=1, columnspan=7)

PDD_logo = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "icons\\pdd_logo.png")).resize((120, 100), Image.ANTIALIAS))
PDD_logoB = Label(image=PDD_logo)
PDD_logoB.image = PDD_logo
PDD_logoB.place(x=0, y=350)

def quit_me():
    app.quit()
    app.destroy()
 
app.protocol("WM_DELETE_WINDOW", quit_me)

showcase_size = 224
showcase_x = 500
showcase_y = 15
placeholder_img = ImageTk.PhotoImage(Image.open(os.path.join(os.getcwd(), "icons\\placeholder_img.png")).resize((showcase_size, showcase_size), Image.ANTIALIAS))
showcase_img = Label(image=placeholder_img)
showcase_img.image = placeholder_img
showcase_img.place(x=showcase_x, y=showcase_y)
predictionLabel = Label(app, anchor='w')

modelpath = StringVar(app)
modelpath_label = Label(app, text='Model Path: ', font=('bold', 12), anchor='w', width=15)
modelpath_label.grid(row=0, column=0)
modelpath_entry = Entry(app, textvariable=modelpath, width=40)
modelpath_entry.grid(row=0, column=1)
def getmodelpath():
    filepath = fd.askopenfilename(parent=app, title='Please select a model: ', filetypes=(("hdf5","*.hdf5"),("All files","*.*")))
    modelpath_entry.delete(0, END)
    modelpath_entry.insert(0, filepath)
Button(app, borderwidth=0, image=folderico, command=getmodelpath).grid(row=0, column=2)

modelType = StringVar()
modelType.set('svm')
modelType1 = Radiobutton(app, text='Svm activation ( > 0 )', font=('bold', 8), variable=modelType, value='svm', anchor='w', width=45)
modelType1.grid(row=1, column=0, columnspan=2)
modelType2 = Radiobutton(app, text='Sigmoid activation [park-net, VGG16, VGG19] ( == 1 )', font=('bold', 8), variable=modelType, value='sigmoid', anchor='w', width=45)
modelType2.grid(row=2, column=0, columnspan=2)

PredOption = ''
def enable_image():
    imagepath_entry.config(state= "normal")
    imagepath_label.config(fg=active_c)
    imagepath_button["state"] = "normal"
    global PredOption
    PredOption = 'image'

def disable_image():
    imagepath_entry.config(state= "disabled")
    imagepath_label.config(fg=grayed_c)
    imagepath_button["state"] = "disabled"

def enable_imageset():
    imagesetpath_entry.config(state= "normal")
    imagesetpath_label.config(fg=active_c)
    imagesetpath_button["state"] = "normal"
    datasplit_label.config(fg=active_c)
    datasplit_entry.config(state="normal")
    datasplit_scale.config(state="normal")
    global PredOption
    PredOption = 'imageset'

def disable_imageset():
    imagesetpath_entry.config(state= "disabled")
    imagesetpath_label.config(fg=grayed_c)
    imagesetpath_button["state"] = "disabled"
    datasplit_label.config(fg=grayed_c)
    datasplit_entry.config(state="disabled")
    datasplit_scale.config(state="disabled")
   
test_type = StringVar()
image_test = Radiobutton(app, text='Image', value='Image', font=('bold', 12), variable=test_type, command=lambda:[disable_imageset(), enable_image()])
image_test.grid(row=3, column=0, pady=(30, 10))
imageset_test = Radiobutton(app, text='Image Set', value='Image Set', font=('bold', 12), variable=test_type, command=lambda:[disable_image(), enable_imageset()])
imageset_test.grid(row=3, column=1, pady=(30, 10))

imagepath = StringVar(app)
imagepath_label = Label(app, text='Image Path: ', font=('bold', 12), fg=grayed_c, anchor='w', width=15)
imagepath_label.grid(row=4, column=0)
imagepath_entry = Entry(app,state= "disabled", textvariable=imagepath, width=40)
imagepath_entry.grid(row=4, column=1)
def getimagepath():
    filepath = fd.askopenfilename(parent=app, title='Please select an image: ', filetypes=(("jpg","*.jpg"),("jpeg","*.jpeg"),("png","*.png"),("All files","*.*")))
    imagepath_entry.delete(0, END)
    imagepath_entry.insert(0, filepath)
imagepath_button = Button(app, borderwidth=0, image=folderico, state= "disabled", command=getimagepath)
imagepath_button.grid(row=4, column=2)

imagesetpath = StringVar(app)
imagesetpath_label = Label(app, text='Image Set Path: ', font='bold 12', fg=grayed_c, anchor='w', width=15)
imagesetpath_label.grid(row=5, column=0, pady=(30, 0))
imagesetpath_entry = Entry(app,state= "disabled", textvariable=imagesetpath, width=40)
imagesetpath_entry.grid(row=5, column=1, pady=(30, 0))
def getimagesetpath():
    filepath = fd.askopenfilename(parent=app, title='Please select an image set (Serialized using pickel): ')
    imagesetpath_entry.delete(0, END)
    imagesetpath_entry.insert(0, filepath)
imagesetpath_button = Button(app, borderwidth=0, image=folderico, state= "disabled", command=getimagesetpath)
imagesetpath_button.grid(row=5, column=2, pady=(30, 0))

datasplit = IntVar(app, value=20)
datasplit_label = Label(app, text="Set datasplit: ", font="8", fg=grayed_c, anchor='w', width=15)
datasplit_entry = Entry(app,state= "disabled", textvariable=datasplit)
datasplit_scale = Scale(app, from_=1, to=100, variable=datasplit, orient=HORIZONTAL, showvalue=0, state= "disabled")
datasplit_label.grid(row=6, column=0)
datasplit_entry.grid(row=6, column=1)
datasplit_scale.grid(row=7, column=1)

def getimagepred(modelpath, imagepath, modelType):

    try:
        thumbnail = ImageTk.PhotoImage(Image.open(imagepath).resize((showcase_size, showcase_size), Image.ANTIALIAS))
        showcase_img.configure(image=thumbnail)
        showcase_img.image = thumbnail

        img = cv2.imread(imagepath)
        img = cv2.resize(img, (224,224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
    except:
        predictionLabel.configure(text="Image not found.", fg='red', font=(8), anchor="w", justify=LEFT)
        predictionLabel.place(x=showcase_x, y=showcase_size + 20)

    try:
        MODEL = load_model(modelpath)
    except:
        predictionLabel.configure(text="Model not found.", fg='red', font=(8), anchor="w", justify=LEFT)
        predictionLabel.place(x=showcase_x, y=showcase_size + 20)

    if modelType=='svm': # park-net
        prediction = (MODEL.predict(img_array) > 0)
    elif modelType=='sigmoid': # VGG
        prediction = (MODEL.predict(img_array) == 1)

    predictionLabel.place(x=int(showcase_x+(showcase_size/3)), y=showcase_size + 20)
    if prediction:
        predictionLabel.configure(text="Positive", fg='red')
    else:
        predictionLabel.configure(text="Negative", fg='green')
    predictionLabel.configure(font=('bald', 16))
    del img
    del img_array
    del MODEL
    gc.collect()

def getimagesetpred(modelpath, imagesetpath, datasplit, modelType):

  datasplit = datasplit/100

  try:
    MODEL = load_model(modelpath)
  except:
    predictionLabel.configure(text="Model not found.", fg='red', font=(8), anchor="w", justify=LEFT)
    predictionLabel.place(x=showcase_x, y=showcase_size + 20)
  
  try:
    X = pickle.load(open(os.path.join(os.path.dirname(imagesetpath), 'X'+os.path.basename(os.path.normpath(imagesetpath))[1:]), 'rb'))
    Y = pickle.load(open(os.path.join(os.path.dirname(imagesetpath), 'Y'+os.path.basename(os.path.normpath(imagesetpath))[1:]), 'rb'))
  except:
    predictionLabel.configure(text="Image Set not found.", fg='red', font=(8), anchor="w", justify=LEFT)
    predictionLabel.place(x=showcase_x, y=showcase_size + 20)
  
  X_val = X[round(X.shape[0]*(1-datasplit)):]
  Y_val = Y[round(Y.shape[0]*(1-datasplit)):]
  
  #get number of healthy and parkinsons in validation set
  N = 0
  P = 0
  for i in Y_val:
    if int(i) == 0:
      N += 1
    else:
      P += 1

  #get predictions
  if modelType=='svm': # park-net
    predictions = (MODEL.predict(X_val) > 0)
  elif modelType=='sigmoid': # VGG
    predictions = (MODEL.predict(X_val) == 1)
    
  #customise matrix
  cf_matrix = confusion_matrix(Y_val, predictions)
  group_names = ["True Neg","False Pos","False Neg","True Pos"]
  group_counts = ['{0:0.0f}'.format(value) for value in cf_matrix.flatten()]
  group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/[N, N, P, P]]
  labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
  labels = np.asarray(labels).reshape(2,2)
  seaborn.set(font_scale = 2)

  cnf_plot = seaborn.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
  cnf_plot.margins(x=0)
  fig = cnf_plot.get_figure()
  fig.savefig("tempcnf.png")
  fig.clf()
  thumbnail = ImageTk.PhotoImage(Image.open('tempcnf.png').resize((int(showcase_size+(showcase_size/4)), showcase_size), Image.ANTIALIAS))
  showcase_img.configure(image=thumbnail)
  showcase_img.image = thumbnail
  
  predictionLabel.place(x=showcase_x, y=showcase_size + 20)
  tn = cf_matrix.flatten()[0] # true positives
  fp = cf_matrix.flatten()[1] # false negatives
  fn = cf_matrix.flatten()[2] # false positives
  tp = cf_matrix.flatten()[3] # true negatives
  predictionLabel.configure(text="Model: "+str(os.path.basename(os.path.normpath(modelpath)))+
    "\nDataset: "+str(os.path.basename(os.path.normpath(imagesetpath)[2:]))+
    "\n"+str(int(datasplit*100))+"%"+" of the Dataset used"+
    "\nN° Healthy: "+str(N)+
    "\nN° PD Patients: "+str(P)+
    "\nAccurecy:  "+str("{:.2f}".format(((tp+tn)/(N+P))*100))+"%"+
    "\nError rate:  "+str("{:.2f}".format(((fp+fn)/(N+P))*100))+"%"+
    "\nSensitivity:  "+str("{:.2f}".format((tp/P)*100))+"%"+
    "\nSpecificity:  "+str("{:.2f}".format((tn/N)*100))+"%"+
    "\nBalanced Acc:  "+str("{:.2f}".format((((tp/P)+(tn/N))/2)*100))+"%", fg='#002b55', font=(8), anchor="w", justify=LEFT)
  
  del X
  del Y
  del X_val
  del Y_val
  del MODEL
  del cf_matrix
  del predictions
  del cnf_plot
  gc.collect()

def getprediction(modelpath, imagepath, imagesetpath, datasplit, modelType):
    if PredOption=='image':
        getimagepred(modelpath, imagepath, modelType)
    elif PredOption=='imageset':
        getimagesetpred(modelpath, imagesetpath, datasplit, modelType)
    else:
        predictionLabel.configure(text="Pls select a prediction option.", fg='red', font=(8), anchor="w", justify=LEFT)
        predictionLabel.place(x=showcase_x, y=showcase_size + 20)

Button(app, text="Proceed",
 command=lambda: getprediction(modelpath.get(), imagepath.get(), imagesetpath.get(), datasplit.get(), modelType.get()), 
 width=30, 
 bg="#0048a2", 
 activebackground=grayed_c,
 fg="white", 
 font=('bold'), 
 borderwidth=0).grid(row=8, column=1, columnspan=2, pady=(10, 30))

app.mainloop()