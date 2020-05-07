from tkinter import *
import tkinter.filedialog as filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

root = Tk()
root.title("Anomaly Detection Toolkit")
root.geometry("800x650")
root.resizable(0,0)
root.config(bg="#486f96")
########## dataset section #############

########### frame for dataset part
frame = LabelFrame(root, text = "Dataset section", padx=50, pady=50, bd=5, bg="#34495e")
frame.pack(padx=10, pady=10, fill=BOTH)

dsLabel = Label(frame,text="Dataset : ", bg="#34495e", bd=5)
dsLabel.config(font=("Times New Roman", 12))
dsLabel.grid(row=0, column=0)

##variable to hold browse path
browseStatus = StringVar(frame)
browseStatus.set("Path")

## function to be called when browse btn is clicked
def browsefunc():
    filename = filedialog.askopenfilename()
    browseStatus.set(filename)
    dsStatus.set("NA")
    pathLabel.config(text=filename.split('/')[-1])
def resetfunc():
    browseStatus.set("Path")
    pathLabel.config(text="Path")

browsebutton = Button(frame, text="Browse", command=browsefunc, bg="#34495e", bd=5, width=40)
browsebutton.grid(row=1, column=0, columnspan=3)

resetBtn = Button(frame, text="Reset", command=resetfunc, bg="#34495e", bd=5, width=10)
resetBtn.grid(row=1, column=4)

pathLabel = Label(frame, text="Path" ,bd=1, anchor=E, bg="#34495e") ## status of the dataset selected
pathLabel.grid(row=1, column=3, sticky=W+E)


# dsInput = Entry(root, borderwidth=5, width=50)
# dsInput.grid(row=0, column=1)

####### tkinter variable for radiobutton
dsStatus = StringVar(frame)
dsStatus.set("NA") ## setting initial value

RadioButtonOptions = [
    ("page_blocks", "page_blocks"),
    ("Lymphography", "Lymphography"),
    ("Cancer", "Cancer"),
    ("postoperative", "postoperative"),
]
dsStatusLabel = Label(frame, text="Dataset Seletected is : " + dsStatus.get(), bd=1, relief=SUNKEN, anchor=E, bg="#34495e") ## status of the dataset selected
dsStatusLabel.grid(row=2, column=0, columnspan=5, sticky=W+E, pady=10)

def dsSelected(*args):
    global dsStatusLabel
    dsStatusLabel.grid_forget()
    dsStatusLabel = Label(frame, text="Dataset Seletected is : " + dsStatus.get(), bd=1, relief=SUNKEN, anchor=E, bg="#34495e") ## status of the dataset selected
    dsStatusLabel.grid(row=2, column=0, columnspan=5, sticky=W+E, pady=10)
    


for idx, (text, value) in enumerate(RadioButtonOptions):
    rbtn = Radiobutton(frame, text=text, variable=dsStatus, value=value, bg="#34495e")
    rbtn.grid(row=0, column=idx+1, pady=10, padx=25)
    rbtn.config(font=("Times New Roman", 12))
    
dsStatus.trace('w', dsSelected)
# Radiobutton(frame, text="dataset 1", variable=r, value="ds0", command=lambda: dsSelected(r.get())).grid(row=0, column=1, pady=10)






####### algo section ##########
frame2 = LabelFrame(root, text="Algorithm section", padx=50, pady=50, bd=5, bg="#34495e")
frame2.pack(padx=10, pady=10, fill=BOTH)


algoLabel = Label(frame2, text="Algorithm : ", bg="#34495e")
algoLabel.grid(row=0, column=0)
algoLabel.config(font=("Times New Roman", 12))
# algoInput = Entry(frame2, borderwidth=5, width=50)
# algoInput.grid(row=0, column=1)
algoStatus = StringVar(frame2)
algoStatus.set("No algo selected")

algoChoices = {"ELM", "One-class SVM", "DBSCAN", "LOF", "KNN", "Mean KNN", "Median KNN", "PCA", "IForest", "Feature Bagging"}

algoDropdown = OptionMenu(frame2, algoStatus, *algoChoices)
algoDropdown.config(bg="#34495e", bd=5, width=50)
algoDropdown["menu"].config(bg="#34495e")
algoDropdown.grid(row=0, column=1, pady=10)


##########result stuff
frame3 = LabelFrame(root, text="Result section", padx=50, pady=50, bd=5,bg="#34495e",)
frame3.pack(padx=10, pady=10, fill=BOTH)
result = DoubleVar()
result.set(0.0)
print(result.get())
resultLabel = Label(frame3, text="Result : ", bg="#34495e")
resultLabel.grid(row=0, column=0)
resultLabel.config(font=("Times New Roman", 12))
accuracy = Label(frame3, text=result.get(), bg="#34495e")
accuracy.grid(row=0, column=1)
accuracy.config(font=("Times New Roman", 12))

##########################################################################################################################

# preprocessing for user uploaded dataset
def user_dataset(path):
    df = pd.read_csv(path)
    df_norm = df[df.Label == 0]
    df_anom = df[df.Label == 1]
    ds_norm = df_norm.values
    ds_anom = df_anom.values

    X_train = ds_norm[:int(np.ceil(ds_norm.shape[0] - 0.3*ds_norm.shape[0])), :-1]
    Y_train = ds_norm[:int(np.ceil(ds_norm.shape[0] - 0.3*ds_norm.shape[0])), -1]

    l = ds_norm.shape[0] - X_train.shape[0] 
    no_of_test_samples = l + ds_anom.shape[0]
    no_of_features = X_train.shape[1]

    X_test = np.zeros((no_of_test_samples, no_of_features))
    Y_test = np.zeros((no_of_test_samples,))

    X_test[:l, :] = ds_norm[int(np.ceil(ds_norm.shape[0] - 0.3*ds_norm.shape[0])):, :-1]
    X_test[l:, :] = ds_anom[:,:-1]
    print(X_test.shape)
    Y_test[:l,] = ds_norm[int(np.ceil(ds_norm.shape[0] - 0.3*ds_norm.shape[0])):, -1]
    Y_test[l:,] = ds_anom[:, -1]
    return X_train, Y_train, X_test, Y_test, ds_anom, ds_norm



###############################################################################

from algos import Lof, DBSCAN, OCSVM, elm, knn, mean_knn, median_knn, pca, iforest, feature_bagging
from preprocessing import pageblocks_dataset, cancer_dataset, lymphography_dataset, postoperative_dataset
# submit stuff
def myClick():
    print(dsStatus.get())
    print(algoStatus.get())

    if dsStatus.get() == "page_blocks" and browseStatus.get()=='Path':
        
        X_train, Y_train, X_test, Y_test, ds_anom, ds_norm = pageblocks_dataset()
        
        if algoStatus.get() =="LOF":
            acc = Lof(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="DBSCAN":
            acc = DBSCAN(X_train, X_test, Y_train, Y_test, dsStatus)
        if algoStatus.get() =="One-class SVM":
            acc = OCSVM(X_train, X_test,Y_test)
        if algoStatus.get() == "ELM":
            acc = elm(X_train, X_test, Y_train, Y_test, ds_anom, ds_norm)
        if algoStatus.get() =="KNN":
            acc = knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Mean KNN":
            acc = mean_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Median KNN":
            acc = median_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="PCA":
            acc = pca(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="IForest":
            acc = iforest(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Feature Bagging":
            acc = feature_bagging(X_train, X_test, Y_train, Y_test)

    elif dsStatus.get() == "Lymphography" and browseStatus.get()=='Path':

        X_train, Y_train, X_test, Y_test, ds_anom, ds_norm = lymphography_dataset()

        if algoStatus.get() =="LOF":
            acc = Lof(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="DBSCAN":
            acc = DBSCAN(X_train, X_test, Y_train, Y_test, dsStatus)
        if algoStatus.get() =="One-class SVM":
            acc = OCSVM(X_train, X_test,Y_test)
        if algoStatus.get() == "ELM":
            acc = elm(X_train, X_test, Y_train, Y_test, ds_anom, ds_norm)
        if algoStatus.get() =="KNN":
            acc = knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Mean KNN":
            acc = mean_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Median KNN":
            acc = median_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="PCA":
            acc = pca(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="IForest":
            acc = iforest(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Feature Bagging":
            acc = feature_bagging(X_train, X_test, Y_train, Y_test)

    elif dsStatus.get() == "Cancer" and browseStatus.get()=='Path':

        X_train, Y_train, X_test, Y_test, ds_anom, ds_norm = cancer_dataset()
        if algoStatus.get() =="LOF":
            acc = Lof(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="DBSCAN":
            acc = DBSCAN(X_train, X_test, Y_train, Y_test, dsStatus)
        if algoStatus.get() =="One-class SVM":
            acc = OCSVM(X_train, X_test,Y_test)
        if algoStatus.get() == "ELM":
            acc = elm(X_train, X_test, Y_train, Y_test, ds_anom, ds_norm)
        if algoStatus.get() =="KNN":
            acc = knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Mean KNN":
            acc = mean_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Median KNN":
            acc = median_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="PCA":
            acc = pca(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="IForest":
            acc = iforest(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Feature Bagging":
            acc = feature_bagging(X_train, X_test, Y_train, Y_test)

    elif dsStatus.get() == "postoperative" and browseStatus.get()=='Path':

        X_train, Y_train, X_test, Y_test, ds_anom, ds_norm = postoperative_dataset()

        if algoStatus.get() =="LOF":
            acc = Lof(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="DBSCAN":
            acc = DBSCAN(X_train, X_test, Y_train, Y_test, dsStatus)
        if algoStatus.get() =="One-class SVM":
            acc = OCSVM(X_train, X_test,Y_test)
        if algoStatus.get() == "ELM":
            acc = elm(X_train, X_test, Y_train, Y_test, ds_anom, ds_norm)
        if algoStatus.get() =="KNN":
            acc = knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Mean KNN":
            acc = mean_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Median KNN":
            acc = median_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="PCA":
            acc = pca(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="IForest":
            acc = iforest(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Feature Bagging":
            acc = feature_bagging(X_train, X_test, Y_train, Y_test)
    else:
        # if browseStatus.get() =="Path":
        #     accuracy.config(text="No dataset selected")
        print(browseStatus.get())
        dsStatus.set('NA')
        X_train, Y_train, X_test, Y_test, ds_anom, ds_norm = user_dataset(browseStatus.get())
        if algoStatus.get() =="LOF":
            acc = Lof(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="DBSCAN":
            acc = DBSCAN(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="One-class SVM":
            acc = OCSVM(X_train, X_test,Y_test)
        if algoStatus.get() == "ELM":
            acc = elm(X_train, X_test, Y_train, Y_test, ds_anom, ds_norm)
        if algoStatus.get() =="KNN":
            acc = knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Mean KNN":
            acc = mean_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Median KNN":
            acc = median_knn(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="PCA":
            acc = pca(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="IForest":
            acc = iforest(X_train, X_test, Y_train, Y_test)
        if algoStatus.get() =="Feature Bagging":
            acc = feature_bagging(X_train, X_test, Y_train, Y_test)

    accuracy.config(text=acc)
    accuracy.config(font=("Times New Roman", 12))

MyButton1 = Button(frame2, text="Submit", width=10, command=myClick, bd=5, bg="#34495e")
MyButton1.grid(row=1, column=1)



root.mainloop()

