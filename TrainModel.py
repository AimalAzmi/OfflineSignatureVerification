# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:24:25 2018

@author: Manoochehr
"""
 
# import the necessary packages
from keras.optimizers import Adam, SGD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from sklearn.svm import SVC

EPOCHS = 20
BS = 64
IMAGE_HEIGHT = 150
IMAGE_WIDTH = 220

pathGen = "DatasetSig1/original/*/*.png"
pathForg = "DatasetSig1/forgeries/*/*.png"

from DataPreparing import DataPreparing
X_Gen , y_Gen =  DataPreparing(pathGen)
X_Forg , y_Forg =  DataPreparing(pathForg)

X_GenForg = np.concatenate((X_Gen, X_Forg), axis=0) # All data
y_GenForg = np.concatenate((y_Gen, y_Forg), axis=0) # category users
Result_GenForg = np.concatenate((np.zeros(len(y_Gen)), np.ones(len(y_Forg))), axis=0) # category GenOrForg

#X_GenForg = np.array(X_GenForg, dtype="float") / 255.0 # Sclae Input to [0,1]

# binarizing
print("[INFO] binarizing labels...")
y_GenForgyLBin = LabelBinarizer()
Result_GenForgLBin = LabelBinarizer()
y_GenForg = y_GenForgyLBin.fit_transform(y_GenForg)
Result_GenForg = Result_GenForgLBin.fit_transform(Result_GenForg)

# Genuine Dataset
X_Gen_Forg_train, X_Gen_Forg_test, y_Gen_Forg_train, y_Forg_Gen_test, Res_Gen_Forg_train, Res_Gen_Forg_test = train_test_split(
                X_GenForg, y_GenForg, Result_GenForg, test_size=0.2, random_state=23, stratify=y_GenForg)

from ModelBuilding import build
# initialize our multi-output network
model = build(IMAGE_HEIGHT, IMAGE_WIDTH,
	numUsers=len(y_Gen_Forg_train[0]),
	numForgOrGen=1,
	finalAct1 = "softmax",finalAct2 = "sigmoid")
 
# define two dictionaries: one that specifies the loss method for
# each output of the network along with a second dictionary that
# specifies the weight per loss
losses = {
	"categoryUser": "categorical_crossentropy",
	"categoryGenOrForg": "binary_crossentropy",
}    
LamdaWeight = 0.7    
lossWeights = {"categoryUser": 1, "categoryGenOrForg": 1}     
lr = 0.001

for i in range(1):
    print(i)
    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    opt = Adam(lr=lr)
    #opt = SGD(lr=lr, momentum=0.9, decay=1e-4, nesterov=False)
    lr = lr/10
    model.compile(optimizer = opt, loss = losses, loss_weights = lossWeights, metrics=["accuracy"])                
    # train the network to perform multi-output classification
    H = model.fit(X_Gen_Forg_train,
                  {"categoryUser": y_Gen_Forg_train, "categoryGenOrForg": Res_Gen_Forg_train},
                  validation_data=(X_Gen_Forg_test, {"categoryUser": y_Forg_Gen_test, "categoryGenOrForg": Res_Gen_Forg_test}),
                  epochs=EPOCHS, batch_size=BS, verbose=1)
    
# 10-5 no learning, 10-6 no leARNING, 10-7 not learning
# LamdaWeight = 0.1 75% ResGenForg, 92% User accuracy
# LamdaWeight = 0.01 72% ResGenForg, 90% User accuracy
# LamdaWeight = 0.3 78% ResGenForg, 90% User accuracy
# LamdaWeight = 0.5 76% ResGenForg, 90% User accuracy
# LamdaWeight = 0.7 80% ResGenForg, 92% User accuracy
# Equal weight = 1: 82% ResGenForg, 89% User accuracy

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# plot the losses
lossNames = ["loss", "categoryUser_loss", "categoryGenOrForg_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))
 
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
	ax[i].legend()
 
plt.tight_layout()
plt.savefig("losses.png")
plt.close()

# plot figure for accuracy
accuracyNames = ["categoryUser_acc", "categoryGenOrForg_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(2, 1, figsize=(8, 8))

for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, EPOCHS), H.history[l], label=l)
	ax[i].plot(np.arange(0, EPOCHS), H.history["val_" + l], label="val_" + l)
	ax[i].legend()
 
# save the accuracies figure
plt.tight_layout()
plt.savefig("accs.png")
plt.close()

# save the model to disk
print("[INFO] serializing network...")
model.save("my_model.h5")

#from keras.models import load_model
#model = load_model('my_model.h5')

# ///////////////////////// get the nth Layer of CNN ///////////////////////////////////////////////
def getFeatures(clf , Data, n):
        get_nth_layer_output = K.function([clf.layers[0].input],
                                  [clf.layers[n].output])
        FeatureLayer = get_nth_layer_output([Data])[0]
        FeatureLayer = np.asarray(FeatureLayer)
        return FeatureLayer

# //////////////////////////////////// SVM Classifier //////////////////////////////////////////
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import accuracy_score
# Confusion Matrix Plot
import numpy as np
import matplotlib.pyplot as plt
from plot_confusion_matrix import plot_confusion_matrix

TrainFeature = getFeatures(model,X_Gen_Forg_train,-2)
TestFeature = getFeatures(model,X_Gen_Forg_test,-2)
# train SVM with features extracted from CNN
Rev_y_Gen_Forg_train = y_GenForgyLBin.inverse_transform(y_Gen_Forg_train)[:,np.newaxis]
Rev_y_Gen_Forg_test = y_GenForgyLBin.inverse_transform(y_Forg_Gen_test)[:,np.newaxis]

# User Classifing with SVM
clfUser_SVM = SVC(kernel = 'rbf', C = 1000, decision_function_shape='ovo',random_state=0, gamma='scale')
clfUser_SVM.fit(TrainFeature, Rev_y_Gen_Forg_train) 
predicted_User = clfUser_SVM.predict(TestFeature)
cnf_matrix_User = confusion_matrix(Rev_y_Gen_Forg_test,predicted_User)
print("accuracy of User", accuracy_score(Rev_y_Gen_Forg_test, predicted_User, normalize=True) )

classes = y_GenForgyLBin.classes_
plt.figure()
plot_confusion_matrix(cnf_matrix_User, classes=classes, normalize=False, title='confusion matrix')
plt.show()


#Gen or Forg Classyfing with SVM
clfGenForg_SVM = SVC(kernel = 'rbf', random_state=0, gamma='scale', C = 100000)
clfGenForg_SVM.fit(TrainFeature, Res_Gen_Forg_train) 
predicted_GenForg = clfGenForg_SVM.predict(TestFeature)
cnf_matrix_GenForg = confusion_matrix(Res_Gen_Forg_test,predicted_GenForg)
print("accuracy of GenForg", accuracy_score(Res_Gen_Forg_test, predicted_GenForg, normalize=True) )


classes=[0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix_GenForg, classes=classes, normalize=False, title='confusion matrix')
plt.show()













# ////////////////////////////// Test a Sample //////////////////////////////////////
import cv2
output = cv2.imread("DatasetSig1/original/user_1/original_1_19.png",cv2.IMREAD_GRAYSCALE)
output = 255 - output      
output = cv2.resize(output,(400, 320)) 
_,thresh1 = cv2.threshold(output,30,1,cv2.THRESH_BINARY) 
output = output * thresh1
output = cv2.fastNlMeansDenoising(output,None,10,7,21)
output1 = output[np.newaxis,:,:]

# draw the category label and color label on the image
(PredUser, PredGenOrForg) = model.predict(output1)

UserIdx = PredUser[0].argmax()
GenOrForgIdx = PredGenOrForg[0].argmax()
User = y_GenForgyLBin.classes_[UserIdx]
GenOrForg = Result_GenForgLBin.classes_[GenOrForgIdx]

UserText = "User: {} ({:.2f}%)".format(User,
	PredUser[0][UserIdx] * 100)
GenOrForgText = "GenOrForg: {} ({:.2f}%)".format(GenOrForg,
	PredGenOrForg[0][GenOrForgIdx] * 100)
 
# display the predictions to the terminal as well
print("[INFO] {}".format(UserText))
print("[INFO] {}".format(GenOrForgText))





# ////////////////////////// Testing model With other Dataset //////////////////////////////////////
pathGen = "DatasetSig1/original/*/*.png"
pathForg = "DatasetSig1/forgeries/*/*.png"

from DataPreparing import DataPreparing
X_Gen , y_Gen =  DataPreparing(pathGen)
X_Forg , y_Forg =  DataPreparing(pathForg)
Gen_Result = np.ones(len(y_Gen))
Forg_Result = np.zeros(len(y_Forg))

X_GenForg = np.concatenate((X_Gen, X_Forg), axis=0) # All data
y_GenForg = np.concatenate((y_Gen, y_Forg), axis=0) # category users
Result_GenForg = np.concatenate((Gen_Result, Forg_Result), axis=0) # category GenOrForg

X_GenForg = np.array(X_GenForg, dtype="float") / 255.0 # Sclae Input to [0,1]

# binarizing
print("[INFO] binarizing labels...")
y_GenForgyLBin = LabelBinarizer()
Result_GenForgLBin = LabelBinarizer()
y_GenForg = y_GenForgyLBin.fit_transform(y_GenForg)
Result_GenForg = Result_GenForgLBin.fit_transform(Result_GenForg)

# Genuine Dataset
X_Gen_train, X_Gen_test, y_Gen_train, y_Gen_test = train_test_split(X_Gen, y_Gen,test_size=0.2, random_state=23, stratify=y_Gen)
# Genuine and Forgeris Dataset
X_Gen_Forg_train, X_Gen_Forg_test, y_Gen_Forg_train, y_Forg_Gen_test, Res_Gen_Forg_train, Res_Gen_Forg_test = train_test_split(
                X_GenForg, y_GenForg, Result_GenForg, test_size=0.2, random_state=23, stratify=y_GenForg)


