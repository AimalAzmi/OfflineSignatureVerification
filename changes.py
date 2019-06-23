lr = 0.001
H = []

for i in range(1):
    print(i)
    # initialize the optimizer and compile the model
    print("[INFO] compiling model...")
    opt = Adam(lr=lr)
    lr = lr/10
    model.compile(optimizer=opt, loss=losses, loss_weights=lossWeights, metrics=["accuracy"])                
    # train the network to perform multi-output classification
    K = model.fit(X_Gen_Forg_train,
            {"categoryUser": y_Gen_Forg_train, "categoryGenOrForg": Res_Gen_Forg_train},
            validation_data=(X_Gen_Forg_test, {"categoryUser": y_Forg_Gen_test, "categoryGenOrForg": Res_Gen_Forg_test}),
            epochs=20, batch_size=BS, verbose=1)
    H.append(K)
 
# save the model to disk
print("[INFO] serializing network...")
model.save("my_model.h5")



trained_model = np.array(H)


lossNames = ["loss", "categoryUser_loss", "categoryGenOrForg_loss"]
accuracyNames = ["categoryUser_acc", "categoryGenOrForg_acc"]

losses_all_iterations = []
accuracy_all_iterations = []

val_losses_all_iterations = []
val_accuracy_all_iterations = []

for loss in lossNames:
    losses_all_iterations.append([])
    val_losses_all_iterations.append([])
    
for accuracy in accuracyNames:
    accuracy_all_iterations.append([])
    val_accuracy_all_iterations.append([])
        
for i, loss in enumerate(lossNames):
    for single_model in trained_model:
        losses_all_iterations[i].append(single_model.history[lossNames[i]])
        val_losses_all_iterations[i].append(single_model.history['val_' + lossNames[i]])
        
for i, accuracy in enumerate(accuracyNames):
    for single_model in trained_model:
        accuracy_all_iterations[i].append(single_model.history[accuracyNames[i]])  
        val_accuracy_all_iterations[i].append(single_model.history['val_' + accuracyNames[i]])
        
        
losses_all_iterations = np.array(losses_all_iterations)
accuracy_all_iterations = np.array(accuracy_all_iterations)

val_losses_all_iterations = np.array(val_losses_all_iterations)
val_accuracy_all_iterations = np.array(val_accuracy_all_iterations)



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
    ax[i].plot(losses_all_iterations[i].flatten(), label=l)
    ax[i].plot(val_losses_all_iterations[i].flatten(),
        label="val_" + l)
    ax[i].legend()
 
plt.tight_layout()
plt.savefig("losses_main_1.png")
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
    ax[i].plot(accuracy_all_iterations[i].flatten(), label=l)
    ax[i].plot(val_accuracy_all_iterations[i].flatten(),
        label="val_" + l)
    ax[i].legend()
 
# save the accuracies figure
plt.tight_layout()
plt.savefig("accuracy_main_1.png")
plt.close()