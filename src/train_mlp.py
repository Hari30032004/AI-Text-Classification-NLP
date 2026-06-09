# MLP classifier

import joblib                                # for loading and saving pre-processd data and models
from sklearn.neural_network import MLPClassifier     # MLP model for classification
from sklearn.metrics import accuracy_score, classification_report     # evaluation metrics
import matplotlib.pyplot as plt               # for plotting tuning results

# Loading pre-processed TF-IDF data

X_train, X_val, X_test, y_train, y_val, y_test = joblib.load("mlp_data.pkl")     # X labels are for TF-IDF feature matrics and y labels are categorised as 0 for false and 1 for True

# Building MLP model and hyperparameters to tune the model

hidden_layer_sizes = [(64,),(128,),(128,64)]   # different hidden layer structures.
learning_rates = [0.001,0.01]            # 2 learning rate values to compare
activations = ["relu","tanh"]            # standard activation functions for deep learning models to test

results = []                    # to store the tuning results
best_acc = 0                    # keeps track of best validation accuracy
best_model = None               # stores the best performing model
best_params = None              # stores the best hyperparameter combination.

# hyperparameter tuning loop.

for h in hidden_layer_sizes :        # loop through each hidden layer structure
    for lr in learning_rates :       # loops through each learning rate
        for act in activations :     # loops through each activation function

            # building model with current hyoeroarameters
            mlp = MLPClassifier(
                hidden_layer_sizes = h,       # setting hidden layer structure
                learning_rate_init = lr,           # setting learning rate
                learning_rate = "constant",
                activation = act,             # setting activation function
                max_iter = 20,                # number of training epochs
                random_state = 42
            )

            # Trains the model on trainig data
            mlp.fit(X_train, y_train)

            # Validation accuracy
            val_pred = mlp.predict(X_val)        # predicts labels for validation set
            val_acc = accuracy_score(y_val, val_pred)    # calculates validation accuracy
            print("Validation accuracy:", val_acc)      
               

            # storing the results
            results.append((h, lr, act, val_acc))   #saving hyperparameters and accuracy

            # updating the best model

            if val_acc > best_acc:            # checking if this model is the best fit or not
                best_acc = val_acc            # updating the best accuracy
                best_model = mlp              # storing best model
                best_params = (h, lr, act)    # storing best hyperparameters

# plotting tuning results

accuracy = [acc for (_,_,_, acc) in results]           # Extracts only the accuracy values from each (model, features, split, accuracy) tuple in results

plt.figure(figsize = (10,8))               # setting plot size
plt.scatter(range(len(accuracy)), accuracy, color = "blue",s=80) 
plt.plot(range(len(accuracy)), accuracy, color="blue")             
plt.xlabel("Model Index")
plt.ylabel("Validation Accuracy")          # y-axis labels
plt.title("MLP hyperparameter tuning results..")    # title
plt.grid(True, linestyle = "--",alpha = 0.6)                         # adjusting the layout
plt.show()                                 


# Evaluating best model on test data set

test_pred = best_model.predict(X_test)          # predicting labels for test set

print("Best hyperparameters:", best_params)       
print("Test accuracy:", accuracy_score(y_test, test_pred))

print(classification_report(y_test, test_pred, target_names=["FALSE","TRUE"]))    # classification report gives precision, recall and f1 score for each class

# Saving best model

joblib.dump(best_model, "mlp_model.pkl")
print("MLP model saved")
