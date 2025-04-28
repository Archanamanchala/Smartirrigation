from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from Environment import *
from Agent import *

main = tkinter.Tk()
main.title("IOT Based Smart Irrigation System using Reinforcement Learning")
main.geometry("1300x1200")

global filename
global dataset
global X, Y, X_train, X_test, y_train, y_test, scaler
global rewards, penalty, env, agent

def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,'Smart Irrigation Dataset loaded\n\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    text.insert(END,str(dataset))
    labels, count = np.unique(dataset['class'], return_counts = True)
    height = count
    bars = labels
    y_pos = np.arange(len(bars))
    plt.figure(figsize = (4, 3)) 
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Irrigation Condition Labels")
    plt.ylabel("Count")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def processDataset():
    global filename, dataset, scaler
    global X, Y
    text.delete('1.0', END)
    dataset = dataset.values
    X = dataset[:,1:7]
    Y = dataset[:,7]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    text.insert(END,"Dataset Processing & Normalization Completed\n\n")
    text.insert(END,"Normalized Dataset = "+str(X))

def splitDataset():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    text.insert(END,"Dataset Train & Test Splits for Reinforcement Learning Rewards & Penalty\n\n")
    text.insert(END,"80% Dataset used to train Reinforcement Learning : "+str(X_train.shape[0])+"\n") 
    text.insert(END,"20% Dataset used to test Reinforcement Learning : "+str(X_test.shape[0])+"\n")

def trainRL():
    text.delete('1.0', END)
    global X, Y, X_train, X_test, y_train, y_test
    global rewards, penalty, env, agent
    env = Environment()
    agent = Agent(env)
    rewards, penalty = agent.step(X_train, y_train, X_test, y_test)
    text.insert(END,"Reinforcement Learning Completed\n\n")
    text.insert(END,"Total Training Rewards   = "+str(rewards)+"\n")
    text.insert(END,"Total Training Penalties = "+str(penalty)+"\n")

def graph():
    global rewards, penalty
    height = [rewards, penalty]
    bars = ('Rewards', 'Penalty')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.xlabel("Type")
    plt.xlabel("Count")
    plt.title("Rewards & Penalty Graph")
    plt.show()

def predict():
    global env, agent
    global X_train, X_test, y_train, y_test, scaler
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    data = pd.read_csv(filename)
    data.fillna(0, inplace = True)
    temp = data.values
    testData = data.values
    testData = testData[:,1:7]
    print(testData)
    testData = scaler.transform(testData)
    for i in range(len(testData)):
        predict = agent.predictCondition(X_train, y_train, testData[i])
        text.insert(END,"Test Data : "+str(temp[i])+" Irrigation Status : "+predict+"\n\n")    
    
    
def close():
    main.destroy()

font = ('times', 16, 'bold')
title = Label(main, text='IOT Based Smart Irrigation System using Reinforcement Learning')
title.config(bg='chocolate', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
upload = Button(main, text="Upload Irrigation Dataset", command=uploadDataset)
upload.place(x=800,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='lawn green', fg='dodger blue')  
pathlabel.config(font=font1)           
pathlabel.place(x=800,y=150)

processButton = Button(main, text="Preprocess Dataset", command=processDataset)
processButton.place(x=800,y=200)
processButton.config(font=font1)

splitButton = Button(main, text="Dataset Train & Test Split", command=splitDataset)
splitButton.place(x=800,y=250)
splitButton.config(font=font1) 

trainButton = Button(main, text="Train Reinforcement Learning Algorithm", command=trainRL)
trainButton.place(x=800,y=300)
trainButton.config(font=font1)

graphButton = Button(main, text="Rewards & Penalty Graph", command=graph)
graphButton.place(x=800,y=350)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Irrigation Status", command=predict)
predictButton.place(x=800,y=400)
predictButton.config(font=font1)

exitButton = Button(main, text="Exit", command=close)
exitButton.place(x=800,y=450)
exitButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='light salmon')
main.mainloop()
