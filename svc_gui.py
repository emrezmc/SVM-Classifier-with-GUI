import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


gui = Tk()

gui.title('SVM Classifier GUI')

gui.geometry('600x500')


def data():
    global filename
    filename = askopenfilename(initialdir='C:\\Users\\emrez\\Desktop\\MLGUI',title = "Select file")
    e1.delete(0, END)
    e1.insert(0, filename)
    #e1.config(text=filename)
    #print(str(filename))
    box1.delete(0,END)
    global df

    df = pd.read_csv(filename)

    global col
    col = list(df.head(0))
    #print(col)

    for i in range(len(col)):
        box1.insert(i, col[i])

def X_values():
    X_values.val = [box1.get(idx) for idx in box1.curselection()]
    for i in range(len(list(X_values.val))):
        box2.insert(i, X_values.val[i])
        box1.selection_clear(i, END)

    print("Predictors are:"+ str(X_values.val))

def y_values():
    y_values.val= [box1.get(idx) for idx in box1.curselection()]
    for i in range(len(list(y_values.val))):
        box3.insert(i, y_values.val[i])

    print("Targets are:"+ str(y_values.val))

def clearBox2():
    del X_values.val
    box2.delete(0,END)
    
def clearBox3():
    del y_values.val
    box3.delete(0,END)
def sol():
    testSize=float(splitEntry.get())
    X=df[df.columns&X_values.val]
#    X=file1[file1.columns.intersection(X_values.val)]
    y=df[df.columns&y_values.val]
#    y = y.reshape((-1,))
    if kernelVar.get()=="poly":
        deg=int(degEntry.get())
#        gam=float(gammaEntry.get())
    else:
        gam=float(gammaEntry.get())
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=testSize,random_state=65)
    
    if kernelVar.get()=="poly":
        print("uses poly")
        svc=SVC(kernel=kernelVar.get(),C=float(cEntry.get()),degree=deg)
    else:
        svc=SVC(kernel=kernelVar.get(),C=float(cEntry.get()),gamma=gam)
        print("not-uses poly")
    svc.fit(X_train,y_train)
    
    pred=svc.predict(X_test)
    
#    print(metrics.accuracy_score(y_test,pred))
#    print("Precision:",metrics.precision_score(y_test, pred))    
#    print("Recall:",metrics.recall_score(y_test, pred))    
#    print(classification_report(y_test, pred, target_names=['Not diseased','Diseased']))
   
    pop=Toplevel()
    pop.title("Results")
    lbl4=Label(pop,text="Predictors are:"+ str(X_values.val))
    lbl4.pack()
    lbl5=Label(pop,text="Targets are:"+ str(y_values.val))
    lbl5.pack()
    lbl=Label(pop,text="Accuracy: "+ str(metrics.accuracy_score(y_test,pred)))
    lbl.pack()
    lbl1=Label(pop,text="Precision: "+ str(metrics.precision_score(y_test, pred)))
    lbl1.pack()
    lbl2=Label(pop,text="Recall: "+ str(metrics.recall_score(y_test, pred)))
    lbl2.pack()
#    lbl3=Label(pop,text=classification_report(y_test, pred, target_names=['Not diseased','Diseased']))
#    lbl3.pack()
    lbl6=Label(pop,text="F1 score: "+ str(metrics.f1_score(y_test,pred)))
    lbl6.pack()
    plot_confusion_matrix(svc, X_test, y_test)

LabelFrame1=LabelFrame(gui,text="Loading data")
LabelFrame1.pack(fill="both",ipadx=5,ipady=5)
l1=Label(LabelFrame1, text='Select Data File')
l1.grid(row=0, column=0)
e1=Entry(LabelFrame1,text='')
e1.grid(row=0, column=1)

Button(LabelFrame1,text='open', command=data).grid(row=0, column=2)

box1 = Listbox(LabelFrame1,selectmode='multiple')
box1.grid(row=10, column=0,padx=35)

box2 = Listbox(LabelFrame1)
box2.grid(row=10, column=1,padx=35)

Button(LabelFrame1, text='Select Predictors', command=X_values).grid(row=12,column=1)
Button(LabelFrame1, text='Clear Predictors',command=clearBox2).grid(row=13,column=1)

box3 = Listbox(LabelFrame1)
box3.grid(row=10, column=2,padx=35) 

Button(LabelFrame1, text='Select Targets', command=y_values).grid(row=12,column=2)
Button(LabelFrame1, text='Clear Targets',command=clearBox3).grid(row=13,column=2)


paramFrame=LabelFrame(gui,text="Parameters")
paramFrame.pack(fill="both")

cLabel=Label(paramFrame, text='Tune C parameter:')
cLabel.grid(row=0,column=0,padx=10,pady=10)

cEntry=Entry(paramFrame,text=' ')
cEntry.grid(row=0, column=1,pady=10,padx=10)


kerneLabel=Label(paramFrame, text='Kernel Func:')
kerneLabel.grid(row=0,column=2,pady=10,padx=10)

OPTIONS = [
"Select model",
"linear",
"poly",
"rbf"
]

kernelVar = StringVar()
kernelVar.set(OPTIONS[0])

kernelFunc=OptionMenu(paramFrame, kernelVar, *OPTIONS)
kernelFunc.grid(row=0,column=3,pady=10)

degLabel=Label(paramFrame, text='  Degree(For poly.kernel):')
degLabel.grid(row=1,column=0,padx=10,pady=10)

degEntry=Entry(paramFrame)
degEntry.grid(row=1, column=1,padx=10,pady=10)

gammaLabel=Label(paramFrame, text='  Gamma(For radial kernel):')
gammaLabel.grid(row=1,column=2,pady=10,padx=10)

gammaEntry=Entry(paramFrame)
gammaEntry.grid(row=1, column=3,pady=10)

splitLabel=Label(paramFrame, text='Test size:')
splitLabel.grid(row=2,column=0,pady=10)

splitEntry=Entry(paramFrame)
splitEntry.grid(row=2, column=1,pady=10)

Button(gui, text='Solution', command=sol).place(x=270,y=450)


gui.mainloop()