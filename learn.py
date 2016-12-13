
"""
Created on Thu Nov 17 23:53:32 2016

@author: Umair Ahmed
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support
from sklearn.svm import SVC



#Method for computing and plotting the confusion matrix, taken and modified from sklearn's official documentation website
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap,aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(round(cm[i, j],2))+'%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#Method for labelling the bar chart bars, taken from Matplotlib documentation
def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%f' % float(height),
                ha='center', va='bottom')
    
        
        
    
    '''
     Main code begins from here
    '''
dataset = np.loadtxt('Features/Features.csv', delimiter=",",skiprows=1)


#Note on Numpy array splicing
#our_array[a:b,c:d]  a and b = rows index of array && c and d = column index of array 


X = dataset[:, 1:]                      #array of 300x56 dimensons i.e. 300 feature vectors each having 56 dimensions
y = dataset[:, 0]      
class_names = ['user-1','user-2','user-3','user-4','user-5','user-6','user-7','user-8','user-9','user-10','user-11','user-12','user-13','user-14','user-15','user-16','user-17','user-18']



classifiers = {'Baseline Classifier':DummyClassifier(),'Random Forest Classifier':RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1), 'KNN':KNeighborsClassifier(10),'Decision Tree':DecisionTreeClassifier(max_depth=6),'Linear SVM':SVC(kernel="linear", C=0.025),'RBF SVM':SVC(gamma=2, C=1)}
classifiers_title = list(classifiers.keys())               
scores=np.empty(10)
means_scores=[]
stddev_scores=[]

#dividing the dataset into training and testing (training 60% and test=40%)
X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.4)


#Performing classification using each classifier and computing the 10-Fold cross-validation on results
for i in range(classifiers.__len__()):
   classifiers[classifiers_title[i]].fit (X_train,y_train)
   y_pred = classifiers[classifiers_title[i]].predict(X_test)
   scores = cross_val_score(classifiers[classifiers_title[i]],X,y,cv=10)
   # Plot normalized confusion matrix
   cnf_matrix = confusion_matrix(y_test, y_pred)
   np.set_printoptions(precision=2)
   plt.figure()
   plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,title=('Confusion matrix of ' + classifiers_title[i]))
   mean = scores.mean()
   stdev = scores.std()
   means_scores.append(mean)
   stddev_scores.append(stdev)
   print("[Results For ",classifiers_title[i], "] Mean: ",mean," Std Dev: ",stdev)
   

#plotting the bar chart showing each classifier's mean and std deviation of cross-validation score
fig,ax= plt.subplots()
rect1 = ax.bar(np.arange(6),means_scores,0.2,color='gray',yerr=stddev_scores)
ax.set_title('K-fold Cross-Validation Scores (K=10)',weight='bold')
ax.set_xticklabels(classifiers_title)
ax.set_ylabel('Cross-Validation Score Means',weight='bold')
autolabel(rect1)


#calculating the precision, recall, fscore and support for the best classifier i.e. Linear SVM
y_pred = classifiers['Linear SVM'].predict(X_test)
prec_rec_fscore_supt = precision_recall_fscore_support(y_test,y_pred)

#plotting precision,recall and fscore values for each class of users
fig,ax = plt.subplots()
x= np.arange(1,19)
ax.plot(x,prec_rec_fscore_supt[0],'o-')
ax.plot(x,prec_rec_fscore_supt[1],'o-')
ax.plot(x,prec_rec_fscore_supt[2],'o-')

ax.set_ylim(0.4,1.1)
ax.set_xlim(0,19)
ax.legend(['precision','recall','f-score'],loc='lower right')
ax.set_xlabel ('User Classes',weight='bold')
ax.set_title ('Precision, Recall and F-score values',weight='bold')

plt.xticks(x, class_names, rotation=45)