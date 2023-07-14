import numpy as np
from sklearn import preprocessing, model_selection, neighbors, tree
import pandas as pd
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,precision_score, recall_score, f1_score
from matplotlib import pyplot as plt

data=pd.read_csv('adv_stress.csv')
pd.set_option('display.max_columns', 100)
#print(data.describe())

X= np.array(data.drop(['sl'],axis=1))
Y= np.array(data['sl'])

X_train, X_test, Y_train, Y_test =model_selection.train_test_split(X,Y,test_size=0.2)

clf= neighbors.KNeighborsClassifier(n_neighbors=100)
clf.fit(X_train,Y_train)
rfc=tree.DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_split=2)
rfc.fit(X_train,Y_train)
Y_pred=rfc.predict(X_test)

accuracy =rfc.score(X_test, Y_test)
print(accuracy)
print(confusion_matrix(Y_test,Y_pred))

print(precision_score(Y_test,Y_pred,average='micro'))
print(recall_score(Y_test,Y_pred,average='weighted'))
print(f1_score(Y_test,Y_pred,average='micro'))
plot_confusion_matrix(rfc,X_test,Y_test)
plt.show()

ex=np.array([[4, 3, 4, 3, 3, 2, 2, 3, 4, 3, 80.0, 15.0, 102.0, 5.0, 99.0, 50.0, 8.0, 4, 30.0]])
print(len(ex))
ex=ex.reshape(len(ex),-1)
pred=clf.predict(ex)
pred1=rfc.predict(ex)
print(pred,clf.predict_proba(ex)[0][pred[0]])
print(pred1,rfc.predict_proba(ex))
