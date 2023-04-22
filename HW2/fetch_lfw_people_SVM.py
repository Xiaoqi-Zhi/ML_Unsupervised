from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

pca = PCA(n_components=150,whiten=True,random_state=114514)
svc = svm.SVC(kernel='linear', class_weight='balanced')
model1 = make_pipeline(pca, svc)
model2 = make_pipeline(svc)

faces = fetch_lfw_people(min_faces_per_person=60)
faces_data = faces.data
Xtrain, Xtest, Ytrain, Ytest = train_test_split(faces.data, faces.target, random_state=114514)

param_grid = {'svc__C': [1, 5, 10, 50]}
grid1 = GridSearchCV(model1,param_grid)
grid2 = GridSearchCV(model2,param_grid)

grid1.fit(Xtrain,Ytrain)
grid2.fit(Xtrain,Ytrain)

print(grid1.best_params_)
print(grid2.best_params_)

model1 = grid1.best_estimator_
model2 = grid2.best_estimator_
yfit1 = model1.predict(Xtest)
yfit2 = model2.predict(Xtest)
print("---------------PCA后SVM---------------")
print(classification_report(Ytest,yfit1,target_names=faces.target_names))
print("---------------直接SVM--------------")
print(classification_report(Ytest,yfit2,target_names=faces.target_names))

def plot_digits(data):
    fig,axes = plt.subplots(4,3,figsize=(15,15),
    subplot_kw={'xticks':[],'yticks':[]},
    gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i,ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(62,47),cmap='bone')
    plt.show()

pca2 = PCA(n_components=150)
pca2.fit(faces_data)
plot_digits(pca2.components_)



