import os
import sys
import cv2 
import pickle
from skimage import feature
import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

def loadImagesFromDir(folder_path, label):
    '''Get a path and load all images in grayscale'''
    images=[]
    labels=[]
    dir_list = os.listdir(folder_path)
    for img_name in dir_list:
        img = cv2.imread(f'{folder_path}/{img_name}')
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        labels.append(label)
        images.append(img_gray)
    return images,labels

def combineSet(folder_path):
    '''combine two genders to each set'''
    images,labels = loadImagesFromDir(folder_path+'/male',0)
    data = loadImagesFromDir(folder_path+'/female',1)
    images += data[0] 
    labels += data[1]

    return images,labels


def loadAllImages(train_path,val_path,test_path):
    '''
        Return for each set (images,labels,..)
        ret_val[0/1] = training
        ret_val[2/3] = val
        ret_val[4/5] = test
    '''
    train_images,train_lables = combineSet(train_path)
    val_images,val_lables = combineSet(val_path)
    test_images,test_lables = combineSet(test_path)

    return train_images,train_lables,val_images,val_lables,test_images,test_lables


def createLBPfeature(img,numPoints,radius):
    '''
        get an image and create LBP Feature and normalize it
    '''
    lbp = feature.local_binary_pattern(img,numPoints,radius,method='uniform')
    hist,_ = np.histogram(lbp.ravel(), bins=range(0, numPoints+3), range=(0,numPoints+2))
    #normalize
    eps=1e-7
    hist=hist.astype('float')
    hist/=(hist.sum()+eps)
    return hist

def getLBPFeatures(images,numPoints,radius):
    '''
        Get an image set and create features set
        with the desired params 
    '''
    features = []
    for img in images:
        features.append(createLBPfeature(img,numPoints,radius))
    return features

def getAllLBPFeatures(data):
    '''
        Create features for each combination
    '''
    train,val,test = data[0],data[2],data[4]
    train_1_8 = getLBPFeatures(train,8,1)
    val_1_8 = getLBPFeatures(val,8,1)
    test_1_8 = getLBPFeatures(test,8,1)
    train_3_24 = getLBPFeatures(train,24,3)
    val_3_24 = getLBPFeatures(val,24,3)
    test_3_24 = getLBPFeatures(test,24,3)
    return train_1_8,val_1_8,test_1_8,train_3_24,val_3_24,test_3_24

def getLinearModelAndScore(C,t_f,t_l,v_f,v_l):
    '''
        * create linear model
        * train it with t_f - traning features
        * return the model and the score he gets for validation set
    '''
    model = svm.SVC(kernel='linear',C=C)
    model.fit(t_f, t_l)
    return model,model.score(v_f,v_l)

def getRBFModelScore(param_grid,t_f,t_l,v_f,v_l):
    '''
        * create RBF model
        * find the best params in param grid
        * train it with t_f - traning features
        * return the model and the score he gets for validation set
    '''
    # Make grid search classifier
    clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
    # Train the classifier
    clf_grid.fit(t_f, t_l)
    # clf = grid.best_estimator_()
    b_p = clf_grid.best_params_
    clf = svm.SVC(kernel='rbf', C = b_p['C'], gamma=b_p['gamma'])
    model = clf.fit(t_f, t_l)
    score = clf.score(v_f,v_l)
    return model,score


def getTunedModel(features,labels):
    '''
        Runing validation of each params combination
        and return the best model and flag the say if its 1,8 or 3,24
    '''
    train_1_8,val_1_8,test_1_8,train_3_24,val_3_24,test_3_24 = features
    train_labels,val_labels,test_labels = labels

    feature_param = None
    model = None
    score = 0
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]}
   
    #Linear model
    for c in param_grid['C']:
        m_1_8,s_1_8 = getLinearModelAndScore(c,train_1_8,train_labels,val_1_8,val_labels)
        if s_1_8 > score:
            score = s_1_8
            model = m_1_8
            feature_param = False
        m_3_24,s_3_24 = getLinearModelAndScore(c,train_3_24,train_labels,val_3_24,val_labels)
        if s_3_24 > score:
            score = s_3_24
            model = m_3_24
            feature_param = True

    #rbf model
    m_1_8,s_1_8 = getRBFModelScore(param_grid,train_1_8,train_labels,val_1_8,val_labels)
    if s_1_8 > score:
        score = s_1_8
        model = m_1_8
        feature_param = False

    m_3_24,s_3_24 = getRBFModelScore(param_grid,train_3_24,train_labels,val_3_24,val_labels)
    if s_3_24 > score:
        score = s_3_24
        model = m_3_24
        feature_param = True

    return model,feature_param


def modelTesting(model,t_f,t_l,f_p):
    '''
        Runing test set on in the model and writing the result to result.txt, and confusion_matrix.txt
    '''
    # Write the params
    f = open('results.txt','w')
    f.write('MODEL RESULTS\n\n1) Highest precision params:\n')
    f.write(f'Feature Params -> {"radius = 3, numOfPoints = 24" if f_p else "radius = 1, numOfPoints = 8"}\n')
    f.write(f'Model Params -> Kernel = {model.kernel}')
    par_str = f', C = {model.C}\n' if model.kernel == 'linear' else f', C = {model.C}, gamma = {model.gamma}\n'
    f.write(par_str)
    f.write('________________________________________________\n\n')
    # write the score
    score = model.score(t_f,t_l)
    f.write(f'2) Model precision:\nAccuracy = {score*100:.2f}%\n')
    f.write('________________________________________________\n\n')
    #create confusion matrix and write it
    predicts = model.predict(t_f)
    cm=confusion_matrix(t_l, predicts, labels=[0,1])
    f.write('3) Confusion matrix:\n\n')
    f.write(f'{"":<8}')
    f.write(f'{"|":<3}{"Male":<5}{"|":<2}{"Female":<6}\n')
    f.write('-------------------------\n')
    f.write(f'{"Male":<8}{"|":<4}{cm[0][0]:<4}{"|":<4}{cm[0][1]:<4}\n')
    f.write('-------------------------\n')
    f.write(f'{"Female":<8}{"|":<4}{cm[1][0]:<4}{"|":<4}{cm[1][1]:<4}\n')
    f.close()


def main():
    # Get the paths and load all images in grayscale
    train_path, val_path, test_path = sys.argv[1],sys.argv[2],sys.argv[3]
    data = loadAllImages(train_path,val_path,test_path)
    
    # lables and features
    labels = (data[1],data[3],data[5])
    features = getAllLBPFeatures(data)

    # Running training and validation and get the best model
    # f_p represent if its 1,8 or 3,24
    model,f_p = getTunedModel(features,labels)
    test_features = features[5] if f_p else features[2]

    # Runing the test set, and create output
    modelTesting(model,test_features,labels[2],f_p)


if __name__ == "__main__":
    main()










# def main():
#     ## Main that load features from file
#     # data = loadAllImages('train','val','test')
#     # filehandler = open('data.txt', 'wb') 
#     # pickle.dump(data, filehandler)
#     filehandler = open('data.txt', 'rb') 
#     data = pickle.load(filehandler)
    
#     labels = (data[1],data[3],data[5])

#     # features = getAllLBPFeatures(data)
#     # filehandler = open('features.txt', 'wb') 
#     # pickle.dump(features, filehandler)
#     filehandler = open('features.txt', 'rb') 
#     features = pickle.load(filehandler)

#     model,f_p = getTunedModel(features,labels)
#     test_features = features[5] if f_p else features[2]

#     modelTesting(model,test_features,labels[2],f_p)
