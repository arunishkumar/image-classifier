import numpy as np
import h5py
def load_dataset():
    train_dataset = h5py.File("datasets/train_catvnoncat.h5","r")#opening file
    
    train_set_x_orig=np.array(train_dataset["train_set_x"][:])#extracted training set data
    train_set_y_orig=np.array(train_dataset["train_set_y"][:])#training set predictions
    classes=np.array(train_dataset["list_classes"][:])#an array having cat and non-cat stored
    
    test_dataset = h5py.File("datasets/test_catvnoncat.h5","r")
    test_set_x_orig=np.array(test_dataset["test_set_x"][:])#test set data
    test_set_y_orig=np.array(test_dataset["test_set_y"][:])#test set predictions
    
    train_set_y=np.array(train_set_y_orig.reshape(1,train_set_y_orig.shape[0]))#converting train datasets to a row vector
    test_set_y=np.array(test_set_y_orig.reshape(1,test_set_y_orig.shape[0]))
    
    return train_set_x_orig,test_set_x_orig,train_set_y,test_set_y,classes