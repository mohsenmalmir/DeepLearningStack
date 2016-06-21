from __future__ import print_function
import os
import sys
import time
import pickle
import numpy as np



# implementing Data loader, that manages loading of data, shuffling, mini batches and selection of next mini batch from specific
class DataLoader(object):
    """ load data from a list of files """
    def __init__(self,list_of_files,file_format,minibatch_size=None):
        """
            Create a input data layer that is 2D, with features lying in rows.
            
            :type list_of_files: tuple, containing the list of files to load from

            :type input: file_format: either mat or pickle
            : in case of matlab file, the last element in the list should be pickle
            : file with obj2label dictionary. Each mat file has these keys:
            : "data":x, "joints":p, "labels":y,
            : "track":t, ["mean":mean]
            : in case of pickle files, each file is
            : x,p,y,t,obj2label
            
        """
        self.load_data(list_of_files,file_format)#load data
        self.minibatch_idx = 0#initialize mini batch index
        if minibatch_size!=None:
            self.minibatch_size = minibatch_size
        
        
    def load_data(self,list_of_files,file_format):
        if file_format=="mat":
            print( "not implemented!")
            assert(False)
        elif file_format=="pkl":
            print( "loading pickled data")
            if len(list_of_files)>1:
                xs,ps,ys,ts = [],[],[],[]
                for fname in list_of_files:
                    f = file(fname, "rb")
                    x,p,y,t,self.obj2label = pickle.load(f)
                    f.close()
                    xs.append(x)
                    ys.append(y)
                    ts.append(t)
                    ps.append(p)
                self.x = np.concantenate(xs)
                self.y = np.concantenate(ys)
                self.t = np.concantenate(ts)
                self.p = np.concantenate(ps)
            else:
                f = file(list_of_files[0], "rb")
                self.x,self.p,self.y,self.t,self.obj2label = pickle.load(f)
                f.close()
            print( "data size:",self.x.shape)
            print( "target shape:",self.y.shape)
            print( "number of target classes:",np.unique(self.y).shape)
            print( "number of tracks:",np.unique(self.t).shape[0])
            assert(self.y.shape[0]==self.x.shape[0])
            assert(self.y.shape[0]==self.t.shape[0])
            assert(self.y.shape[0]==self.p.shape[0])
        else:
            print("wrong file format!")
            assert(False)
        self.update_tracks_record()#keep track of where different tracks are

    """ subtract self.mean or the supplied mean from the data"""
    def subtract_mean(self,mean=None):
        """
            the supplied 'mean' has higher priority if not None
        """
        if mean==None:
            self.x = self.x - self.data_mean
        else:
            self.x = self.x - mean

    """ create a map from (tracks,pose) to index of the data"""
    def update_tracks_record(self):
        self.track2idx = dict()#each track has two lists: one indicating the pose and the other indices
        tracks         = np.unique(self.t)
        for t in tracks:
            idx               = np.where(self.t == t)[0]
            pose              = self.p[idx]
            self.track2idx[t] = (pose,idx)


    """ returns the next mini batch according to the internal counter"""
    def get_next_minibatch(self):
        rng                    = range(self.minibatch_idx,self.minibatch_idx+self.minibatch_size)
        self.minibatch_idx    += self.minibatch_size
        if self.minibatch_idx >= self.x.shape[0]:#automatically reset the mini batch index to 0 if larger than the data size
            self.minibatch_idx = 0
        elif self.x.shape[0] - self.minibatch_idx < self.minibatch_size:
            self.minibatch_idx = self.x.shape[0] - self.minibatch_size
        return (self.x[rng],self.y[rng],self.p[rng],self.t[rng],rng)

    """ reset the internal mini-batch state"""
    def reset_minibatch_counter(self):
        #return data, labels and track information
        self.minibatch_idx = 0

    def shuffle_data(self):
        #shuffle data
        idx = range(self.x.shape[0])
        np.random.shuffle(idx)
        self.x = self.x[idx]
        self.y = self.y[idx]
        self.p = self.p[idx]
        self.t = self.t[idx]
        self.update_tracks_record()#update the map since the indices have changed
        return idx


    """ adapt the data labels to the specified obj2label dictionary """
    def adapt_labels(self,obj2label):
        #check if both dictionaries have the same set of object names
        #make sure that supplied object names are a superset of current object names
        for obj in self.obj2label.keys():
            assert(obj in obj2label.keys())
        newy = self.y.copy()
        for obj in self.obj2label.keys():
            idx        = np.where(self.y==self.obj2label[obj])
            newy[idx] = obj2label[obj]
        self.y = newy


    """ returns the specified data from each track and with the specified joint position (nearest neighbor) """
    def get_data_for_pose(self,tracks,ps):
        #for each specified track, find the closes joint position, stack the corresponding data and labels
        indices = []
        for t,p in zip(tracks,ps):
            pose,idx = self.track2idx[t]

            #wrap around the rotation
            if p > pose.max():
#                p = pose.min()
                p = p - pose.max()
            elif p > 0 and p < pose.min():
#                p = pose.max()
                p = pose.max() - p
            elif p < 0:
#                p = pose.max()
                p = pose.max() - (pose.min()-p)
            #wrap around the rotation

            diff     = np.abs(pose - p)
            closest  = np.argmin(diff)#find the data with the smallest distance
            indices.append(idx[closest])
        return (self.x[indices],self.y[indices],self.p[indices],self.t[indices],indices)

    def set_minibatch_size(self,sz):
        self.minibatch_size = minibatch_size










