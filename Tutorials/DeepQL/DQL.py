from __future__ import print_function
import sys
import select
import tty
import termios
import time
import theano
import pprint
import theano.tensor as T
#import cv2
import cPickle
import copy
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from theano.tensor.shared_randomstreams import RandomStreams
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from DataLoader import DataLoader
import DeepLearningStack
from DeepLearningStack import FeedForwardNet


"""
This code implements the Deep Q-Learning model for active object recognition,
described in

Malmir M, Sikka K, Forster D, Movellan J, Cottrell GW. 
Deep Q-learning for Active Recognition of GERMS: 
Baseline performance on a standardized dataset for active learning. 
InProceedings of the British Machine Vision Conference (BMVC), 
pages 2016 Apr 13 (pp. 161-1).


This code requires the following data files:
train-[ARM].pickle
test-[ARM].pickle
val-[ARM].pickle

These files contain the belief encoding of single images of GERMS,
using features obtained from VGG deep network trained on ImageNet.
Data files can be found here in VGG-Beliefs folder:
https://drive.google.com/folderview?id=0BxZOUQHBUnwmQUdWRGlPMGw4WHM&usp=sharing


The code for VGG model is obtained from:
http://www.robots.ox.ac.uk/~vgg/software/deep_eval/


"""


batch_size = 128
D          = 136#number of classes

arm = "left"
#load the data
print("##################################################")
print("loading train data")
data_files = ["train-"+arm+".pickle"]
train_data = DataLoader(data_files,"pkl",minibatch_size=batch_size)
train_data.shuffle_data()
C = np.unique(train_data.y).shape[0]
#print(train_data.y == np.argmax(train_data.x,axis=1)).mean()
print("data size:",train_data.x.shape)
print("number of classes:",C)
print("number of tracks:",np.unique(train_data.t).shape[0])


print("##################################################")
print("loading validation data")
data_files = ["val-"+arm+".pickle"]
val_data   = DataLoader(data_files,"pkl",minibatch_size=batch_size)
val_data.shuffle_data()
C = np.unique(val_data.y).shape[0]
val_data.adapt_labels(train_data.obj2label)#data are already unified in their labels
print("data size:",val_data.x.shape)
print("number of classes:",C)
print("number of tracks:",np.unique(val_data.t).shape[0])



print("##################################################")
print("loading test data")
data_files = ["test-"+arm+".pickle"]
test_data  = DataLoader(data_files,"pkl",minibatch_size=batch_size)
test_data.adapt_labels(train_data.obj2label)#data are already unified in their labels
test_data.shuffle_data()
print("data size:",test_data.x.shape)
print("number of classes:",np.unique(test_data.y).shape[0])
print("number of tracks:",np.unique(test_data.t).shape[0])






experiment_data = dict()
#train 20 different models, report the mean average

for exp_num in range(20):
    
    test_data.shuffle_data()
    
    print( "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print( "EXPERIMENT ", exp_num)
    print( "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    experiment_data[exp_num] = dict()

    lr          = 0.01#initial learning rate
    lr_dec_step = 1000#learning rate decrease step
    lr_dec_start= 0
    num_actions = 10#number of actions
    gamma       = 0.9#RL discount factor
    alpha       = 0.01#stochastic approximation coefficient
    R           = 10.0#reward
    n_moves     = 5#length of object inspection sequence
    n_test_moves= 5#length of inspection sequence for test objects
    epsilon     = 1.#for e-greedy annealing
    iter_cnt    = 1
    epsilon_dec_step = 100
    n_epochs    = 50


    #create deep net
    print("##################################################")
    print("Creating deep net...")
    input      = T.matrix("data",dtype=theano.config.floatX)#the input is concatenation of action history and beliefs
    config     = "DQLArch.xml"
    rng        = RandomStreams(seed=int(time.time()))
    train_net  = FeedForwardNet.FeedForwardNet(rng,input,config)
    test_net   = FeedForwardNet.FeedForwardNet(rng,input,config,clone_from=train_net)
    pprint.pprint(train_net.output_dims)

    print("##################################################")
    print("creating cost layer...")
    input_shared          = theano.shared(np.zeros([D,batch_size],dtype=theano.config.floatX),borrow=True)
    rot_target_shared     = theano.shared(np.zeros([batch_size,],dtype=theano.config.floatX),borrow=True)
    rot_index_shared      = theano.shared(np.zeros([batch_size,],dtype=np.int32),borrow=True)
    learning_rate         = T.scalar("learning_rate",dtype=theano.config.floatX)
    #target value consists of the target for rotation values and the target for sign-of-rotation values
    layer_action_value    = "fc3"
    layer_action          = "act1"
    cost                  = T.sqrt( T.mean( (train_net.name2layer[layer_action_value].output[rot_index_shared,T.arange(batch_size)] -  rot_target_shared)**2 ) )

    grads                 = [theano.grad(cost,param) for param in train_net.params]

    updates               = [ (param,param-learning_rate * grad) for param,grad in zip(train_net.params,grads)]

    fnx_action_selection  = theano.function(inputs=[],outputs=[train_net.name2layer[layer_action].output,train_net.name2layer[layer_action_value].output,cost],
                                                givens={
                                                            input:input_shared,
                                                        },
                                            )
    fnx_train             = theano.function(inputs=[learning_rate],outputs=[train_net.name2layer[layer_action_value].output,cost],
                                                    givens={
                                                            input:input_shared,
                                                           },
                                                    updates=updates,
                                            )


    fnx_test              = theano.function(inputs=[],outputs=[test_net.name2layer[layer_action].output,test_net.name2layer[layer_action_value].output],
                                            givens={
                                                    input:input_shared,
                                                },
                                            )

    print("##################################################")
    print("classifying tracks tracks")
    track_indices  = dict()
    for t in np.unique(test_data.t):
        idx        = np.where(test_data.t == t)[0]
        poses      = test_data.p[idx]
        sorted_idx = np.argsort(poses)
        track_indices[t] = idx[sorted_idx]
    accuracy = np.zeros(n_test_moves)
    for t in track_indices.keys():
        #    print"for track:",t
        #    print"number of frames:",len(track_indices[t])
        belief   = np.ones(C)
        #    print
        for i in range(n_test_moves):
            next_idx = track_indices[t][i]
            belief   = belief * test_data.x[next_idx,:]
            belief   = belief / belief.sum()
            lbl      = np.argmax(belief)
            accuracy[i] += lbl == test_data.y[next_idx]
    print("test-sequential",accuracy / np.unique(test_data.t).shape[0])
    seq_acc = accuracy / np.unique(test_data.t).shape[0]
    experiment_data[exp_num]["test_seq_acc"] = seq_acc


    test_data.reset_minibatch_counter()
    corrects    = np.zeros([n_test_moves,batch_size])
    for i in range(test_data.x.shape[0] / batch_size + 1):
        x,y,p,t,rng    = test_data.get_next_minibatch()
        beliefs        = x.copy()
        for mv in range(n_test_moves):
            pred_rslt      = np.argmax(beliefs,axis=1)
            corrects[mv,:] += (pred_rslt==y)
            rot            = np.random.randint(0,num_actions,[batch_size])
    #        rot            = num_actions/2 *  np.ones([batch_size])
            tgt            = -1 * ( rot < num_actions/2) * 2**(rot+num_actions/2) + ( rot >= num_actions/2) * 2**(rot)
    #        printtgt+p
            x,y,p,t,_      = test_data.get_data_for_pose(t,p + tgt)#get the data for the proposed set of rotations
            beliefs        = beliefs * x
            beliefs        = beliefs / beliefs.sum(axis=1).reshape([-1,1])
    print("test-random:",corrects.sum(axis=1) / float(test_data.x.shape[0]))
    rnd_acc = corrects.sum(axis=1) / float(test_data.x.shape[0])
    experiment_data[exp_num]["test_rnd_acc"] = rnd_acc

    print("##################################################")
    print("training network...")
    test_accuracies = []
    costs           = []
    test_costs      = []
    for epoch in range(n_epochs):
        print("Epoch:",epoch)
        train_data.reset_minibatch_counter()
        corrects         = np.zeros([n_moves,batch_size])
        move_hist        = np.zeros([num_actions,],dtype=np.int32)
        poses_hist       = []
        
        test_data.reset_minibatch_counter()
        test_move_hist   = np.zeros([num_actions,],dtype=np.int32)
        
        for i in range(train_data.x.shape[0] / batch_size + 1):
            if i is 0:
                print("iteration:",iter_cnt)
            alpha          = max(0.00, 1. - iter_cnt / 20000.)#1. / iter_cnt
            x,y,p,t,rng    = train_data.get_next_minibatch()
            poses_hist.append(p)
            beliefs        = x.copy()
            for mv in range(n_moves):
                iter_cnt      += 1
                epsilon = max(0.00, 0.5 - iter_cnt / 800.)
#                lr      = max(1.e-7, 0.01 * (1. - iter_cnt / 16000.) )
                if iter_cnt>=lr_dec_start and iter_cnt%lr_dec_step==lr_dec_step-1:
                    lr = max(1.e-10,lr * 0.1)
                input_shared.set_value(beliefs.T.astype(theano.config.floatX))
                rot,prot,_     = fnx_action_selection()
                rot            = rot.reshape(-1)
                #epsilon-greedy exploration
                rand_acts      = np.random.randint(0,num_actions,[batch_size])
                rand_mask      = np.random.binomial(1,epsilon,[batch_size])
                idx_random     = np.where(rand_mask == 1)[0]
                idx_net_act    = np.where(rand_mask != 1)[0]
                temp111        = rot.copy()
                temp111[idx_random] = rand_acts[idx_random]
                temp111[idx_net_act]= rot[idx_net_act]
                rot            = temp111
                rot_idx        = rot.copy().astype(np.int32)
                assert(rot_idx.shape[0]==batch_size)
                hst            = np.histogram(rot_idx,bins=range(0,num_actions))[0]
                for kk in range(rot_idx.shape[0]):
                    move_hist[rot_idx[kk]] += 1
                tgt            = -1 * ( rot < num_actions/2) * 2**(rot+num_actions/2) + ( rot >= num_actions/2) * 2**(rot)
                x1,y1,p1,t1,_  = train_data.get_data_for_pose(t,p + tgt)#get the data for the proposed set of rotations
                assert((t1==t).sum()==batch_size)
                assert((y1==y).sum()==batch_size)
                x1             = x1 * beliefs
                x1             = x1 / x1.sum(axis=1).reshape([-1,1])
                input_shared.set_value(x1.T.astype(theano.config.floatX))
                rot1,prot1,_   = fnx_action_selection()#calculate the Q(s,a) for all as in the next state
                pred_rslt      = np.argmax(x1,axis=1)#x1 should be 'beliefs' if we use Q = r(t) + gamma max_a Q(s,a').
                prot_max       = gamma * np.max(prot1,axis=0).reshape(-1).astype(theano.config.floatX)
                #reward each move based on the amount of belief increase
                srtd_beliefs   = np.sort(x1,axis=1)#x1 should be 'beliefs' if we use Q = r(t) + gamma max_a Q(s,a').
                if mv == n_moves-1:
                    prot_max      += R * (pred_rslt==y1)* (srtd_beliefs[:,-1] - srtd_beliefs[:,-2]).reshape(-1)
                    prot_max      -= R * (pred_rslt!=y1)
                prot_max       = alpha * prot_max + (1-alpha) * prot[rot_idx,range(batch_size)].reshape(-1)
                corrects[mv,:] += (pred_rslt==y)
                input_shared.set_value(beliefs.T.astype(theano.config.floatX))
                rot_target_shared.set_value(prot_max.astype(theano.config.floatX))
                rot_index_shared.set_value(rot_idx.reshape(-1))
                prot2,c         = fnx_train(lr)
                costs.append(c)
                x,y,p,t,_       = train_data.get_data_for_pose(t,p + tgt)#get the data for the proposed set of rotations
                poses_hist.append(p)
                beliefs         = beliefs * x
                beliefs         = beliefs / beliefs.sum(axis=1).reshape([-1,1])



            x,y,p,t,rng    = test_data.get_next_minibatch()
            beliefs        = x.copy()
            prev_actval    = np.zeros([batch_size,]).reshape(-1)
            for mv in range(n_moves+1):
                input_shared.set_value(beliefs.T.astype(theano.config.floatX))
                rot,prot       = fnx_test()
                rand_acts      = np.random.randint(0,num_actions,[batch_size])
                rand_mask      = np.random.binomial(1,epsilon,[batch_size])
                rot            = (1-rand_mask) * rot + rand_mask * rand_acts
                if mv>0:
                    c          = np.sqrt(np.mean((prot.max(axis=0).reshape(-1) - prev_actval)**2))
                    test_costs.append(c)
                prev_actval    = prot[rot,range(batch_size)].reshape(-1)
                rot_idx        = rot.copy().astype(np.int32)
                for kk in range(rot_idx.shape[1]):
                    test_move_hist[rot_idx[0,kk]] += 1
                rot            = rot.reshape(-1)
                tgt            = -1 * ( rot < num_actions/2) * 2**(rot+num_actions/2) + ( rot >= num_actions/2) * 2**(rot)

                pred_rslt      = np.argmax(beliefs,axis=1)
    #            test_corrects[mv,:]+= (pred_rslt==y)
                x,y,p,t,_      = test_data.get_data_for_pose(t,p + tgt)#get the data for the proposed set of rotations
                beliefs        = beliefs * x
                beliefs        = beliefs / beliefs.sum(axis=1).reshape([-1,1])

        print("epoch cost:",np.sum(costs))
        print("train accuracy:",corrects.sum(axis=1) / float(train_data.x.shape[0]))
        print("learning rate:",lr," RL epsilon:",epsilon)



        test_poses_hist    = []
        corrects           = np.zeros([n_test_moves,batch_size])
        test_data.reset_minibatch_counter()
        test_move_hist     = np.zeros([num_actions,],dtype=np.int32)
        for i in range(test_data.x.shape[0] / batch_size + 1):
            x,y,p,t,rng    = test_data.get_next_minibatch()
            test_poses_hist.append(p)
            beliefs        = x.copy()
            move_hist      = np.zeros([n_test_moves,batch_size])
            for mv in range(n_test_moves):
                input_shared.set_value(beliefs.T.astype(theano.config.floatX))
                rot            = fnx_test()[0]
                rot            = rot.reshape(-1)
                rand_acts      = np.random.randint(0,num_actions,[batch_size])
                rand_mask      = np.random.binomial(1,epsilon,[batch_size])
                idx_random     = np.where(rand_mask == 1)[0]
                idx_net_act    = np.where(rand_mask != 1)[0]
                temp111        = rot.copy()
                temp111[idx_random] = rand_acts[idx_random]
                temp111[idx_net_act]= rot[idx_net_act]
                rot            = temp111
                rot_idx        = rot.copy().astype(np.int32)
                for kk in range(rot_idx.shape[0]):
                    test_move_hist[rot_idx[kk]] += 1
                tgt            = -1 * ( rot < num_actions/2) * 2**(rot+num_actions/2) + ( rot >= num_actions/2) * 2**(rot)
                pred_rslt      = np.argmax(beliefs,axis=1)
                corrects[mv,:] += (pred_rslt==y)
                x,y,p,t,_      = test_data.get_data_for_pose(t,p + tgt)#get the data for the proposed set of rotations
                test_poses_hist.append(p)
                beliefs        = beliefs * x
                beliefs        = beliefs / beliefs.sum(axis=1).reshape([-1,1])
            hst            = np.histogram(move_hist.reshape(-1),bins=range(0,num_actions))[0]
        pprint.pprint(test_move_hist)
        print"test:",corrects.sum(axis=1) / float(test_data.x.shape[0])
        test_accuracies.append(corrects.sum(axis=1) / float(test_data.x.shape[0]))

    experiment_data[exp_num]["test_dpq_acc"] = corrects.sum(axis=1) / float(test_data.x.shape[0])
    experiment_data[exp_num]["test_RMSE"]    = test_costs
    experiment_data[exp_num]["train_RMSE"]   = costs
    experiment_data[exp_num]["train_net"]    = copy.deepcopy(train_net)
    experiment_data[exp_num]["test_net"]     = test_net
    experiment_data[exp_num]["train_poses_hist"]     = poses_hist
    experiment_data[exp_num]["test_poses_hist"]     = test_poses_hist


colors    = ["r","g","b","c","y","m","k"]
linestyle = ["-","--","-.",":"]
marker    = ["o","v","^","<",">","*"]
i = 0
seq_acc = []
rnd_acc = []
dpq_acc = []
for i in experiment_data.keys():
    seq_acc.append(experiment_data[i]["test_seq_acc"].reshape([1,-1]))
    rnd_acc.append(experiment_data[i]["test_rnd_acc"].reshape([1,-1]))
    dpq_acc.append(experiment_data[i]["test_dpq_acc"].reshape([1,-1]))


seq_acc = np.concatenate(seq_acc,axis=0)
rnd_acc = np.concatenate(rnd_acc,axis=0)
dpq_acc = np.concatenate(dpq_acc,axis=0)

plt.figure(1)
plt.hold(True)
i = 0
plt.errorbar(x=range(n_test_moves),y=dpq_acc.mean(axis=0),xerr=0,yerr=dpq_acc.std(axis=0),color=colors[i%len(colors)],linestyle=linestyle[i%len(linestyle)],label="DeepQ"+str(i),marker=marker[i%len(marker)],linewidth=4.)
i += 1
plt.errorbar(x=range(n_test_moves),y=seq_acc.mean(axis=0),xerr=0,yerr=seq_acc.std(axis=0),color=colors[i%len(colors)],linestyle=linestyle[i%len(linestyle)],label="sequential",marker=marker[i%len(marker)],linewidth=4.)
i += 1
plt.errorbar(x=range(n_test_moves),y=rnd_acc.mean(axis=0),xerr=0,yerr=rnd_acc.std(axis=0),color=colors[i%len(colors)],linestyle=linestyle[i%len(linestyle)],label="random",marker=marker[i%len(marker)],linewidth=4.)

handles, labels = plt.gca().get_legend_handles_labels()
plt.gca().legend(handles[::-1], labels[::-1],loc=4)
plt.xlabel("Number of Actions")
plt.ylabel("Accuracy")
for axis in ['top','bottom','left','right']:
    plt.gca().spines[axis].set_linewidth(2.0)
i += 1
plt.xticks(range(n_test_moves+1))

plt.figure(2)
i=2
plt.plot(np.log(experiment_data[i]["train_RMSE"]),c='b')
plt.hold(True)
plt.plot(np.log(experiment_data[i]["test_RMSE"]),c='r')


plt.show()


print("saving experiment results...")
f = open("expresults-"+arm+"-"+str(n_test_moves)+".pickle","wb")
cPickle.dump(experiment_data,f,protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
