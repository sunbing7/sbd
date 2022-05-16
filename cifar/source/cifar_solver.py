import keras
from keras import applications
from keras import backend as K
import tensorflow as tf
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import time
import imageio
import utils_backdoor

from keras.layers import Input
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
import copy
import random

import os
import tensorflow

import sys
sys.path.append('../../')

DATA_DIR = '../../data'  # data folder
DATA_FILE = 'cifar.h5'  # dataset file
NUM_CLASSES = 10
BATCH_SIZE = 32
RESULT_DIR = "../results/"


class solver:
    CLASS_INDEX = 1
    ATTACK_TARGET = 6
    VERBOSE = True


    def __init__(self, model, verbose, mini_batch, batch_size):
        self.model = model
        self.splited_models = []

        self.target = self.ATTACK_TARGET
        self.current_class = self.CLASS_INDEX
        self.verbose = verbose
        self.mini_batch = mini_batch
        self.batch_size = batch_size
        self.layer = [2, 6, 13]
        self.classes = [0,1,2,3,4,5,6,7,8,9]
        self.random_sample = 1 # how many random samples
        self.plot = False
        self.rep_n = 0
        self.rep_neuron = []
        self.num_target = 1
        self.base_class = None
        self.target_class = None

        self.kmeans_range = 10
        # split the model for causal inervention
        pass

    def split_keras_model(self, lmodel, index):

        model1 = Model(inputs=lmodel.inputs, outputs=lmodel.layers[index - 1].output)
        model2_input = Input(lmodel.layers[index].input_shape[1:])
        model2 = model2_input
        for layer in lmodel.layers[index:]:
            model2 = layer(model2)
        model2 = Model(inputs=model2_input, outputs=model2)

        return (model1, model2)

    def split_model(self, lmodel, indexes):
        # split the model to n sub models
        models = []
        model = Model(inputs=lmodel.inputs, outputs=lmodel.layers[indexes[0]].output)
        models.append(model)
        for i in range (1, len(indexes)):
            model_input = Input(lmodel.layers[(indexes[i - 1] + 1)].input_shape[1:])
            model = model_input
            for layer in lmodel.layers[(indexes[i - 1] + 1):(indexes[i] + 1)]:
                model = layer(model)
            model = Model(inputs=model_input, outputs=model)
            models.append(model)

        # output
        model_input = Input(lmodel.layers[(indexes[len(indexes) - 1] + 1)].input_shape[1:])
        model = model_input
        for layer in lmodel.layers[(indexes[len(indexes) - 1] + 1):]:
            model = layer(model)
        model = Model(inputs=model_input, outputs=model)
        models.append(model)

        return models

    # util function to convert a tensor into a valid image
    def deprocess_image(self, x):
        # normalize tensor: center on 0., ensure std is 0.1
        #'''
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255

        x = np.clip(x, 0, 255).astype('uint8')
        '''
        x = np.clip(x, 0, 1)
        '''
        return x

    def solve(self, gen, train_adv_gen, test_adv_gen):

        # analyze hidden neuron importancy
        start_time = time.time()
        self.solve_analyze_hidden(gen, train_adv_gen, test_adv_gen)
        analyze_time = time.time() - start_time

        # detect semantic backdoor
        bd = self.solve_detect_semantic_bd()
        detect_time = time.time() - analyze_time - start_time

        if len(bd) == 0:
            print('No abnormal detected!')
            return

        # identify candidate neurons for repair: outstanding neurons from base class to target class
        for i in range (0, len(bd)):
            if i == 0:
                candidate = self.locate_candidate_neuron(bd[i][0], bd[i][1])
            else:
                candidate = np.append(candidate, self.locate_candidate_neuron(bd[i][0], bd[i][1]), axis=0)

        # remove duplicates
        candidate = set(tuple(element) for element in candidate)
        candidate = np.array([list(t) for t in set(tuple(element) for element in candidate)])

        self.rep_n = int(len(candidate) * 1.0)

        top_neuron = candidate[:self.rep_n,:]

        ind = np.argsort(top_neuron[:,0])
        top_neuron = top_neuron[ind]

        print('Number of neurons to repair:{}'.format(self.rep_n))

        np.savetxt(RESULT_DIR + 'rep_neu.txt', top_neuron, fmt="%s")

        for l in self.layer:
            idx_l = []
            for (i, idx) in top_neuron:
                if l == i:
                    idx_l.append(int(idx))
            self.rep_neuron.append(idx_l)

        # repair
        #self.repair(base_class, target_class)
        print('analyze time: {}'.format(analyze_time))
        print('detect time: {}'.format(detect_time))
        pass

    def solve_detect_semantic_bd(self):
        # analyze class embedding
        ce_bd = self.solve_analyze_ce()
        if len(ce_bd) != 0:
            print('Semantic attack detected ([base class, target class]): {}'.format(ce_bd))
            return ce_bd

        bd = []
        bd.extend(self.solve_detect_common_outstanding_neuron())
        bd.extend(self.solve_detect_outlier())

        if len(bd) != 0:
            print('Potential semantic attack detected ([base class, target class]): {}'.format(bd))
        return bd

    def solve_analyze_hidden(self, gen, train_adv_gen, test_adv_gen):
        '''
        analyze hidden neurons and find important neurons for each class
        '''
        print('Analyzing hidden neuron importancy.')
        for each_class in self.classes:
            self.current_class = each_class
            print('current_class: {}'.format(each_class))
            self.analyze_eachclass_expand(gen, each_class, train_adv_gen, test_adv_gen)

        pass

    def solve_analyze_ce(self):
        '''
        analyze hidden neurons and find class embeddings
        '''
        flag_list = []
        print('Analyzing class embeddings.')
        for each_class in self.classes:
            self.current_class = each_class
            if self.verbose:
                print('current_class: {}'.format(each_class))
            ce = self.analyze_eachclass_ce(each_class)
            pred = np.argmax(ce, axis=1)
            if pred != each_class:
                flag_list.append([each_class, pred[0]])

        return flag_list

    def solve_detect_common_outstanding_neuron(self):
        '''
        find common outstanding neurons
        return potential attack base class and target class
        '''
        print('Detecting common outstanding neurons.')

        flag_list = []
        top_list = []
        top_neuron = []

        for each_class in self.classes:
            self.current_class = each_class
            if self.verbose:
                print('current_class: {}'.format(each_class))

            top_list_i, top_neuron_i = self.detect_eachclass_all_layer(each_class)
            top_list = top_list + top_list_i
            top_neuron.append(top_neuron_i)
            #self.plot_eachclass_expand(each_class)

        #top_list dimension: 10 x 10 = 100
        flag_list = self.outlier_detection(top_list, max(top_list))
        base_class, target_class = self.find_target_class(flag_list)

        if len(flag_list) == 0:
            return []

        if self.num_target == 1:
            base_class = int(base_class[0])
            target_class = int(target_class[0])

        #print('Potential semantic attack detected (base class: {}, target class: {})'.format(base_class, target_class))

        return [[base_class, target_class]]

    def solve_detect_outlier(self):
        '''
        analyze outliers to certain class, find potential backdoor due to overfitting
        '''
        print('Detecting outliers.')

        tops = []   #outstanding neuron for each class

        for each_class in self.classes:
            self.current_class = each_class
            if self.verbose:
                print('current_class: {}'.format(each_class))

            #top_ = self.find_outstanding_neuron(each_class, prefix="all_")
            top_ = self.find_outstanding_neuron(each_class, prefix="")
            tops.append(top_)

        save_top = []
        for top in tops:
            save_top = [*save_top, *top]
        save_top = np.array(save_top)
        flag_list = self.outlier_detection(1 - save_top/max(save_top), 1)
        np.savetxt(RESULT_DIR + "outlier_count.txt", save_top, fmt="%s")

        base_class, target_class = self.find_target_class(flag_list)

        out = []
        for i in range (0, len(base_class)):
            if base_class[i] != target_class[i]:
                out.append([base_class[i], target_class[i]])

        return out

    def find_target_class(self, flag_list):
        #if len(flag_list) < self.num_target:
        #    return None
        a_flag = np.array(flag_list)

        ind = np.argsort(a_flag[:,1])[::-1]
        a_flag = a_flag[ind]

        base_classes = []
        target_classes = []

        i = 0
        for (flagged, mad) in a_flag:
            base_class = int(flagged / NUM_CLASSES)
            target_class = int(flagged - NUM_CLASSES * base_class)
            base_classes.append(base_class)
            target_classes.append(target_class)
            i = i + 1
            #if i >= self.num_target:
            #    break

        return base_classes, target_classes


    def analyze_eachclass_ce(self, cur_class):
        '''
        use samples from base class, find class embedding
        '''
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        ce = self.hidden_ce_test_all(class_gen, cur_class)
        return ce

    def analyze_eachclass_expand(self, gen, cur_class, train_adv_gen, test_adv_gen):
        '''
        use samples from base class, find important neurons
        '''
        ana_start_t = time.time()
        self.verbose = False
        x_class, y_class = load_dataset_class(cur_class=cur_class)
        class_gen = build_data_loader(x_class, y_class)

        hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)

        hidden_test_all = []
        hidden_test_name = []

        for this_class in self.classes:
            hidden_test_all_ = []
            for i in range (0, len(self.layer)):

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

            hidden_test_all.append(hidden_test_all_)

            hidden_test_name.append('class' + str(this_class))

        if self.plot:
            self.plot_multiple(hidden_test_all, hidden_test_name, save_n="test")

        pass

    def plot_eachclass_expand(self,  cur_class, prefix=""):
        # find hidden neuron permutation on cmv images
        #hidden_cmv = self.hidden_permutation_cmv_all(gen, img, cur_class)
        '''
        hidden_cmv = []
        for cur_layer in self.layer:
            hidden_cmv_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_cmv.append(hidden_cmv_)
        hidden_cmv = np.array(hidden_cmv)
        '''
        #hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test.append(hidden_test_)
        hidden_test = np.array(hidden_test)

        #adv_train = self.hidden_permutation_adv_all(train_adv_gen, cur_class)
        '''
        if cur_class == 6:
            adv_train = []
            for cur_layer in self.layer:
                adv_train_ = np.loadtxt(RESULT_DIR + "adv_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
                adv_train.append(adv_train_)
            adv_train = np.array(adv_train)
        #adv_test = self.hidden_permutation_adv(test_adv_gen, cur_class)
        '''
        #hidden_cmv_all = []
        #hidden_cmv_name = []
        hidden_test_all = []
        hidden_test_name = []
        #adv_train_all = []
        #adv_train_name = []
        #adv_test_all = []
        #adv_test_name = []
        for this_class in self.classes:
            hidden_cmv_all_ = []
            hidden_test_all_ = []
            adv_train_all_ = []
            for i in range (0, len(self.layer)):
                #temp = hidden_cmv[i][:, [0, (this_class + 1)]]
                #hidden_cmv_all_.append(temp)

                temp = hidden_test[i][:, [0, (this_class + 1)]]
                hidden_test_all_.append(temp)

                #if cur_class == 6:
                #    temp = adv_train[i][:, [0, (this_class + 1)]]
                #    adv_train_all_.append(temp)

            #hidden_cmv_all.append(hidden_cmv_all_)
            hidden_test_all.append(hidden_test_all_)

            #hidden_cmv_name.append('class' + str(this_class))
            hidden_test_name.append('class' + str(this_class))

            #if cur_class == 6:
            #    adv_train_all.append(adv_train_all_)
            #    adv_train_name.append('class' + str(this_class))

        #self.plot_multiple(hidden_cmv_all, hidden_cmv_name, save_n="cmv")
        self.plot_multiple(hidden_test_all, hidden_test_name, save_n=prefix + "test")
        #if cur_class == 6:
        #    self.plot_multiple(adv_train_all, adv_train_name, save_n="adv_train")
            #self.plot_multiple(adv_test_all, adv_test_name, save_n="adv_test")

        pass

    def detect_eachclass_all_layer(self,  cur_class):
        #hidden_test = self.hidden_permutation_test_all(class_gen, cur_class)
        hidden_test = []
        for cur_layer in self.layer:
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)

        hidden_test = np.array(hidden_test)

        # check common important neuron
        #num_neuron = int(self.top * len(hidden_test[i]))

        # get top self.top from current class
        temp = hidden_test[:, [0, 1, (cur_class + 2)]]
        ind = np.argsort(temp[:,2])[::-1]
        temp = temp[ind]

        # find outlier hidden neurons
        top_num = len(self.outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False))
        num_neuron = top_num
        if self.verbose:
            print('significant neuron: {}'.format(num_neuron))
        cur_top = list(temp[0: (num_neuron - 1)][:, [0, 1]])

        top_list = []
        top_neuron = []
        # compare with all other classes
        for cmp_class in self.classes:
            if cmp_class == cur_class:
                top_list.append(0)
                top_neuron.append(np.array([0] * num_neuron))
                continue
            temp = hidden_test[:, [0, 1, (cmp_class + 2)]]
            cmp_top_num = len(self.outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False))
            ind = np.argsort(temp[:,2])[::-1]
            temp = temp[ind]
            cmp_top = list(temp[0: (num_neuron - 1)][:, [0, 1]])
            temp = np.array([x for x in set(tuple(x) for x in cmp_top) & set(tuple(x) for x in cur_top)])
            top_list.append(len(temp))
            top_neuron.append(temp)

        # top_list x10
        # find outlier
        #flag_list = self.outlier_detection(top_list, top_num, cur_class)

        # top_list: number of intersected neurons (10,)
        # top_neuron: layer and index of intersected neurons    ((2, n) x 10)
        return list(np.array(top_list) / top_num), top_neuron

        pass

    def find_outstanding_neuron(self,  cur_class, prefix=""):
        '''
        find outstanding neurons for cur_class
        '''
        '''
        hidden_test = []
        for cur_layer in self.layer:
            #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)
        '''
        hidden_test = np.loadtxt(RESULT_DIR + prefix + "test_pre0_"  + "c" + str(cur_class) + "_layer_13" + ".txt")
        #'''
        hidden_test = np.array(hidden_test)

        # find outlier hidden neurons for all class embedding
        top_num = []
        # compare with all other classes
        for cmp_class in self.classes:
            temp = hidden_test[:, [0, (cmp_class + 1)]]
            ind = np.argsort(temp[:,1])[::-1]
            temp = temp[ind]
            cmp_top = self.outlier_detection_overfit(temp[:, (1)], max(temp[:, (1)]), verbose=False)
            top_num.append((cmp_top))

        return top_num

    def locate_candidate_neuron(self, base_class, target_class):
        '''
        find outstanding neurons for target class
        '''
        hidden_test = []
        for cur_layer in self.layer:
            #hidden_test_ = np.loadtxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(cur_class) + "_layer_" + str(cur_layer) + ".txt")
            hidden_test_ = np.loadtxt(RESULT_DIR + "test_pre0_" + "c" + str(base_class) + "_layer_" + str(cur_layer) + ".txt")
            #l = np.ones(len(hidden_test_)) * cur_layer
            hidden_test_ = np.insert(np.array(hidden_test_), 0, cur_layer, axis=1)
            hidden_test = hidden_test + list(hidden_test_)
        hidden_test = np.array(hidden_test)

        # find outlier hidden neurons for target class embedding
        temp = hidden_test[:, [0, 1, (target_class + 2)]]
        ind = np.argsort(temp[:,2])[::-1]
        temp = temp[ind]
        top = self.outlier_detection(temp[:, 2], max(temp[:, 2]), verbose=False)
        ret = temp[0: (len(top) - 1)][:, [0, 1]]
        return ret


    def detect_common_outstanding_neuron(self,  tops):
        '''
        find common important neurons for each class with samples from current class
        @param tops: list of outstanding neurons for each class
        '''
        top_list = []
        top_neuron = []
        # compare with all other classes
        for base_class in self.classes:
            for cur_class in self.classes:
                if cur_class <= base_class:
                    continue
                temp = np.array([x for x in set(tuple(x) for x in tops[base_class]) & set(tuple(x) for x in tops[cur_class])])
                top_list.append(len(temp))
                top_neuron.append(temp)

        flag_list = self.outlier_detection(top_list, max(top_list))

        # top_list: number of intersected neurons (10,)
        # top_neuron: layer and index of intersected neurons    ((2, n) x 10)

        return flag_list

    def find_common_neuron(self, cmv_top, tops):
        '''
        find common important neurons for cmv top and base_top
        @param tops: activated neurons @base class sample
               cmv_top: important neurons for this attack from base to target
        '''

        temp = np.array([x for x in set(tuple(x) for x in tops) & set(tuple(x) for x in cmv_top)])
        return temp


    def hidden_permutation_test_all(self, gen, pre_class, prefix=''):
        # calculate the importance of each hidden neuron
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            self.mini_batch = 3
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

                # randomize each hidden
                perm_predict = []
                for i in range(0, len(out_hidden_[0])):
                    perm_predict_neu = []
                    out_hidden_ = out_hidden.reshape(out_hidden.shape[0], -1).copy()
                    for j in range (0, self.random_sample):
                        #hidden_random = np.random.uniform(low=min[i], high=max[i], size=len(out_hidden)).transpose()
                        hidden_do = np.zeros(shape=out_hidden_[:,i].shape)
                        out_hidden_[:, i] = hidden_do
                        sample_pre = partial_model2.predict(out_hidden_.reshape(out_hidden.shape)) # 8k x 32
                        perm_predict_neu.append(sample_pre)

                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    perm_predict_neu = np.abs(ori_pre - perm_predict_neu)
                    perm_predict_neu = np.mean(np.array(perm_predict_neu), axis=0)
                    to_add = []
                    to_add.append(int(i))
                    for class_n in self.classes:
                        to_add.append(perm_predict_neu[class_n])
                    perm_predict.append(np.array(to_add))
                perm_predict_avg.append(perm_predict)
            # average of all baches
            perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            #ind = np.argsort(perm_predict_avg[:,1])[::-1]
            #perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + prefix + "test_pre0_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return np.array(out)

    def hidden_act_test_all(self, gen, pre_class, prefix=''):
        # calculate the importance of each hidden neuron
        out = []
        for cur_layer in self.layer:
            model_copy = keras.models.clone_model(self.model)
            model_copy.set_weights(self.model.get_weights())

            # split to current layer
            partial_model1, partial_model2 = self.split_keras_model(model_copy, cur_layer + 1)

            self.mini_batch = 3
            perm_predict_avg = []
            for idx in range(self.mini_batch):
                X_batch, Y_batch = gen.next()
                out_hidden = partial_model1.predict(X_batch)    # 32 x 16 x 16 x 32
                ori_pre = partial_model2.predict(out_hidden)    # 32 x 10

                predict = self.model.predict(X_batch) # 32 x 10

                out_hidden_ = copy.deepcopy(out_hidden.reshape(out_hidden.shape[0], -1))

            # average of all baches
            perm_predict_avg = np.mean(np.array(out_hidden_), axis=0)

            #now perm_predict contains predic value of all permutated hidden neuron at current layer
            perm_predict_avg = np.array(perm_predict_avg)
            out.append(perm_predict_avg)
            #ind = np.argsort(perm_predict_avg[:,1])[::-1]
            #perm_predict_avg = perm_predict_avg[ind]
            np.savetxt(RESULT_DIR + prefix + "test_act_" + "c" + str(pre_class) + "_layer_" + str(cur_layer) + ".txt", perm_predict_avg, fmt="%s")
            #out.append(perm_predict_avg)

        return np.array(out)


    # class embedding
    def hidden_ce_test_all(self, gen, pre_class):
        # calculate the importance of each hidden neuron
        out = []
        cur_layer = 15

        model_copy = keras.models.clone_model(self.model)
        model_copy.set_weights(self.model.get_weights())

        self.mini_batch = 3
        perm_predict_avg = []
        for idx in range(self.mini_batch):
            X_batch, Y_batch = gen.next()
            ce = model_copy.predict(X_batch)    # 32 x 16 x 16 x 32
            perm_predict_avg = perm_predict_avg + list(ce)
        # average of all baches
        perm_predict_avg = np.mean(np.array(perm_predict_avg), axis=0)

        #now perm_predict contains predic value of all permutated hidden neuron at current layer
        perm_predict_avg = np.array(perm_predict_avg)
        out.append(perm_predict_avg)
        #ind = np.argsort(perm_predict_avg[:,1])[::-1]
        #perm_predict_avg = perm_predict_avg[ind]
        np.savetxt(RESULT_DIR + "test_ce_" + "c" + str(pre_class) + ".txt", perm_predict_avg, fmt="%s")
        #out.append(perm_predict_avg)

        #out: ce of cur_class
        return np.array(out)


    def outlier_detection(self, cmp_list, max_val, verbose=False):
        cmp_list = list(np.array(cmp_list) / max_val)
        consistency_constant = 1.4826  # if normal distribution
        median = np.median(cmp_list)
        mad = consistency_constant * np.median(np.abs(cmp_list - median))   #median of the deviation
        min_mad = np.abs(np.min(cmp_list) - median) / mad

        #print('median: %f, MAD: %f' % (median, mad))
        #print('anomaly index: %f' % min_mad)

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > 2:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
        return flag_list
        pass

    def outlier_detection_overfit(self, cmp_list, max_val, verbose=True):
        #'''
        mean = np.mean(np.array(cmp_list))
        standard_deviation = np.std(np.array(cmp_list))
        distance_from_mean = abs(np.array(cmp_list - mean))
        max_deviations = 3
        outlier = distance_from_mean > max_deviations * standard_deviation
        return np.count_nonzero(outlier == True)
        #'''
        cmp_list = list(np.array(cmp_list) / max_val)
        consistency_constant = 1.4826  # if normal distribution
        median = np.median(cmp_list)
        mad = consistency_constant * np.median(np.abs(cmp_list - median))   #median of the deviation
        min_mad = np.abs(np.min(cmp_list) - median) / mad

        #print('median: %f, MAD: %f' % (median, mad))
        #print('anomaly index: %f' % min_mad)
        debug_list = np.abs(cmp_list - median) / mad
        #print(debug_list)
        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > 2:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
        return len(flag_list)
        pass


    def plot_hidden(self, _cmv_rank, _test_rank, normalise=True):
        # plot the permutation of cmv img and test imgs
        cmv_rank = copy.deepcopy(_cmv_rank)
        test_rank = copy.deepcopy(_test_rank)
        plt_row = 2
        #for i in range (0, len(self.layer)):
        #    if len(self.do_neuron[i]) > plt_row:
        #        plt_row = len(self.do_neuron[i])
        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)
        #fig.tight_layout()

        col = 0
        #self.layer = [2]
        for do_layer in self.layer:
            row = 0
            # plot ACE
            ax[row, col].set_title('Layer_' + str(do_layer))
            #ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            # Baseline is np.mean(expectation_do_x)
            if normalise:
                cmv_rank[col][:,1] = cmv_rank[col][:,1] / np.max(cmv_rank[col][:,1])

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_cmv', color='b')
            ax[row, col].legend()

            row = row + 1

            # plot ACE
            #ax[row, col].set_title('Layer_' + str(do_layer))
            ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            # Baseline is np.mean(expectation_do_x)
            if normalise:
                test_rank[col][:,1] = test_rank[col][:,1] / np.max(test_rank[col][:,1])
            ax[row, col].scatter(test_rank[col][:,0].astype(int), test_rank[col][:,1], label = str(do_layer) + '_test', color='b')
            ax[row, col].legend()

            #if row == len(self.do_neuron[col]):
            #    for off in range(row, plt_row):
            #        ax[off, col].set_axis_off()
            #ie_ave.append(ie_ave_l)
            col = col + 1
        if normalise:
            plt.savefig(RESULT_DIR + "plt_n_c" + str(self.current_class) + ".png")
        else:
            plt.savefig(RESULT_DIR + "plt_c" + str(self.current_class) + ".png")
        plt.show()

    def plot_multiple(self, _rank, name, normalise=False, save_n=""):
        # plot the permutation of cmv img and test imgs
        plt_row = len(_rank)

        rank = []
        for _rank_i in _rank:
            rank.append(copy.deepcopy(_rank_i))

        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)

        col = 0
        for do_layer in self.layer:
            for row in range(0, plt_row):
                # plot ACE
                if row == 0:
                    ax[row, col].set_title('Layer_' + str(do_layer))
                    #ax[row, col].set_xlabel('neuron index')
                    #ax[row, col].set_ylabel('delta y')

                if row == (plt_row - 1):
                    #ax[row, col].set_title('Layer_' + str(do_layer))
                    ax[row, col].set_xlabel('neuron index')

                ax[row, col].set_ylabel(name[row])

                # Baseline is np.mean(expectation_do_x)
                if normalise:
                    rank[row][col][:,1] = rank[row][col][:,1] / np.max(rank[row][col][:,1])

                ax[row, col].scatter(rank[row][col][:,0].astype(int), rank[row][col][:,1], label = str(do_layer) + '_cmv', color='b')
                ax[row, col].legend()

            col = col + 1
        if normalise:
            plt.savefig(RESULT_DIR + "plt_n_c" + str(self.current_class) + save_n + ".png")
        else:
            plt.savefig(RESULT_DIR + "plt_c" + str(self.current_class) + save_n + ".png")
        #plt.show()


    def plot_diff(self, _cmv_rank, _test_rank, normalise=True):
        # plot the permutation of cmv img and test imgs
        cmv_rank = copy.deepcopy(_cmv_rank)
        test_rank = copy.deepcopy(_test_rank)
        plt_row = 2
        #for i in range (0, len(self.layer)):
        #    if len(self.do_neuron[i]) > plt_row:
        #        plt_row = len(self.do_neuron[i])
        plt_col = len(self.layer)
        fig, ax = plt.subplots(plt_row, plt_col, figsize=(7*plt_col, 5*plt_row), sharex=False, sharey=True)
        #fig.tight_layout()

        col = 0
        #self.layer = [2]
        for do_layer in self.layer:
            row = 0
            # plot ACE
            #ax[row, col].set_title('Layer_' + str(do_layer))
            ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            hidden_diff = np.abs(cmv_rank[col][:,1] - test_rank[col][:,1])
            cmv_rank[col][:,1] = hidden_diff

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_diff', color='b')
            ax[row, col].legend()

            row = row + 1

            ax[row, col].set_title('Layer_' + str(do_layer))
            #ax[row, col].set_xlabel('neuron index')
            ax[row, col].set_ylabel('delta y')

            cmv_rank[col][:,1] = cmv_rank[col][:,1] / np.max(cmv_rank[col][:,1])

            test_rank[col][:,1] = test_rank[col][:,1] / np.max(test_rank[col][:,1])

            hidden_diff = np.abs(cmv_rank[col][:,1] - test_rank[col][:,1])
            cmv_rank[col][:,1] = hidden_diff

            ax[row, col].scatter(cmv_rank[col][:,0].astype(int), cmv_rank[col][:,1], label = str(do_layer) + '_diffn', color='b')
            ax[row, col].legend()

            col = col + 1
        plt.savefig(RESULT_DIR + "plt_diff_c" + str(self.current_class) + ".png")
        plt.show()


def load_dataset_class(data_file=('%s/%s' % (DATA_DIR, DATA_FILE)), cur_class=0):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32") / 255
    x_test = X_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #print("x_train shape:", x_train.shape)
    #print(x_train.shape[0], "train samples")
    #print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tensorflow.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tensorflow.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    x_out = []
    y_out = []
    for i in range (0, len(x_test)):
        if np.argmax(y_test[i], axis=0) == cur_class:
            x_out.append(x_test[i])
            y_out.append(y_test[i])

    return np.array(x_out), np.array(y_out)

def build_data_loader(X, Y):

    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)

    return generator

def build_data_loader_aug(X, Y):

    datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE)

    return generator

