import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def OCSVM(X_train, X_test, Y_test):
    from sklearn.svm import OneClassSVM

    ocSVM = OneClassSVM()

    ocSVM.fit(X_train)

    pred = ocSVM.predict(X_test)

    pred[pred==1] = 0
    pred[pred==-1] = 1

    acc = np.sum(pred == Y_test)/X_test.shape[0]
    print("ocSVM:" + str(acc))
    return (acc*100)

def elm(X_train, X_test, Y_train, Y_test, ds_anom, ds_norm):
    # CHECK : Constants
    omega = 1.

    class ELM(object):
        def __init__(self, sess, batch_size, input_len, hidden_num, output_len, W, b):
            '''
            Args:
            sess : TensorFlow session.
            batch_size : The batch size (N)
            input_len : The length of input. (L)
            hidden_num : The number of hidden node. (K)
            output_len : The length of output. (O)
            W : randomly initialized weights
            b : randomly initialized bias
            '''
        
            self._sess = sess 
            self._batch_size = batch_size
            self._input_len = input_len
            self._hidden_num = hidden_num
            self._output_len = output_len 

            # for train
            self._x0 = tf.placeholder(tf.float32, [self._batch_size, self._input_len])
            self._t0 = tf.placeholder(tf.float32, [self._batch_size, self._output_len])

            # for test
            self._x1 = tf.placeholder(tf.float32, [None, self._input_len])
            self._t1 = tf.placeholder(tf.float32, [None, self._output_len])

    #         self._W = tf.Variable(
    #           tf.random_normal([self._input_len, self._hidden_num]),
    #           trainable=False, dtype=tf.float32)
    #         self._b = tf.Variable(
    #           tf.random_normal([self._hidden_num]),
    #           trainable=False, dtype=tf.float32)

            ## Wts initialisation
            self._W = W
            self._b = b
            
            self._beta = tf.Variable(
            tf.zeros([self._hidden_num, self._output_len]),
            trainable=False, dtype=tf.float32)
            self._var_list = [self._W, self._b, self._beta]

            self.H0 = tf.matmul(self._x0, self._W) + self._b # N x L
            self.H0_T = tf.transpose(self.H0)

            self.H1 = tf.matmul(self._x1, self._W) + self._b # N x L
            self.H1_T = tf.transpose(self.H1)

            # beta analytic solution : self._beta_s (K x O)
            if self._input_len < self._hidden_num: # L < K
                identity = tf.constant(np.identity(self._hidden_num), dtype=tf.float32)
                self._beta_s = tf.matmul(tf.matmul(tf.matrix_inverse(
                    tf.matmul(self.H0_T, self.H0) + identity/omega), 
                    self.H0_T), self._t0)
            # _beta_s = (H_T*H + I/om)^(-1)*H_T*T
            else:
                identity = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
                self._beta_s = tf.matmul(tf.matmul(self.H0_T, tf.matrix_inverse(
                    tf.matmul(self.H0, self.H0_T)+identity/omega)), self._t0)
            # _beta_s = H_T*(H*H_T + I/om)^(-1)*T

            self._assign_beta = self._beta.assign(self._beta_s)
            self._fx0 = tf.matmul(self.H0, self._beta)
            self._fx1 = tf.matmul(self.H1, self._beta)

            self._cost = tf.reduce_mean(tf.cast(tf.losses.mean_squared_error(labels=self._t0, predictions=self._fx0), tf.float32))
                                            
            self._init = False
            self._feed = False

            # Cost for every sample point
    #         self._correct_prediction = tf.equal(tf.argmax(self._fx1,1), tf.argmax(self._t1,1))
    #         self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))
            self._testcost = tf.cast(tf.losses.mean_squared_error(labels=self._t1, predictions=self._fx1), tf.float32)


        def feed(self, x, t):
            '''
            Args :
            x : input array (N x L)
            t : label array (N x O)
            '''

            if not self._init : self.init()
            self._sess.run(self._assign_beta, {self._x0:x, self._t0:t})
    #         print(self._sess.run(self._cost, {self._x0:x, self._t0:t}))
            
            self._feed = True

        def init(self):
            self._sess.run(tf.initialize_variables(self._var_list))
            self._init = True

        def test(self, x, t=None):
            if not self._feed : exit("Not feed-forward trained")
            if t is not None :
    #             print("Accuracy: {:.9f}".format(self._sess.run(self._accuracy, {self._x1:x, self._t1:t})))
                return self._sess.run(self._testcost, {self._x1:x, self._t1:t})
                
            else :
                return self._sess.run(self._fx1, {self._x1:x})


    # ## Initializing Parameters

    # In[55]:


    import tensorflow as tf


    # In[56]:


    sess = tf.Session()
    batch_size = X_train.shape[0]
    hidden_num = 150
    input_len = X_train.shape[1]
    print("batch_size : {}".format(batch_size))
    print("hidden_num : {}".format(hidden_num))
    print(input_len)
    W = tf.Variable(
    tf.random_normal([input_len, hidden_num]),
    trainable=False, dtype=tf.float32)
    b = tf.Variable(
    tf.random_normal([hidden_num]),
    trainable=False, dtype=tf.float32)


    # ## Initializing list of W and b

    # In[57]:


    init_list = []
    for i in range(10):
            init_list.append((tf.Variable(tf.random_normal([input_len, hidden_num],seed=i),trainable=False, dtype=tf.float32),tf.Variable(tf.random_normal([hidden_num], seed=i),trainable=False, dtype=tf.float32)))
    


    # ## Accuracy Function

    # In[58]:


    def accuracy(anom_pred):
        cnt = 0
        for pt in anom_pred:
            if pt[1]>=ds_norm.shape[0]-X_train.shape[0] and pt[1]<X_test.shape[0]:
                cnt+=1
        return (cnt/float(ds_anom.shape[0]))*100


    # ## Evaluation

    # In[ ]:



    results = {}
    itr = 0
    best_W = tf.Variable(tf.zeros([input_len,hidden_num]))
    best_b = tf.Variable(tf.zeros([hidden_num]))
    best_acc = 0.0
    best_acc_idx = 0

    for W,b in init_list:
        
        ## feed W,b from list and evaluate error and accuracy corresponding to them
        elm = ELM(sess, batch_size, input_len, hidden_num, input_len, W, b)
        train_x, train_y = (X_train[:batch_size], X_train[:batch_size])

        elm.feed(train_x, train_y)
        
        ## error list
        err = []
        for idx,test_pt in enumerate(X_test):
            x = test_pt.reshape(1,-1)
            err.append((elm.test(x, x), idx))
        
        err.sort(reverse=True)
        
        anom_pred = err[:ds_anom.shape[0]]
        
        acc = accuracy(anom_pred) 
        
        results[itr] = [(err,acc)]
        itr += 1
        if acc>best_acc:
            best_W = W
            best_b = b
            best_acc = acc
            best_acc_idx = itr-1
        


    err = results[best_acc_idx][0][0]

    err_array = np.array(err)


    W_final = best_W.eval(session=sess)
    b_final = best_b.eval(session=sess)

    print("ELM: "+ str(best_acc))
    return best_acc

def knn(X_train, X_test, Y_train, Y_test):
    from pyod.models.knn import KNN
    model = KNN()
    model.fit(X_train)
    pred = model.predict(X_test)
    acc = np.sum(pred == Y_test)/X_test.shape[0]
    print(acc)
    return (acc*100)


def mean_knn(X_train, X_test, Y_train, Y_test):
    from pyod.models.knn import KNN
    model = KNN(method='mean')
    model.fit(X_train)
    pred = model.predict(X_test)
    acc = np.sum(pred == Y_test)/X_test.shape[0]
    print(acc)
    return (acc*100)

def median_knn(X_train, X_test, Y_train, Y_test):
    from pyod.models.knn import KNN
    model = KNN(method='median')
    model.fit(X_train)
    pred = model.predict(X_test)
    acc = np.sum(pred == Y_test)/X_test.shape[0]
    print(acc)
    return (acc*100)

def pca(X_train, X_test, Y_train, Y_test):
    from pyod.models.pca import PCA
    model = PCA()
    model.fit(X_train)
    pred = model.predict(X_test)
    acc = np.sum(pred == Y_test)/X_test.shape[0]
    print(acc)
    return (acc*100)

def iforest(X_train, X_test, Y_train, Y_test):
    from pyod.models.iforest import IForest
    model = IForest(random_state=0)
    model.fit(X_train)
    pred = model.predict(X_test)
    acc = np.sum(pred == Y_test)/X_test.shape[0]
    print(acc)
    return (acc*100)

def feature_bagging(X_train, X_test, Y_train, Y_test):
    from pyod.models.feature_bagging import FeatureBagging
    model = FeatureBagging(random_state=1)
    model.fit(X_train)
    pred = model.predict(X_test)
    acc = np.sum(pred == Y_test)/X_test.shape[0]
    print(acc)
    return (acc*100)



## Data preparation for clustering algorithms



# ## DBSCAN

def DBSCAN(X_train, X_test, Y_train, Y_test, dsStatus):
    samples = X_train.shape[0]+X_test.shape[0]
    X = np.zeros((samples,X_train.shape[1]))
    X[:X_train.shape[0], :] = X_train[:, :]
    X[X_train.shape[0]:, :] = X_test[:, :]
    X.shape

    Y = np.zeros(samples,)
    Y[:Y_train.shape[0]] = Y_train
    Y[Y_train.shape[0]:] = Y_test
    Y.shape
    from sklearn.cluster import DBSCAN
    if dsStatus.get() == 'page_blocks':
        e = 50.0
    elif dsStatus.get() == 'cancer':
        e = 1.0
    elif dsStatus.get() == 'lymphography':
        e=20.0
    else:
        e=3.0
    dbscan = DBSCAN(eps=e)
    pred = dbscan.fit_predict(X)
    pred

    acc = np.sum(pred==Y)/Y.shape[0]
    print("DBSCAN:" + str(acc))
    return (acc*100)


# ## LOF
def Lof(X_train, X_test, Y_train, Y_test):
    samples = X_train.shape[0]+X_test.shape[0]
    X = np.zeros((samples,X_train.shape[1]))
    X[:X_train.shape[0], :] = X_train[:, :]
    X[X_train.shape[0]:, :] = X_test[:, :]
    X.shape

    Y = np.zeros(samples,)
    Y[:Y_train.shape[0]] = Y_train
    Y[Y_train.shape[0]:] = Y_test
    Y.shape
    from sklearn.neighbors import LocalOutlierFactor


    lof = LocalOutlierFactor()

    pred = lof.fit_predict(X)

    np.unique(pred, return_counts=True)


    pred[pred==1] = 0
    pred[pred == -1] = 1
 

    acc = np.sum(pred==Y)/Y.shape[0]
    print("LOF:" + str(acc))
    return (acc*100)
