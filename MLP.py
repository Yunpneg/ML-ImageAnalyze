import numpy as np
import matplotlib.pyplot as mtb
import sklearn.metrics as sk


class Mlp:

    # create data
    @staticmethod
    def generate_dataset():
        # design the range of data for Class1 and Class2
        _class1_x1 = np.random.uniform(3.0, 6.0, 500)
        _class1_x2 = np.random.uniform(3.0, 4.0, 500)
        _class2_x1 = np.random.uniform(2.0, 4.0, 500)
        _class2_x2 = np.random.uniform(-4.0, 1.0, 500)
        # vertical joint data in x1-axis and data in x2-axis, get a 2*500 matrix
        _class1_data = np.hstack((_class1_x1, _class1_x2))
        _class2_data = np.hstack((_class2_x1, _class2_x2))
        # horizontal joint two matrix, get a 2*1000 matrix, each column is a point
        _training_data = np.hstack((_class1_data, _class2_data))
        # Rotation, make Θ as 45 degree
        _degree = 45
        _R = [[np.cos(_degree*np.pi/180), -np.sin(_degree*np.pi/180)], [np.sin(_degree*np.pi/180),
                                                                        np.cos(_degree*np.pi/180)]]
        # Rotating the original matrix and transpose it to get a 1000*2 matrix, each row is a point
        _training_data = np.dot(_R, _training_data).transpose()
        # design the data for class3 and class4
        _class3_mean3 = np.array([-2, -3])
        _class3_cov3 = np.array([[0.5, 0], [0, 3]])
        _class4_mean4 = np.array([-4, -1])
        _class4_cov4 = np.array([[3, 0.5], [0.5, 0.5]])
        _class3_data = np.random.multivariate_normal(_class3_mean3, _class3_cov3, size=500)
        _class4_data = np.random.multivariate_normal(_class4_mean4, _class4_cov4, size=500)
        # add class name
        _array1 = np.ones((500, 1))
        _array2 = _array1*2
        _array3 = _array1*3
        _array4 = _array1*4
        _arr = np.vstack((_array1, _array2))
        # adding the third column to the data matrix as the class name
        _training_data = np.hstack((_training_data, _arr))
        _class3_data = np.hstack((_class3_data, _array3))
        _class4_data = np.hstack((_class4_data, _array4))
        # aggregate and shuffle the three types of data
        _whole_data = np.vstack((_training_data, _class3_data, _class4_data))
        np.random.shuffle(_whole_data)
        _training_data = _whole_data[0:1000, :]
        _validation_data = _whole_data[1000:1500, :]
        _test_data = _whole_data[1500:2000, :]
        return _training_data, _validation_data, _test_data

    # classify training data
    @staticmethod
    def classify_mlp(w, h, x):
        '''
        :param w: weights
        :param h: the number of hidden neurons
        :param x: the input features
        :return: predicted value, is a 1000*4 matrix
        '''
        # create a ones matrix according to the column number of x as bias
        _x0 = np.ones((len(x[:, 0]), 1))
        _classMatrix = np.hstack((_x0, x[:, 0:2]))
        _classMatrix = Mlp.sigmoid(np.dot(_classMatrix, w[0]))
        _classMatrix = np.hstack((_x0, _classMatrix))
        c = Mlp.sigmoid(np.dot(_classMatrix, w[1]))
        return c

    # create updated weights function
    @staticmethod
    def train_mlp(w, h, eta, D):
        '''
        :param w: weights
        :param h: the number of hidden layers
        :param eta: learning rate
        :param D: the input matrix
        :return: updated weights
        '''
        _t = Mlp.expect_class(D)  # the desire value of input data
        _x0 = np.array([1])  # bias
        _classMatrix = D[:, 0:2]
        weights = w
        # iterate all input points
        for i in range(0, len(D[:, 0])):
            # compute the output of hidden neurons
            _input_hidden = np.hstack((_x0, _classMatrix[i]))
            _output_hidden = Mlp.sigmoid(np.dot(_input_hidden, weights[0]))
            # adding bias to the output of hidden nodes
            _output_hidden = np.hstack((_x0, _output_hidden))
            # get the predicted result
            _y = Mlp.sigmoid(np.dot(_output_hidden, weights[1]))
            # the error item of output layer
            _outputErrorItem = (_y - _t[i])*_y*(1 - _y)
            # the error item of hidden layer
            _hiddenErrorItem = _output_hidden * (1 - _output_hidden) * np.dot(weights[1], _outputErrorItem)
            # let the error item of hidden layer be a (h+1)*1 matrix
            _hiddenErrorItem = _hiddenErrorItem.reshape(h+1, 1)
            # get weights[1], which is the updated weight of output layer
            _output_hidden = _output_hidden.reshape(h+1, 1)
            _outputErrorItem = np.copy(_outputErrorItem)
            _outputErrorItem = _outputErrorItem.reshape(1, 4)
            weights[1] = weights[1] - eta*np.dot(_output_hidden, _outputErrorItem)
            # remove the bias item of hidden error item
            _hiddenErrorItem = _hiddenErrorItem[1:, :]
            _input_hidden = _input_hidden.reshape(3, 1)
            _hiddenErrorItem = _hiddenErrorItem.reshape(1, h)
            # get weights[0], which is the updated weight of hidden layer
            weights[0] = weights[0] - eta*np.dot(_input_hidden, _hiddenErrorItem)
        return weights

    # evaluate function
    @staticmethod
    def evaluate_mlp(w, h, D):
        '''
        Get the sum of squared error by desired value and predicted value of the input data
        :param w: weights matrix
        :param h: the number of hidden neurons
        :param D: Validation data, is a 500*3 matrix
        :return: the number of error points and the sum of error square
        '''
        _expectClass = Mlp.expect_class(D)
        _computingClass = Mlp.classify_mlp(w, h, D)
        _predictClass = np.zeros_like(_computingClass)
        _predictClass[np.arange(len(_computingClass)), _computingClass.argmax(1)] = 1
        _errorData = Mlp.normal_data(_expectClass, _computingClass)
        # get the sum of error square according to 0.5*sum(t - y)^2
        e = np.sum(np.square(_errorData), axis=0)/2
        _error = np.equal(_predictClass, _expectClass)
        # get the number of error points, through comparing predicted value with expected value
        _error_item = 0
        for i in range(0, len(D[:, 0])):
            if np.all(_error[i] == True):
                continue
            else:
                _error_item += 1
        return _error_item, e

    # create sigmoid function
    @staticmethod
    def sigmoid(x):
        return 1.0/(1 + np.exp(-x))

    # get the desired value of input data
    @staticmethod
    def expect_class(D):
        '''
        Set the desired value of input data as n*4 matrix
        :param D: the validation data
        :return:  the desired class
        '''
        _expectClass = D[:, 2:] - 1
        _expect_result = np.zeros((len(D[:, 0]), 4))
        for i in range(0, len(D[:, 0])):
            if _expectClass[i] == 0:
                _expect_result[i] = [1, 0, 0, 0]
            elif _expectClass[i] == 1:
                _expect_result[i] = [0, 1, 0, 0]
            elif _expectClass[i] == 2:
                _expect_result[i] = [0, 0, 1, 0]
            else:
                _expect_result[i] = [0, 0, 0, 1]
        return _expect_result

    # normal the output data
    @staticmethod
    def normal_data(t, y):
       '''
       a part of normalization
       :param t: the desired class
       :param y: the class through computing
       :return:  the normal data
       '''
       _errorData= np.square(t - y)
       _errorData = np.sum(_errorData, axis=1)
       _errorData = np.sqrt(_errorData)
       return _errorData

    # main function
    @staticmethod
    def main():
        '''
        Implement the three functions and data
        :return: various data and plot
        '''
        # # get the whole data and saving it to a txt file
        # _training_data, _validation_data, _test_data = Mlp.generate_dataset()
        # _whole_data = np.hstack(_training_data, _validation_data, _test_data)
        # np.savetxt('/home/chen/Chen_Yunpeng_data.txt', _whole_data)

        # read the data from file
        _whole_data = np.loadtxt('/home/chen/Chen_Yunpeng_data.txt')
        _training_data = _whole_data[0:1000, :]
        _validation_data = _whole_data[1000:1500, :]
        _test_data = _whole_data[1500:2000, :]

        # # plot the whole data
        # mtb.scatter(_training_data[:500, 0], _training_data[:500, 1], c='r', marker='+')
        # mtb.scatter(_training_data[500:, 0], _training_data[500:, 1], s=10, c='b', marker='.')
        # mtb.scatter(_validation_data[:, 0], _validation_data[:, 1], c='g', marker='x')
        # mtb.scatter(_test_data[:, 0], _test_data[:, 1], s=10, c='y', marker='*')
        # mtb.show()

        # pass the value of parameters
        h = 700
        eta = 0.01
        w1 = np.random.uniform(-0.5, 0.5, (3, h))
        w2 = np.random.uniform(-0.5, 0.5, (h + 1, 4))
        w = np.array([w1, w2])
        _weights = w
        _error_1 = np.zeros((100, 2))
        _error_2 = np.zeros((100, 2))
        for i in range(0, len(_training_data[:, 0]), 10):
            weights = Mlp.train_mlp(_weights, h, eta, _training_data[i:10 + i, :])
            e_1 = Mlp.evaluate_mlp(weights, h, _validation_data)
            e_2 = Mlp.evaluate_mlp(weights, h, _training_data)
            i = int(i / 10)
            _error_1[i] = [i, e_1[1]]
            _error_2[i] = [i, e_2[1]]
            _weights = weights

        # # get the confusion matrix
        # c = Mlp.classify_mlp(_weights, h, _training_data)
        # c = np.argmax(c, axis=1) + 1  # we assume that the first column of predicted value is class one
        # confusion_m = sk.confusion_matrix(_training_data[:, 2:], c)
        # mtb.matshow(confusion_m)
        # mtb.show()

        # # evaluate the error on test set
        # e = Mlp.evaluate_mlp(_weights, h, _test_data)
        # print(e[0])

        # #  training_data is red line; validation data is blue line
        # mtb.plot(_error_1[:, 0], _error_1[:, 1], 'r-')
        # mtb.plot(_error_2[:, 0], _error_2[:, 1], 'b--')
        # mtb.show()

        # plot the test set
        _test_class = Mlp.classify_mlp(_weights, h, _test_data)
        _test_class = np.argmax(_test_class, axis=1) + 1
        _test_class = _test_class.reshape(500, 1)
        _test_classify = np.hstack((_test_data[:, 0:2], _test_class))
        for i in range(0, len(_test_class[:, 0])):
            if _test_class[i] == 1:
                color = 'c'
                marker = '+'
            elif _test_class[i] == 2:
                color = 'k'
                marker = '.'
            elif _test_class[i] == 3:
                color = 'm'
                marker = '*'
            else:
                color = 'y'
                marker = 'o'
            mtb.scatter(_test_classify[i, 0], _test_classify[i, 1], c=color, marker=marker)
        mtb.show()
        e = Mlp.evaluate_mlp(_weights, h, _test_data)
        print(e[0])


# operate the main function
Mlp.main()
