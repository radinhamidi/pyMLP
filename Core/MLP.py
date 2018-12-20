from Core.util import *

class Activation_Mode(Enum):
    Sigmoid = auto()
    Tanh = auto()
    ReLu = auto()
    SoftMax = auto()


class Error_Mode(Enum):
    MSE = auto()
    Cross_Entropy = auto()


class Steepest_Descent(Enum):
    On = auto()
    Off = auto()


class Gradient_Mode(Enum):
    Stochastic = auto()
    Batch = auto()


class MLP:
    def __init__(self, hidden_layers, n_classes, n_features, activation_mode:Activation_Mode, error_mode:Error_Mode, steepest_descent:Steepest_Descent, gradient_mode:Gradient_Mode, weight_damp_factor=1):
        self.__hidden_layers_size = len(hidden_layers)
        self.__hidden_layers = hidden_layers
        self.__n_classes = n_classes
        self.__n_features = n_features
        self.__activation_mode = activation_mode
        self.__error_mode = error_mode
        self.__weight_damp_factor = weight_damp_factor
        self.__steepest_descent = steepest_descent
        self.__gradient_mode = gradient_mode
        self.init_weights()


    def init_weights(self):
        self.__weights = []
        self.__n_neurons_layers = [self.__n_features] + self.__hidden_layers + [self.__n_classes]
        for i in range(self.__hidden_layers_size+1):
            self.__weights.append(np.random.uniform(-1, 1, (self.__n_neurons_layers[i]+1, self.__n_neurons_layers[i+1])) * self.__weight_damp_factor)
        self.__delta_weights = np.zeros_like(np.asarray(self.__weights))


    def feed_foreward(self, s):
        foreward_network = [np.asarray(s)]
        for weight in self.__weights:
            foreward_network[-1] = np.concatenate(([1],foreward_network[-1]))
            if self.__activation_mode == Activation_Mode.Sigmoid: foreward_network.append(expit(np.dot(foreward_network[-1], weight)))
            elif self.__activation_mode == Activation_Mode.Tanh: foreward_network.append(np.tanh(np.dot(foreward_network[-1], weight)))
        self.__feedForeward = foreward_network
        return foreward_network


    def back_propagate(self, y_layers, desired_output):
        delta = np.asarray(y_layers[-1] - desired_output)
        if self.__error_mode == Error_Mode.MSE:
            E = self.cost_mse(delta)
        elif self.__error_mode == Error_Mode.Cross_Entropy:
            E = self.cost_cross_entropy(y_layers[-1], desired_output)
        deltas = [delta]
        errors = []
        gradients = []

        deltas.append(np.dot(delta,self.__weights[-1].T))
        for w in self.__weights[-2:0:-1]:
            deltas.append(np.dot(deltas[-1][1:], w.T))

        for j, y in enumerate(y_layers[-1:0:-1]):
            if self.__error_mode == Error_Mode.MSE:
                if self.__activation_mode == Activation_Mode.Sigmoid: errors.append(self.dSigmoid(y) * deltas[j])
                elif self.__activation_mode == Activation_Mode.Tanh: errors.append(self.dTanh(y) * deltas[j])
            elif self.__error_mode == Error_Mode.Cross_Entropy:
                if j == 0:
                    errors.append(np.ones_like(y) * deltas[j])
                else:
                    if self.__activation_mode == Activation_Mode.Sigmoid: errors.append(self.dSigmoid(y) * deltas[j])
                    elif self.__activation_mode == Activation_Mode.Tanh: errors.append(self.dTanh(y) * deltas[j])

        self.__errors = deltas

        for k, y in enumerate(y_layers[:-2]):
            gradients.append(np.dot(y.reshape((len(y),1)), errors[-k-1][1:].reshape((1,len(errors[-k-1][1:])))))
        gradients.append(np.dot(y_layers[-2].reshape((len(y_layers[-2]),1)), errors[0].reshape((1,len(errors[0])))))

        return gradients, E


    def update_weights(self, gradients, samples=None, desired_output=None):
        self.__ex_weights = deepcopy(self.__weights)
        if self.__steepest_descent == Steepest_Descent.On:
            self.__eta = 2 * self.__initial_eta
            ex_E = np.inf
            while(True):
                ###Trial Epoch
                for i in range(len(self.__weights)):
                    self.__weights[i] += (gradients[i] * -self.__eta) + (self.__momentum * self.__delta_weights[i])
                ###Eta calculation
                new_E = 0
                for s, t in zip(samples, desired_output):
                    new_E += self.cost_mse(np.asarray(self.feed_foreward(s)[-1] - t))
                new_E /= len(samples)
                if ex_E == new_E:
                    self.__weights = deepcopy(self.__ex_weights)
                    break
                if new_E < self.__initial_E:
                    self.__initial_E = new_E
                    break
                else:
                    self.__weights = deepcopy(self.__ex_weights)
                    self.__eta /= 2
                    ex_E = new_E
        elif self.__steepest_descent == Steepest_Descent.Off:
            for i in range(len(self.__weights)):
                self.__weights[i] += (gradients[i] * -self.__eta) + (self.__momentum * self.__delta_weights[i])
        self.__delta_weights = np.asarray(self.__weights) - np.asarray(self.__ex_weights)


    def train(self, train_samples, train_classes, validate_samples, validate_classes, eta=0.001, momnentum=0.0, max_epoches=500, accuracy_threshold=0.0, batch_length=10):
        self.__train_samples = train_samples
        self.__train_classes = train_classes
        self.__validate_samples = validate_samples
        self.__validate_classes = validate_classes
        self.__test_samples = []
        self.__test_classes = []
        self.__train_length = len(self.__train_samples)
        self.__validate_length = len(self.__validate_samples)
        self.__batch_length = batch_length
        self.__eta = eta
        self.__initial_eta = eta
        self.__momentum = momnentum
        self.__accuracy_threshold = accuracy_threshold
        self.__maximum_epoches = max_epoches

        plt.ion()
        figError = plt.figure()
        plt.suptitle("Accuracy Curve")
        axError = figError.add_subplot(1, 1, 1)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        axError.grid()
        hTrain, = plt.plot([], [], '-b', label='Train Accuracy')
        hValidation, = plt.plot([], [], '-r', label='Validation Accuracy')
        axError.legend(loc='best')
        figWeight = plt.figure()
        plt.suptitle("BackPropagation Error")
        weightHs = []
        weightAxs = []
        w = self.__hidden_layers_size+1
        for i in range(w):
            axWeightTemp = figWeight.add_subplot(np.ceil(w / 2), 2, i + 1)
            plt.ylabel('BackPropagation Error')
            plt.xlabel('Epoch')
            hWeightemp, = plt.plot([], [], label='Layer #{}'.format(i + 1))
            axWeightTemp.grid()
            axWeightTemp.legend(loc='best')
            weightHs.append(hWeightemp)
            weightAxs.append(axWeightTemp)

        self.print_config()
        initial_E = 0
        for s, t in zip(train_samples, train_classes):
            initial_E += self.cost_mse(np.asarray(self.feed_foreward(s)[-1] - t))
        initial_E /= self.__train_length
        if self.__steepest_descent == Steepest_Descent.On: self.__train_samples = scale(train_samples);self.__validate_samples = scale(validate_samples)
        self.__initial_E = initial_E
        print("Initial Weights Accuracy = {} Error = {}".format(self.accuracy(self.__train_samples, self.__train_classes), self.__initial_E))

        for epoch in range(self.__maximum_epoches):
            total_E = 0
            sample_counter = 0
            batch_gradients = []
            batch_samples = []
            batch_classes = []
            batch_step = 0
            backpropagate_error = []
            for s,t in zip(self.__train_samples, self.__train_classes):
                y_layers = self.feed_foreward(s)
                gradients, E = self.back_propagate(y_layers,t)
                sample_counter+=1
                if self.__gradient_mode == Gradient_Mode.Stochastic:
                    self.update_weights(gradients)
                elif self.__gradient_mode == Gradient_Mode.Batch:
                    batch_gradients.append(np.asarray(gradients))
                    batch_samples.append(s)
                    batch_classes.append(t)
                    if (sample_counter % batch_length) == 0:
                        batch_step += 1
                        self.update_weights(np.mean(batch_gradients, axis=0), batch_samples, batch_classes)
                        batch_gradients = []
                        batch_samples = []
                        batch_classes = []
                        # print('Step {} of Epoch {} Completed.'.format(batch_step, epoch))
                backpropagate_error.append([np.mean(self.__errors[i]) for i in range(self.__errors.__len__())])
                total_E += E

            total_E /= self.__train_length
            train_accuracy = self.accuracy(self.__train_samples, self.__train_classes)
            validate_accuracy = self.accuracy(self.__validate_samples, self.__validate_classes)
            print("Epoch {} completed. \nTrain Accuracy = {} \nValidate Accuracy = {} \nError = {}".format(epoch,train_accuracy,validate_accuracy,total_E))

            axError.relim()
            axError.autoscale_view()
            hTrain.set_ydata(np.append(hTrain.get_ydata(), train_accuracy))
            hTrain.set_xdata(np.append(hTrain.get_xdata(), epoch))
            hValidation.set_ydata(np.append(hValidation.get_ydata(), validate_accuracy))
            hValidation.set_xdata(np.append(hValidation.get_xdata(), epoch))
            for idx, (ax, hl) in enumerate(zip(weightAxs, weightHs)):
                ax.relim()
                ax.autoscale_view()
                hl.set_ydata(np.append(hl.get_ydata(), np.mean(backpropagate_error, axis=0)[idx]))
                hl.set_xdata(np.append(hl.get_xdata(), epoch))
            figError.canvas.draw()
            figWeight.canvas.draw()
            plt.pause(0.00001)

            if train_accuracy >= self.__accuracy_threshold: break


    def predict(self, x):
        y = [x]
        for weight in self.__weights:
            y[-1] = np.concatenate(([1], y[-1]))
            if self.__activation_mode == Activation_Mode.Sigmoid: y.append(expit(np.dot(y[-1], weight)))
            elif self.__activation_mode == Activation_Mode.Tanh: y.append(np.tanh(np.dot(y[-1], weight)))

        output = expit(y[-1])
        idx = np.argmax(output)
        output = np.zeros_like(output)
        output[idx] = 1

        return output, expit(y[-1])


    def accuracy(self,x,y):
        accuracy = 0
        for sample,desired in zip(x,y):
            if np.array_equal(desired.flatten(),self.predict(sample)[0].flatten()): accuracy+=1
        try:
            return (accuracy/len(x))*100
        except ZeroDivisionError:
            return 0.0


    def dSigmoid(self, x):
        return x*(np.ones_like(x) - x)


    def dTanh(self, x):
        return np.ones_like(x) - (np.tanh(x)**2)


    def test(self, test_samples, test_classes):
        self.__test_samples = test_samples
        self.__test_classes = test_classes
        return self.accuracy(self.__test_samples,self.__test_classes)


    def cost_mse(self, delta):
        return np.sum(0.5 * delta**2)


    def cost_cross_entropy(self, y_layer, desired_output):
        y_layer = y_layer/np.sum(y_layer)
        return -np.sum(desired_output * np.log(y_layer) + (1 - desired_output) * np.log(1 - y_layer))


    def print_config(self):
        print('*'*15,'NN config','*'*15)
        print('Number of Features: ',self.__n_features)
        print('Number of Classes: ',self.__n_classes)
        print('Number of Hidden layers: ',self.__hidden_layers_size)
        print('Network Arch.: ',self.__n_neurons_layers)
        print('Train Length:', self.__train_length)
        print('Validation Length:', self.__validate_length)
        print('Activation Function: ', self.__activation_mode)
        print('Weight Damp Factor={}'.format(self.__weight_damp_factor))
        print('Gradient Mode:{}'.format(self.__gradient_mode))
        print('Bratch Length:{}'.format(self.__batch_length))
        print('Steepest Descent:{}'.format(self.__steepest_descent))
        print('Learning rate={} Momentum={}'.format(self.__eta,self.__momentum))
        print('Maximum Epoches={} Error Threshold={}'.format(self.__maximum_epoches,self.__accuracy_threshold))
        print('*'*41,'\n')


    def get_config(self):
        dictionary={'n_features':self.__n_features,
                    'n_classes':self.__n_classes,
                    'hidden_layers_size':self.__hidden_layers_size,
                    'arch':self.__n_neurons_layers,
                    'train_len':self.__train_length,
                    'validate_len':self.__validate_length,
                    'activation':self.__activation_mode,
                    'eta':self.__eta,
                    'momentum':self.__momentum,
                    'weight_damp_factor':self.__weight_damp_factor,
                    'gradient_mode':self.__gradient_mode,
                    'batch_length':self.__batch_length,
                    'steepest_descent_mode':self.__steepest_descent,
                    'maximum_epoches':self.__maximum_epoches,
                    'accuracy_threshold':self.__accuracy_threshold}
        return dictionary


    def get_performance(self):
        dictionary = {'train_accuracy':self.accuracy(self.__train_samples,self.__train_classes),
                      'validate_accuracy':self.accuracy(self.__validate_samples,self.__validate_classes),
                      'test_accuracy':self.accuracy(self.__test_samples,self.__test_classes)}
        return dictionary


    def save_model(self, model_name, directory=None):
        if directory == None: directory='./models/'
        config = self.get_config()
        accuracy = self.get_performance()
        d = {'config':config,
             'accuracy':accuracy,
             'weights':self.__weights}
        np.save(directory+model_name, d)
        print("Model {} save successfully at {}.".format(model_name,directory))


    def load_model(self, model_name):
        model = np.load(model_name)
        config = model.item().get('config')
        self.__n_features = config.get('n_features')
        self.__n_classes = config.get('n_classes')
        self.__hidden_layers_size = config.get('hidden_layers_size')
        self.__n_neurons_layers = config.get('arch')
        self.__train_length = config.get('train_len')
        self.__validate_length = config.get('validate_len')
        self.__activation_mode = config.get('activation')
        self.__eta = config.get('eta')
        self.__momentum = config.get('momentum')
        self.__weight_damp_factor = config.get('weight_damp_factor')
        self.__gradient_mode = config.get('gradient_mode')
        self.__batch_length = config.get('batch_length')
        self.__steepest_descent = config.get('steepest_descent_mode')
        self.__maximum_epoches = config.get('maximum_epoches')
        self.__accuracy_threshold = config.get('accuracy_threshold')
        self.print_config()

        print('*'*15,'Recorded Accuracies','*'*15)
        accuracies = model.item().get('accuracy')
        print("Train Accuracy: ",accuracies.get('train_accuracy'))
        print("Validation Accuracy: ",accuracies.get('validate_accuracy'))
        print("Test Accuracy: ",accuracies.get('test_accuracy'))
        print('*'*51,'\n')

        self.__weights = model.item().get('weights')
        print("Activation_Mode Loaded Successfully.")
