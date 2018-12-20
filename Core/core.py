from util import *

def perceptron(Xtrain, Ytrain, w, Xtest, Ytest, totalEpoches=700, errorThreshold=0.0, eta=0.01, theta=0):
    global epoch
    w[0] = theta
    errorsTrain = []
    errorsTest = []
    Ws = [w]

    plt.ion()

    figError = plt.figure()
    plt.suptitle("Perceptron Error Rate")
    axError = figError.add_subplot(1,1,1)
    plt.ylabel('Total Error')
    plt.xlabel('Epoch')
    axError.grid()
    hTrain, = plt.plot([], [],'-b', label='Train Error')
    hTest, = plt.plot([], [],'-r', label='Test Error')
    axError.legend(loc='best')

    figWeight = plt.figure()
    plt.suptitle("Perceptron Weights")
    weightHs = []
    weightAxs = []
    for i in range(len(w)):
        axWeightTemp = figWeight.add_subplot(np.ceil(len(w)/2),2,i+1)
        plt.ylabel('Weight')
        plt.xlabel('Epoch')
        hWeightemp, = plt.plot([], [], label='Weight #{}'.format(i+1))
        axWeightTemp.grid()
        axWeightTemp.legend(loc='best')
        weightHs.append(hWeightemp)
        weightAxs.append(axWeightTemp)

    for epoch in range(totalEpoches):
        accErrorTrain = 0
        accErrorTest = 0

        for i, x in enumerate(Xtrain):
            prediction = predict(w, x)
            error = (Ytrain[i] - prediction)
            accErrorTrain += np.abs(Ytrain[i]-prediction)
            for j in range(len(w)):
                w[j] += eta * error * x[j]
        accErrorTrain = float(accErrorTrain/len(Xtrain))
        errorsTrain.append(accErrorTrain)

        for i, x in enumerate(Xtest):
            prediction = predict(w, x)
            accErrorTest += np.abs(Ytest[i]-prediction)
        accErrorTest = float(accErrorTest/len(Xtest))
        errorsTest.append(accErrorTest)

        Ws.append(w)

        axError.relim()
        axError.autoscale_view()
        hTrain.set_ydata(np.append(hTrain.get_ydata(), accErrorTrain))
        hTrain.set_xdata(np.append(hTrain.get_xdata(), epoch))
        hTest.set_ydata(np.append(hTest.get_ydata(), accErrorTest))
        hTest.set_xdata(np.append(hTest.get_xdata(), epoch))

        for idx,(ax,hl) in enumerate(zip(weightAxs,weightHs)):
            ax.relim()
            ax.autoscale_view()
            hl.set_ydata(np.append(hl.get_ydata(), w[idx]))
            hl.set_xdata(np.append(hl.get_xdata(), epoch))

        figError.canvas.draw()
        figWeight.canvas.draw()
        plt.pause(0.00001)

        if accErrorTrain<=errorThreshold: break

    return Ws, epoch+1, errorsTrain, errorsTest, figError, figWeight

def adaline(Xtrain, Ytrain, w, Xtest, Ytest, totalEpoches=700, errorThreshold=0.0, eta=0.01, theta=0):
    global epoch
    w[0] = theta
    errorsTrain = []
    errorsTest = []
    Ws = [w]

    plt.ion()

    figError = plt.figure()
    plt.suptitle("ADALINE Error Rate")
    axError = figError.add_subplot(1,1,1)
    plt.ylabel('Total Error')
    plt.xlabel('Epoch')
    axError.grid()
    hTrain, = plt.plot([], [],'-b', label='Train Error')
    hTest, = plt.plot([], [],'-r', label='Test Error')
    axError.legend(loc='best')

    figWeight = plt.figure()
    plt.suptitle("ADALINE Weights")
    weightHs = []
    weightAxs = []
    for i in range(len(w)):
        axWeightTemp = figWeight.add_subplot(np.ceil(len(w)/2),2,i+1)
        plt.ylabel('Weight')
        plt.xlabel('Epoch')
        hWeightemp, = plt.plot([], [], label='Weight #{}'.format(i+1))
        axWeightTemp.grid()
        axWeightTemp.legend(loc='best')
        weightHs.append(hWeightemp)
        weightAxs.append(axWeightTemp)


    for epoch in range(totalEpoches):
        accErrorTrain = 0
        accErrorTest = 0

        for i, x in enumerate(Xtrain):
            prediction = activation(w, x)
            error = (Ytrain[i] - prediction)
            accErrorTrain += np.abs(Ytrain[i]-predict(w,x))
            for j in range(len(w)):
                w[j] += eta * error * x[j]
        accErrorTrain = float(accErrorTrain/len(Xtrain))
        errorsTrain.append(accErrorTrain)

        for i, x in enumerate(Xtest):
            accErrorTest += np.abs(Ytest[i]-predict(w,x))
        accErrorTest = float(accErrorTest/len(Xtest))
        errorsTest.append(accErrorTest)

        Ws.append(w)

        axError.relim()
        axError.autoscale_view()
        hTrain.set_ydata(np.append(hTrain.get_ydata(), accErrorTrain))
        hTrain.set_xdata(np.append(hTrain.get_xdata(), epoch))
        hTest.set_ydata(np.append(hTest.get_ydata(), accErrorTest))
        hTest.set_xdata(np.append(hTest.get_xdata(), epoch))

        for idx,(ax,hl) in enumerate(zip(weightAxs,weightHs)):
            ax.relim()
            ax.autoscale_view()
            hl.set_ydata(np.append(hl.get_ydata(), w[idx]))
            hl.set_xdata(np.append(hl.get_xdata(), epoch))

        figError.canvas.draw()
        figWeight.canvas.draw()
        plt.pause(0.00001)

        if accErrorTrain<=errorThreshold: break

    return Ws, epoch+1, errorsTrain, errorsTest, figError, figWeight

def approximate(Xtrain, Ytrain, w, Xtest, Ytest, features, totalEpoches=700, errorThreshold=0.0, eta=0.01, theta=0):
    global epoch
    w[0] = theta
    errorsTrain = []
    errorsTest = []
    Ws = [w]

    plt.ion()

    figError = plt.figure()
    plt.suptitle("Function Approximation Error Rate")
    axError = figError.add_subplot(1,1,1)
    plt.ylabel('Normalized MSE')
    plt.xlabel('Epoch')
    axError.grid()
    hTrain, = plt.plot([], [],'-b', label='Train Error')
    hTest, = plt.plot([], [],'-r', label='Test Error')
    axError.legend(loc='best')

    figWeight = plt.figure()
    plt.suptitle("Approximated Weights")
    weightHs = []
    weightAxs = []
    for i in range(len(w)):
        axWeightTemp = figWeight.add_subplot(np.ceil(len(w)/2),2,i+1)
        plt.ylabel('Weight')
        plt.xlabel('Epoch')
        hWeightemp, = plt.plot([], [], label='Weight #{}'.format(i+1))
        axWeightTemp.grid()
        axWeightTemp.legend(loc='best')
        weightHs.append(hWeightemp)
        weightAxs.append(axWeightTemp)

    figFunc = plt.figure()
    plt.suptitle("Approximated Function")
    funcHs = []
    funcAxs = []
    for i in range(len(w)):
        axFuncTemp = figFunc.add_subplot(np.ceil(len(w)/2),2,i+1)
        plt.ylabel('mpg')
        plt.xlabel(features[i])
        plt.plot(Xtrain[:,i],Ytrain, 'r*', label='Real')
        hFunctemp, = plt.plot([], [], 'b-', label='{} (Estimated)'.format(features[i]))
        axFuncTemp.grid()
        axFuncTemp.legend(loc='best')
        funcHs.append(hFunctemp)
        funcAxs.append(axFuncTemp)

    for epoch in range(totalEpoches):
        accErrorTrain = 0.0
        accErrorTest = 0.0

        for i, x in enumerate(Xtrain):
            prediction = activation(w, x)
            error = (Ytrain[i] - prediction)
            accErrorTrain +=  float(0.5 * (Ytrain[i]-prediction)**2)
            for j in range(len(w)):
                w[j] += eta * error * x[j]
        errorsTrain.append(accErrorTrain/len(Xtrain))

        for i, x in enumerate(Xtest):
            accErrorTest += float(0.5 * (Ytest[i]-activation(w,x))**2)
        errorsTest.append(accErrorTest/len(Xtest))

        Ws.append(w)

        axError.relim()
        axError.autoscale_view()
        hTrain.set_ydata(np.append(hTrain.get_ydata(), accErrorTrain))
        hTrain.set_xdata(np.append(hTrain.get_xdata(), epoch))
        hTest.set_ydata(np.append(hTest.get_ydata(), accErrorTest))
        hTest.set_xdata(np.append(hTest.get_xdata(), epoch))

        for idx,(ax,hl) in enumerate(zip(weightAxs,weightHs)):
            ax.relim()
            ax.autoscale_view()
            hl.set_ydata(np.append(hl.get_ydata(), w[idx]))
            hl.set_xdata(np.append(hl.get_xdata(), epoch))

        for idx,(ax,hl) in enumerate(zip(funcAxs,funcHs)):
            ax.relim()
            ax.autoscale_view()
            hl.set_xdata(Xtrain[:,idx])
            hl.set_ydata(w[idx]*Xtrain[:,idx]+w[0])

        figError.canvas.draw()
        figWeight.canvas.draw()
        figFunc.canvas.draw()
        plt.pause(0.00001)

        if accErrorTrain<=errorThreshold: break

    return Ws, epoch+1, errorsTrain, errorsTest, figError, figWeight, figFunc