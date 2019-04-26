import csv
import math
import random
from flask import Flask , render_template,request

app = Flask(__name__)


'''

    Description : Function readFile reads the input csv data and stores the result in a list
    Input : None
    Output : dataset
'''

def readFile():
    lines = csv.reader(open(r'diabetes.csv'))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    # print(dataset)
    return dataset

'''
    Description : Function splitDataset reads the input dataset and splitratio and splits the data in that ratio
                  percentage of the ratio is used as traning data and remaning is considered as test data.dataset is 
                  copied in 'copy' and then using a randomisazed approach , we keep poping the values from copy
                  and storing it into trainset 
    Input : dataset , splitratio
    Output : dataset , copy


'''
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) <= trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    # print(len(trainSet))
    return [trainSet, copy]


'''

    Description : function separateByClass splits the dataset with respect to the last columns , i.e the result column based on 
                  its values 1 and 0 and is stored in separated (list datastructure)
    Input : dataset
    Output : separated


'''


def separateByClass(dataset):
    separated = {}
    classA = []
    classB = []

    for i in range(len(dataset)):
        vector = dataset[i]

        if (vector[-1] == 1.0):
            classA.append(vector)
        else:
            classB.append(vector)
    separated.update({1.0: classA})
    separated.update({0.0: classB})
    return separated


'''
    Description : function clculatestandarddeviation is used to calculate the standard deviation of the input numbers
                  First the mean value is calculated for list of numbers and then the we calculate the variance.
                  Since std deviation is root of variance we calculate  the root of variance
    Input :   numbers
    Output :  math.sqrt(variance) 

'''

def clculatestandarddeviation(numbers):
    avg = calculatemean(numbers)
    variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)

'''
    Description : function calculatemean is used to calculate the average of the input numbers
                
    Input :   numbers
    Output :  sum(numbers) / float(len(numbers))

'''
def calculatemean(numbers):
    return sum(numbers) / float(len(numbers))



'''
    Description : function summarize calls means and standard deviation for all the  input numbers and 
                  stores it in summaries list of tuple.
                
    Input :   dataset
    Output :  summaries

'''

def summarize(dataset):
    summaries = [(calculatemean(attribute), clculatestandarddeviation(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


'''
    Description : function summarizeByClass stores the data with respect to class variable .
    Input :   dataset
    Output :  summaries

'''


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    print("summarised data for model {}".format(summaries))
    return summaries



'''
    Description : function calculateProbability calculates the probability using the gaussian distribution .
    Input :   x, mean, stdev
    Output :  gaussian distribution ( (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent )

'''

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


'''
    Description : function calculateClassProbabilities calculates the probability using the gaussian distribution model for each of the 
                  attribute present in the testdata(inputvector) and then stores the value against the class label in dictionary(probabilities)
    Input :   summaries, inputVector
    Output :  probabilities

'''


def calculateClassProbabilities(summaries, inputVector):

    print("inputvetor {}".format(inputVector))
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        print(classValue , classSummaries)
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            #print("calculating the mean and standard deviation for {}".format(classSummaries[i]))
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    print(probabilities)
    return probabilities




'''
    Description : function predict predicts the value of the input test data i.e input vector and returns 
                  its class label
    Input :   summaries, inputVector
    Output :  bestLabel

'''
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability >= bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel



'''
    Description : function getPredictions simply calls the predict function above and stores
                  result in list. This is called incase where we are calculating the metrics of
                  the classifier model
    Input :   summaries, testSet
    Output :  bestLabel

'''


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
    #result = predict(summaries, testSet)
        predictions.append(result)
    return predictions




'''
    Description : function getPredictions1 simply calls the predict function above and stores
                  result in list. This is called incase where we are trying to predict result 
                  of some input test data
    Input :   summaries, testSet
    Output :  bestLabel

'''
def getPredictions1(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet)
    #result = predict(summaries, testSet)
        predictions.append(result)
    return predictions



'''
    Description : function getmetrics calculates the accuracy , precision and recall  
    Input :   summaries, testSet
    Output :  bestLabel

'''

def getmetrics(testSet, predictions):
    correct = 0
    truepostive = 0
    truenegative=0
    falsepositive = 0
    falsenegative = 0

    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1

        # False negative , result is positive and prediction is negative
        if testSet[x][-1] == 1 and predictions[x] == 0:
            falsenegative += 1

        # True positive, result is positive and prediction is also positive
        if testSet[x][-1] == 1 and predictions[x] == 1:
            truepostive+= 1

        # True negative, result is negative and prediction is also negative
        if testSet[x][-1] ==  0 and predictions[x] == 0:
            truenegative += 1

        # False positive, result is negative and prediction is positive
        if testSet[x][-1] == 0 and predictions[x] == 1:
            falsepositive += 1

    accuracy = (correct / float(len(testSet))) * 100.0
    precision = (truepostive/(truepostive + falsepositive)) * 100.0
    recall = (truepostive/(truepostive + falsenegative)) * 100.0
    print(accuracy , precision , recall)
    return accuracy , precision , recall

@app.route("/")
def index():
    return render_template('home.html')


@app.route("/classifier")
def classifier():
    return render_template('classifier.html')



'''

    Description : function classify is the route function invoked by wsgi    
    Input :   summaries, testSet
    Output :  bestLabel

'''


@app.route("/classify")
def classify():
     preganancycount = float(request.args.get('preg'))
     glucose = float(request.args.get('glucose'))
     bloodpressure = float(request.args.get('bp'))
     skin = float(request.args.get('skin'))
     insulin = float(request.args.get('insulin'))
     bmi =float(request.args.get('bmi'))
     diabetes = float(request.args.get('diabetes'))
     age =float( request.args.get('age'))
     splitRatio = 0.70
     dataset = readFile()
     trainingSet, testSet = splitDataset(dataset, splitRatio)

     testSet=[preganancycount,glucose,bloodpressure,skin,insulin,bmi,diabetes,age]

     print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
     # prepare model
     summaries = summarizeByClass(trainingSet)
     # test model
     predictions = getPredictions1(summaries, testSet)
     testresults = None
     if predictions[0] == 1.0:
         testresults="Test result is positive :-( , You are diabetic"
     elif predictions[0] == 0.0:
         testresults = "Test result is negative :-) , You are not diabetic"

     #accuracy = getAccuracy(testSet, predictions)

    # print('Accuracy: {}'.format(accuracy))
     print("predictions {}".format(predictions))
     return render_template('classifierresult.html' , testresult = testresults)



'''
    Description : function accuracy is the route function invoked by wsgi    
    Input :   summaries, testSet
    Output :  bestLabel
'''
@app.route("/accuracy")
def accuracy():

     splitRatio = 0.70
     dataset = readFile()
     trainingSet, testSet = splitDataset(dataset, splitRatio)
     print('Split {0} rows into train = {1} and test = {2} rows'.format(len(dataset), len(trainingSet), len(testSet)))
     # prepare model
     summaries = summarizeByClass(trainingSet)
     # test model
     predictions = getPredictions(summaries, testSet)
     accuracy , precision , recall = getmetrics(testSet, predictions)
     print("predictions {}".format(predictions))
     return render_template('accuracyresult.html' , accuracy = accuracy , precision=precision , recall=recall )


if __name__ == '__main__':

    app.run()