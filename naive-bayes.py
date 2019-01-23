#!python
# Example of Naive Bayes implemented from Scratch in Python
import csv
import math
import random
import statistics


class NaiveBayes:
    def _separateByClass(self, dataset):
        separated = {}
        for vector in dataset:
            classification = vector[-1]
            if classification not in separated:
                separated[classification] = []
            separated[classification].append(vector)
        return separated

    def _summarize(self, dataset):
        summaries = [(statistics.mean(attrib), statistics.stdev(attrib))
                     for attrib in zip(*dataset)]
        return summaries[:-1]

    def summarizeByClass(self, dataset):
        separated = self._separateByClass(dataset)
        summaries = {classValue: self._summarize(instances)
                     for classValue, instances in separated.items()}
        return summaries

    def _calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return exponent / (math.sqrt(2 * math.pi) * stdev)

    def _calculateClassProbabilities(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in summaries.items():
            probabilities[classValue] = 1
            for x, summary in zip(inputVector, classSummaries):
                probabilities[classValue] *= self._calculateProbability(x, *summary)
        return probabilities

    def _predict(self, summaries, inputVector):
        probabilities = self._calculateClassProbabilities(summaries, inputVector)
        return max(probabilities, key=probabilities.get)

    def getPredictions(self, summaries, testSet):
        return [self._predict(summaries, i) for i in testSet]

    def getAccuracy(self, testSet, predictions):
        correct = sum(vec[-1] == predict for vec, predict in zip(testSet, predictions))
        return correct * 100 / len(testSet)


class NaiveBayesMain:
    def _loadCsv(self, filename):
        lines = csv.reader(open(filename))
        return [list(map(float, line)) for line in lines]

    def _splitDataset(self, dataset, splitRatio):
        trainSize = int(len(dataset) * splitRatio)
        random.shuffle(dataset)
        return dataset[:trainSize], dataset[trainSize:]

    def main(self):
        filename = 'diabetes.csv'
        splitRatio = 0.67

        dataset = self._loadCsv(filename)
        trainingSet, testSet = self._splitDataset(dataset, splitRatio)
        print('Split {0} rows into train = {1} and test = {2} rows'
              .format(len(dataset), len(trainingSet), len(testSet)))

        naiveBayes = NaiveBayes()
        # prepare model
        summaries = naiveBayes.summarizeByClass(trainingSet)
        # test model
        predictions = naiveBayes.getPredictions(summaries, testSet)
        accuracy = naiveBayes.getAccuracy(testSet, predictions)
        print('Accuracy: {0:.2f}%'.format(accuracy))


if __name__ == '__main__':
    NaiveBayesMain().main()
