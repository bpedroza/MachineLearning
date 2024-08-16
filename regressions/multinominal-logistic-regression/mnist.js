const _ = require('lodash');
const BatchLogisticRegressionTF = require('./BatchLogisticRegressionTF');
const mnist = require('mnist-data');

const mnistTrainingData = mnist.training(0, 2000);
const mnistTestData = mnist.testing(0, 100)

const features = mapFeatures(mnistTrainingData.images.values);
const encodedLabels = encodeLabels(mnistTrainingData.labels.values);
const testFeatures = mapFeatures(mnistTestData.images.values);
const encodedTestLabels = encodeLabels(mnistTestData.labels.values);

const batchSize = 10;
const regression = new BatchLogisticRegressionTF(batchSize, features, encodedLabels, {
    learningRate: .1,
    iterations: 15
});

regression.train();
console.log(regression.test(testFeatures, encodedTestLabels));


function mapFeatures(features) {
    return features.map(image => _.flatMap(image));
}

function encodeLabels(labels) {
    return labels.map(label => {
        const row = new Array(10).fill(0);
        row[label] = 1;
        return row;
    });
}