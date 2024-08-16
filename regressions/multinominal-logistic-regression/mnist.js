const _ = require('lodash');
const BatchLogisticRegressionTF = require('./BatchLogisticRegressionTF');
const mnist = require('mnist-data');


const batchSize = 500;
const trainingDataSize = 60000;
const testDataSize = 10000;

const regression = new BatchLogisticRegressionTF(
    batchSize, 
    {
        learningRate: 1,
        iterations: 80
    }
);

regression.train(...trainingData());
console.log(regression.test(...testData()));

function trainingData() {
    const mnistTrainingData = mnist.training(0, trainingDataSize);
    return [
        mapFeatures(mnistTrainingData.images.values),
        encodeLabels(mnistTrainingData.labels.values)
    ];
}

function testData() {
    const mnistTestData = mnist.testing(0, testDataSize);
    return [
        mapFeatures(mnistTestData.images.values),
        encodeLabels(mnistTestData.labels.values)
    ];
}

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