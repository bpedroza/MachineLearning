
const loadCSV = require('../data/load-csv');
const _ = require('lodash');
const BatchLogisticRegressionTF = require('./BatchLogisticRegressionTF');

let { features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 20,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    converters: {
        'mpg': (value) => { 
            const mpg = parseFloat(value);

            if(mpg < 15) {
                return [1, 0, 0];
            } else if (mpg < 30) {
                return [0, 1, 0];
            } else {
                return [0, 0, 1];
            }
        }
    }
});
const batchSize = 10;
const regression = new BatchLogisticRegressionTF(batchSize, features, _.flatMap(labels), {
    maxIterations: 100, 
    decisionBoundary: 0.5,
    classificationAlgorithm: 'softmax'
});

regression.train();
console.log(regression.test(testFeatures, _.flatMap(testLabels)));
