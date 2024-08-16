
const loadCSV = require('../data/load-csv');
const BatchLogisticRegressionTF = require('./BatchLogisticRegressionTF');

let { features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    converters: {
        'passedemissions': (value) => value === 'TRUE' ? 1 : 0
    }
});
const batchSize = 7;
const regression = new BatchLogisticRegressionTF(batchSize, features, labels, {maxIterations: 10, decisionBoundary: 0.5});
regression.train();
console.log(regression.test(testFeatures, testLabels));
