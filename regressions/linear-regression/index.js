
const loadCSV = require('../data/load-csv');
const BatchLinearRegressionTF = require('./BatchLinearRegressionTF');

let { features, labels, testFeatures, testLabels} = loadCSV('../data/cars.csv', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['weight', 'displacement'],
    labelColumns: ['mpg']
});

const batchSize = 7;
const regression = new BatchLinearRegressionTF(batchSize, features, labels, {maxIterations: 5});

regression.train();
console.log(regression.test(testFeatures, testLabels));
console.log(regression.predict([
    [2.5, 350]
]));
