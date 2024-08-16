

const tf = require('@tensorflow/tfjs-node');
const loadCSV = require('./load-csv');

let { features, labels, testFeatures, testLabels} = loadCSV('kc_house_data.csv', {
    shuffle: true,
    splitTest: 10,
    dataColumns: ['lat', 'long', 'yr_built', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms'],
    labelColumns: ['price']
});

testFeatures.forEach((row, key) => {
    const guess = knn(
        tf.tensor(features), 
        tf.tensor(labels), 
        tf.tensor(row), 
        10
    );
    const actual = testLabels[key][0];
    const err = (actual - guess) / actual;

    console.log('Guess:', guess, 'Actual: ', actual);
    console.log('Error: ', err * 100);
})

function knn(features, labels, predictionPoint, k) {
    const {mean, variance} = tf.moments(features, 0);
    const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));
    return features
        .sub(mean)
        .div(variance.pow(0.5))
        .sub(scaledPrediction)
        .pow(2)
        .sum(1)
        .pow(0.5)
        .expandDims(1)
        .concat(labels, 1)
        .unstack()
        .sort((a, b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1)
        .slice(0, k)
        .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k;
}