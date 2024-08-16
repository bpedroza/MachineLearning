const tf = require('@tensorflow/tfjs-node');

class LinearRegressionTF {

    constructor(features, labels, options) {
        var numWeights = (features[0].length || 0) + 1;
        this.weights = tf.zeros([numWeights, 1]);
        this.labels = tf.tensor(labels);
        this.features = this.processFeatures(features);
        this.mse = null;
        this.lastMse = null;

        this.options = {
            learningRate: 0.1,
            maxIterations: 1000,
            ...options
        };
    }

    appendOnesToFeatures(features) {
        return tf.ones([features.shape[0], 1])
            .concat(features, 1)
    }

    gradientDescent(features, labels) {
        const currentGuesses = features.matMul(this.weights);
        const differences = currentGuesses.sub(labels);

        const slopes = features.transpose().matMul(differences).div(features.shape[0]);
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    predict(observations) {
        const obsTensor = this.processFeatures(observations);
        return obsTensor.matMul(this.weights).arraySync();
    }

    processFeatures(features) {
        features = tf.tensor(features);
        features = this.standardizeFeatureValues(features);
        features = this.appendOnesToFeatures(features);

        return features;
    }

    recordMSE() {
        this.mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .arraySync();
    }

    standardizeFeatureValues(features) {
        if (!this.mean && !this.variance) {
            const { mean, variance } = tf.moments(features, 0);
            this.mean = mean;
            this.variance = variance;
        }
        return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    test(testFeatures, testLabels) {
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        const predictions = testFeatures.matMul(this.weights);
        const ssResiduals = testLabels.sub(predictions).pow(2).sum().arraySync();
        const ssTotal = testLabels.sub(testLabels.mean()).pow(2).sum().arraySync();

        const coefficientOfDetermination = 1 - (ssResiduals / ssTotal);
        // Negative number is worse than just guessing the average.
        // Positive number indicates a good result.
        return coefficientOfDetermination;
    }

    train() {
        for (var i = 0; i < this.options.maxIterations; i++) {
            this.gradientDescent(this.features, this.labels);
            this.lastMse = this.mse;
            this.recordMSE();
            this.updateLearningRate();
        }
    }

    updateLearningRate() {
        if(this.lastMse === null) {
            return;
        }

        if(this.mse > this.lastMse) {
            this.options.learningRate /= 1.2; 
        } else {
            this.options.learningRate *= 1.01;
        }
    }
}

module.exports = LinearRegressionTF;