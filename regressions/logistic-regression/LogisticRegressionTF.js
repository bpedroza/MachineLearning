const tf = require('@tensorflow/tfjs-node');

class LogisticRegressionTF {

    constructor(features, labels, options) {
        var numWeights = (features[0].length || 0) + 1;
        this.weights = tf.zeros([numWeights, 1]);
        this.labels = tf.tensor(labels);
        this.features = this.processFeatures(features);
        this.crossEntropyHistory = [];

        this.options = {
            decisionBoundary: 0.5,
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
        const currentGuesses = features.matMul(this.weights).sigmoid();
        const differences = currentGuesses.sub(labels);

        const slopes = features.transpose().matMul(differences).div(features.shape[0]);
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    predict(observations) {
        return this.processFeatures(observations)
            .matMul(this.weights)
            .sigmoid()
            .greater(this.options.decisionBoundary)
            .cast('float32');
    }

    processFeatures(features) {
        features = tf.tensor(features);
        features = this.standardizeFeatureValues(features);
        features = this.appendOnesToFeatures(features);

        return features;
    }

    recordCrossEntropy() {
        const guesses = this.features
        .matMul(this.weights)
        .sigmoid();

        const firstTerm = this.labels.transpose().matMul(guesses.log());
        const secondTerm = this.labels.mul(-1)
            .add(1)
            .transpose()
            .matMul(guesses.mul(-1).add(1).log());

        const crossEntropy = firstTerm
            .add(secondTerm)
            .div(this.features.shape[0])
            .mul(-1)
            .arraySync();

        this.crossEntropyHistory.unshift(crossEntropy);
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
        testLabels = tf.tensor(testLabels);

        const predictions = this.predict(testFeatures);
        const incorrect = predictions.sub(testLabels).abs().sum().arraySync();

        // return percent correct
        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    train() {
        for (var i = 0; i < this.options.maxIterations; i++) {
            this.gradientDescent(this.features, this.labels);
            this.recordCrossEntropy();
            this.updateLearningRate();
        }
    }

    updateLearningRate() {
        if(this.crossEntropyHistory.length <= 1) {
            return;
        }

        if(this.crossEntropyHistory[0] > this.crossEntropyHistory[1]) {
            this.options.learningRate /= 1.2; 
        } else {
            this.options.learningRate *= 1.01;
        }
    }
}

module.exports = LogisticRegressionTF;