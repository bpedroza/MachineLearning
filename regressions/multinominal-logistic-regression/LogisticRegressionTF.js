const tf = require('@tensorflow/tfjs-node');

class LogisticRegressionTF {

    constructor(options) {
        this.crossEntropy = null;
        this.lastCrossEntropy = null;

        this.options = {
            decisionBoundary: 0.5,
            // sigmoid (multi true classifications possible)| softmax (single true classification)
            classificationAlgorithm: 'sigmoid',
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
        const currentGuesses = this.predictRaw(features);
        const differences = currentGuesses.sub(labels);

        const slopes = features.transpose().matMul(differences).div(features.shape[0]);
        return this.weights.sub(slopes.mul(this.options.learningRate));
    }

    predict(observations) {
        return this.predictRaw(this.processFeatures(observations))
            .greater(this.options.decisionBoundary)
            .cast('float32')
            .argMax(1);
    }

    predictRaw(features) {
        if(this.options.classificationAlgorithm == 'sigmoid') {
            return features
            .matMul(this.weights)
            .sigmoid();
        }

        return features
        .matMul(this.weights)
        .softmax();
    }

    processFeatures(features) {
        features = tf.tensor(features);
        features = this.standardizeFeatureValues(features);
        features = this.appendOnesToFeatures(features);

        return features;
    }

    recordCrossEntropy(features, labels) {
        this.crossEntropy = tf.tidy(() => {
            const guesses = this.predictRaw(features);

            const firstTerm = labels.transpose().matMul(guesses.add(0.00000001).log());
            const secondTerm = labels.mul(-1)
                .add(1)
                .transpose()
                .matMul(guesses.mul(-1).add(1.00000001).log());

            return firstTerm
                .add(secondTerm)
                .div(features.shape[0])
                .mul(-1)
                .arraySync();
        });
    }

    standardizeFeatureValues(features) {
        if (!this.mean && !this.variance) {
            const { mean, variance } = tf.moments(features, 0);
            this.mean = mean;
            this.variance = variance;
            // Set all variance values of 0 to 1 to avoid division by 0.
            const addOneTensor = this.variance.cast('bool').logicalNot().cast('float32');
            this.variance = this.variance.add(addOneTensor);
        }
        return features.sub(this.mean).div(this.variance.pow(0.5));
    }

    test(testFeatures, testLabels) {
        const predictions = this.predict(testFeatures);
        testLabels = tf.tensor(testLabels).argMax(1);

        const incorrect = predictions
            .notEqual(testLabels)
            .sum()
            .arraySync();

        return (predictions.shape[0] - incorrect) / predictions.shape[0];
    }

    train(features, labels) {
        this.setWeights(features, labels);
        features = this.processFeatures(features);
        labels = tf.tensor(labels);
        for (var i = 0; i < this.options.maxIterations; i++) {
            this.weights = tf.tidy(() => {
                return this.gradientDescent(features, labels);
            });
            
            this.recordCrossEntropy(features, labels);
            this.updateLearningRate();
            this.lastCrossEntropy = this.crossEntropy;
        }
    }

    setWeights(features, labels) {
        var numWeights = (features[0].length || 0) + 1;
        this.weights = tf.zeros([numWeights, labels[0].length || 0]);
    }

    updateLearningRate() {
        if(this.lastCrossEntropy === null) {
            return;
        }

        if(this.crossEntropy > this.lastCrossEntropy) {
            this.options.learningRate /= 1.2; 
        } else {
            this.options.learningRate *= 1.01;
        }
    }
}

module.exports = LogisticRegressionTF;