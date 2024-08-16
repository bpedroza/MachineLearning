const tf = require('@tensorflow/tfjs-node');
const LogisticRegressionTF = require('./LogisticRegressionTF');

class BatchLogisticRegressionTF extends LogisticRegressionTF{

    constructor(batchSize, options) {
        super(options);
        this.batchSize = batchSize;
    }

    train(features, labels) {
        this.setWeights(features, labels);
        features = this.processFeatures(features);
        labels = tf.tensor(labels);

        const batchQuantity = Math.floor(features.shape[0] / this.batchSize);
        for (var i = 0; i < this.options.maxIterations; i++) {
            for (var j = 0; j < batchQuantity; j++) {
                this.weights = tf.tidy(() => {
                    const startRow = j * this.batchSize;
                    const featureBatch = features.slice([startRow, 0], [this.batchSize, -1]);
                    const labelBatch = labels.slice([startRow, 0], [this.batchSize, -1]);
                    return this.gradientDescent(featureBatch, labelBatch);
                });
            }
            
            this.recordCrossEntropy(features, labels);
            this.updateLearningRate();
            this.lastCrossEntropy = this.crossEntropy;
        }
    }
}

module.exports = BatchLogisticRegressionTF;