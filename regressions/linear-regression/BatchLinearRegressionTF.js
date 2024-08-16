const tf = require('@tensorflow/tfjs-node');
const LinearRegressionTF = require('./LinearRegressionTF');

class BatchLinearRegressionTF extends LinearRegressionTF{

    constructor(batchSize, features, labels, options) {
        super(features, labels, options);
        this.batchSize = batchSize;
    }

    train() {
        const features = this.features;
        const batchQuantity = Math.floor(features.shape[0] / this.batchSize);
        for (var i = 0; i < this.options.maxIterations; i++) {
            for (var j = 0; j < batchQuantity; j++) {
                const startRow = j * this.batchSize;
                const featureBatch = features.slice([startRow, 0], [this.batchSize, -1]);
                const labelBatch = this.labels.slice([startRow, 0], [this.batchSize, -1]);
                this.gradientDescent(featureBatch, labelBatch);
            }
            
            this.lastMse = this.mse;
            this.recordMSE();
            this.updateLearningRate();
        }
    }
}

module.exports = BatchLinearRegressionTF;