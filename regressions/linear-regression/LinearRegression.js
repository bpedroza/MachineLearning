const tf = require('@tensorflow/tfjs-node');

class LinearRegression {

    constructor(features, labels, options) {
        this.features = features;
        this.labels = labels;
        this.options = {
            learningRate: 0.1,
            maxIterations: 1000,
            ...options
        };
        this.m = 0;
        this.b = 0;
    }

    gradientDescent() {
        const currentGuesses = this.features.map(row => this.m * row[0] + this.b);
        const bSlope = (currentGuesses
            .map((guess, idx) => guess - this.labels[idx][0])
            .reduce((acc, curr) => acc + curr) * 2) / this.features.length;
        
        const mSlope = (currentGuesses
            .map((guess, idx) => (this.features[idx][0]) * (guess - this.labels[idx][0]) )
            .reduce((acc, curr) => acc + curr) * 2) / this.features.length;
        this.m = this.m - (mSlope * this.options.learningRate);
        this.b = this.b - (bSlope * this.options.learningRate);
    }

    predict() {

    }

    test() {

    }

    train() {
        for(var i = 0; i < this.options.maxIterations; i++) {
            this.gradientDescent();
        }
        console.log(this.m, this.b);
    }
}

module.exports = LinearRegression;