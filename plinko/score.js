const outputs = [];
const k = 9;

function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([dropPosition, bounciness, size, bucketLabel]);
}

function runAnalysis() {
  const testSetSize = 100;
  
  for(var feature = 0; feature < 3; feature++) {
    const data = _.map(outputs, row => [row[feature], _.last(row)]);
    const splitData = splitDataSet(normalizeAll(data), testSetSize);
    let countCorrect = 0;
    splitData.testSet.forEach((record) => {
      const guess = knnAlgorithm(splitData.trainingSet, _.initial(record), k);
      const actual = _.last(record);
      if(guess == actual) {
        countCorrect++;
      }
    });
  
    console.log('K value:', k, 'Feature at index:', feature, 'Accuracy:', (countCorrect / testSetSize) * 100);
  }
  
}

function distance(coordinatesA, coordinatesB) {
  return _.chain(coordinatesA)
    .zip(coordinatesB)
    .map(([a,b]) => (a - b) ** 2)
    .sum()
    .value() ** 0.5;

}

function splitDataSet(data, testCount) {
  const shuffled = _.shuffle(data);

  return {
    testSet: _.slice(shuffled, 0, testCount),
    trainingSet: _.slice(shuffled, testCount)
  };
}

function knnAlgorithm(data, toPredict, k) {
  return _.chain(data)
    .map(row => {
      return [
        distance(_.initial(row), toPredict), 
        _.last(row)
      ]
    })
    .sortBy(row => row[0])
    .slice(0, k)
    .countBy(row => row[1])
    .toPairs()
    .sortBy(row => row[1])
    .last()
    .first()
    .parseInt()
    .value();
}

function normalizeAll(data)
{
  const cloned = _.cloneDeep(data);
  const featureLength = _.initial(data[0]).length;
  for (var i = 0; i < featureLength; i++) {
    const featureData = cloned.map(row => row[i]);
    const min = _.min(featureData);
    const max = _.max(featureData);
    for (var j = 0; j  < cloned.length; j++) {
      cloned[j][i] = (cloned[j][i] - min) / (max - min);
    }
  }

  return cloned;
}
