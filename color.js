let data

function preload(){
  data = loadJSON('colorData.json')
}

let labelList = [
  'red-ish',
  'green-ish',
  'blue-ish',
  'orange-ish',
  'yellow-ish',
  'pink-ish',
  'purple-ish',
  'brown-ish',
  'grey-ish'
]
let model
let xs
let ys
let labelP
let lossP
let vlossP
let accuracyP
let trainStatus

let rSlider, gSlider, bSlider

function setup(){
  // html elements
  labelP = createP('color prediction')
  lossP = createP('loss: 0')
  vlossP = createP('vloss: 0')
  accuracyP = createP('accuracy: 0')
  rSlider = createSlider(0,255,255)
  gSlider = createSlider(0,255,255)
  bSlider = createSlider(0,255,0)

  trainStatus = createP('status: stopped')
  let colors = []
  let labels = []

  for (let record of data.entries){
    let rgb = [record.r, record.g, record.b]
    colors.push(rgb)
    labels.push(labelList.indexOf(record.label))
  }
  // console.log(labelList[labels[0]])
  // console.log(colors[0])
  xs = tf.tensor(colors)
  const labelsTensor = tf.tensor1d(labels, 'int32')
  ys = tf.oneHot(labelsTensor, 9)
  labelsTensor.dispose()

  // labelsTensor.dispose()
  // console.log(labelsTensor.shape)
  // console.log(xs.shape[0] === labelsTensor.shape[0])
  // console.log(labelsTensor.shape)
  
  model = tf.sequential()
  
  // layers
  const hidden = tf.layers.dense({
    units: 16,
    inputShape: [3],
    activation: 'sigmoid'    
  })
  const output = tf.layers.dense({
    units: 9,
    activation: 'softmax',

  })

  model.add(hidden)
  model.add(output)

  const optimizer = tf.train.sgd(0.05)

  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  })

  train().then(result => console.log(result.history.loss))
  
}

async function train(){
  const options = {
    shuffle: true,
    epochs: 20,
    validationSplit: 0.1,
    callbacks: {
      onTrainBegin: () => trainStatus.html('status: training'),
      onTrainEnd: () => trainStatus.html('status: completed'),
      onEpochEnd: (epoch, logs) => {
        console.log(epoch, logs)
        lossP.html('loss: ' + logs.loss.toFixed(2))
        vlossP.html('vloss: ' + logs.val_loss.toFixed(2))
        accuracyP.html('accuracy: ' + logs.acc.toFixed(2))
      },
      onBatchEnd: tf.nextFrame,
    }
  }
  return await model.fit(xs, ys, options)
}

function draw(){
  let r = rSlider.value()
  let g = gSlider.value()
  let b = bSlider.value()
  background(r,g,b)

  tf.tidy(()=> {
    const xs = tf.tensor2d([
      [r/255, g/255, b/255]
    ])
  
    let results = model.predict(xs)
    let index = results.argMax(1).dataSync()[0]
    let label = labelList[index]
    labelP.html(label)
  })
  
  // console.log(label)
}

  


