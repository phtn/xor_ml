
// Machine Learning example
// Dataset = XOR
// Using p5js for data visualization



  
let model
let xs, ys, train_xs, train_ys

function setup(){
  createCanvas(400,400)
  cols = width / resolution
  rows = height / resolution

  // create data
  let inputs = []
  for (let i = 0; i < cols; i++){
    for (let j = 0; j < rows; j++){
      let x1 = i / cols
      let x2 = j / rows
      inputs.push([x1,x2])
    }
  }
  // xor training data
  train_xs = tf.tensor2d([
    [1,1],
    [1,0],
    [0,1],
    [0,0],
  ])
  
  train_ys = tf.tensor([
    [0],
    [1],
    [1],
    [0],
  ])

  // assign xs from inputs loop
  xs = tf.tensor(inputs)
  // declare sequential model
  model = tf.sequential()
  // define hidden layer
  const hidden = tf.layers.dense({
    units: 2,
    inputShape: [2],
    activation: 'sigmoid'
  })
  // define output layer
  const outputs = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
  })
  // add layers
  model.add(hidden) // hidden layer 2 nodes
  model.add(outputs) // output layer 1 node
  // declare optimizer
  const optimizer = tf.train.sgd(0.3)
  // compile model
  model.compile({
    optimizer,
    loss: 'meanSquaredError'    
  })
}
// train model
async function trainModel(){
  return await model.fit(train_xs, train_ys, {
    shuffle: true
  })
}

// draw variables
let resolution = 50
let cols, rows

// p5 draw function
function draw(){
  
  trainModel().then(response => console.log(response.history.loss[0]))

  stroke(254)
  
  // create predictions
  ys = model.predict(xs).dataSync()
  

  // draw data
  let index = 0
  for (let i = 0; i < cols; i++){
    for (let j = 0; j < rows; j++){
      
      fill(ys[index] * 255)
      rect(i * resolution, j * resolution, resolution, resolution)
      index++
    }
  }
  // noLoop()
}