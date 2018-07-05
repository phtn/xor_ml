function setup(){
  // model
  const model = tf.sequential()
  
  // layers
  const hidden = tf.layers.dense({
    units: 4,
    inputShape: [2],
    activation: 'sigmoid'
  })
   const output = tf.layers.dense({
    units: 2,
    activation: 'sigmoid'
  })

  // add layers to the model
  model.add(hidden)
  model.add(output)
  
  // optimizer
  const sgdOptimizer = tf.train.sgd(0.5)

  // compile model
  model.compile({
    optimizer: sgdOptimizer,
    loss: tf.losses.meanSquaredError
  })

  // declare inputs
  const xs = tf.tensor([
    [0.5, 0.1],
    [0.74, 0.4],
    [0.3, 0.5]
  ])

  const ys = tf.tensor([
    [1, 2],
    [0, 3],
    [0.7, 4]
  ])

   

  // fit the model
  train().then(()=> {
    // predict outputs
    const outputs = model.predict(xs)
    
    console.log('training complete.')
    outputs.print()
  })

  async function train(){
    for (let i = 0; i < 100; i++){
      const response = await model.fit(xs, ys, {
        epochs:100,
        shuffle: true
      })
      console.log(response.history.loss[0])
    }
  }

 
  
}