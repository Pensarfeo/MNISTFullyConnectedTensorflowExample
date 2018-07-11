# MNISTFullyConnectedTensorflowExample

This repo contains the code for the Second meetup of WLML.
In this example we use a fully connected NN to solve the MNIST handwritten numbers classification task.

To view results run `tensorboard --logdir ./tmp` on terminal and follow instructions.

## Files

### Dataset

A utility that imports the MNIST data from remote and pre-processes for use in the tensorflow mode.
In this module defines

 - the size of the training image set `train_size`
 - the size of the testing image set `test_size`
 - the size of the images 'image_size'
 - `feed_dict_gen(BATCH_SIZE[integer], STEP[integer], test[any])`: Gives the next batch according to the number of steps. If no value is given for test it will use the data from the training set else it will use the data from the test set

### utils.py

defines few internal utilities

### trainer.py

Defines the loss, step, optimizer and accuracy subgraphs.

### model.py

Defines the model graph

### main.py

Initializes TF graph, runs training and testing session.
