
import time
import shutil
from six.moves import xrange
import os
import tensorflow as tf
import time

# show how much time has ellapsed
def epoch_tracker(step, prevTime, bs_ts, EVAL_FREQUENCY):
  # bs_ts = BATCH_SIZE / train_size
  elapsed_time = time.time() - prevTime
  prevTime = time.time()
  print('Step %d (epoch %.2f), %.1f ms' %(step, float(step) * bs_ts, 1000 * elapsed_time / EVAL_FREQUENCY))
  return prevTime

# 
def get_mislabeled_cases(predictions, labels, data, step, BATCH_SIZE):
  """show misclassified cases and output image"""
  misclassifiedClasses = []
  for i in xrange(0, predictions.size):
    if (predictions[i]!=labels[i]):
      newMiss = [predictions[i], labels[i], (step * BATCH_SIZE)+i]
      misclassifiedClasses = newMiss
  return misclassifiedClasses

def fresh_log_writer(*paths):
  writer_dir = os.path.join(*paths)
  return tf.summary.FileWriter(writer_dir)

def pprint(a, *b, **all):
  """just a more organized printing"""
  print('')
  print(a)
  for x in b:
    print(x)
  print('#############################')