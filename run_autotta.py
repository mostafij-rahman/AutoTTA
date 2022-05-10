import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF to use only the CPU

import shutil
import time
import numpy as np

import tensorflow as tf
from keras import backend

from child_network import *
from controller import *
from data_loader import *

# Open file to write log

file1 = open('log_vgg_model_run_14.txt', 'a', encoding='utf-8')
if(os.path.isdir('augmented_images/')):
  shutil.rmtree('augmented_images')
os.mkdir('augmented_images')
    
# Load data

(Xtr, ytr), (Xva, yva), (Xts, yts) = get_dataset('cifar10', False) #True
Xva = Xva.astype(np.float32) / 255
Xts = Xts.astype(np.float32) / 255


# Experiment parameters

N_SUBPOLICIES = 4
N_SUBPOLICY_OPS = 2
CONTROLLER_EPOCHS = 0 # 500, 15000 or 20000

CHILD_BATCH_SIZE = 128
CHILD_EPOCHS = 200
CHILD_TRAIN = False
TEST_POLICY = False
SAVE_IMAGES = True

# Generate new data using policy for test time augmentation

def tt_autoaugment(subpolicy, X, y):
    _X = subpolicy((X*255).astype(np.uint8))
    _X = _X.astype(np.float32) / 255
    return _X, y
            
# Get metrics with test time augmentations

def tt_augment_with_policy(child, subpolicies, preds, X, Y, save_images = False):
    preds_ensemble = preds.copy()
    for i, subpolicy in enumerate(subpolicies):
        save_dir = "augmented_images/" + str(i+1)+"/"
        if(os.path.isdir(save_dir)):
          shutil.rmtree(save_dir)
        os.mkdir(save_dir) 
          
        file1.write('# Sub-policy %d\n' % (i+1))
        print('# Sub-policy %d' % (i+1))
        file1.write(str(subpolicy))
        file1.write('\n')
        print(subpolicy)
        X_aug, y_aug = tt_autoaugment(subpolicy, X, Y)
            
        preds_with_policy = child.predict(X_aug)
        preds_ensemble += preds_with_policy
        if(save_images):
          for k in range(len(X_aug)):
              aug_x = PIL.Image.fromarray((X_aug[k]*255).astype(np.uint8))
              aug_x.save(save_dir + "" + str(k) + "_t" + str(np.argmax(Y[k])) + "_p" + str(np.argmax(preds_with_policy[k])) + ".png")
    avg_preds = preds_ensemble/(len(subpolicies)+1)
    return avg_preds
    
def get_accuracy_per_class(Y_pred, Y_true):
    for cls in range(10):
    	predicted_per_class = np.argmax(Y_pred,1)== cls
    	true_per_class = np.argmax(Y_true,1)== cls
    	print([np.sum(true_per_class), np.sum(predicted_per_class)])
    	accuracy_per_class = sum(predicted_per_class*true_per_class)/np.sum(true_per_class)
    	file1.write('-> Child accuracy_per_class: %.5f\n' % (accuracy_per_class))
    	print('-> Child accuracy per class: %.5f' % (accuracy_per_class))
            
if __name__ == '__main__':
    mem_softmaxes = []
    mem_accuracies = []
    
    file1.write('Baseline: \n')
    print('Baseline: ')
    child_baseline = ChildNetwork(Xtr.shape[1:], n_samples = Xtr.shape[0], num_classes = ytr.shape[1])
    if(CHILD_TRAIN):
      child_baseline.fit(Xtr, ytr, Xva, yva, CHILD_EPOCHS, CHILD_BATCH_SIZE)
      child_baseline.model.save_weights('child_models/cifar10vgg_baseline_e' + str(CHILD_EPOCHS) +'.h5')
    child_baseline.model.load_weights('child_models/cifar10vgg_baseline_e60.h5')

    tic = time.time()
    accuracy = child_baseline.evaluate(Xva, yva)
    toc = time.time()

    file1.write('-> Baseline child validation accuracy: %.5f (elaspsed time: %ds)\n' % (accuracy, (toc-tic)))
    print('-> Baseline child validation accuracy: %.5f (elaspsed time: %ds)' % (accuracy, (toc-tic)))
    
    mem_accuracies.append(accuracy)
    
    tic = time.time()
    baseline_test_accuracy = child_baseline.evaluate(Xts, yts)
    toc = time.time()
    
    file1.write('-> Baseline child test accuracy: %.5f (elaspsed time: %ds)\n' % (baseline_test_accuracy, (toc-tic)))
    print('-> Baseline child test accuracy: %.5f (elaspsed time: %ds)' % (baseline_test_accuracy, (toc-tic)))

    print('Controller: ')
    file1.write('Controller: \n')
    controller = Controller(n_subpolicies = N_SUBPOLICIES, n_subpolicy_ops = N_SUBPOLICY_OPS)
    
    print('Baseline validation prediction:')
    file1.write('Baseline validation prediction: \n')
    predictions_val = child_baseline.predict(Xva)
    get_accuracy_per_class(predictions_val, yva)
            
    print('Baseline test prediction:')
    file1.write('Baseline test prediction: \n')
    predictions_test = child_baseline.predict(Xts)
    get_accuracy_per_class(predictions_test, yts)
    
    for epoch in range(CONTROLLER_EPOCHS):
    
        file1.write('Controller: Epoch %d / %d\n' % (epoch+1, CONTROLLER_EPOCHS))
        print('Controller: Epoch %d / %d' % (epoch+1, CONTROLLER_EPOCHS))

        softmaxes, subpolicies = controller.predict(N_SUBPOLICIES)
        mem_softmaxes.append(softmaxes)
        
        predictions_val_copy = predictions_val.copy()
        tic = time.time()
        avg_predictions = tt_augment_with_policy(child_baseline, subpolicies, predictions_val_copy, Xva, yva)
        get_accuracy_per_class(avg_predictions, yva)
        accuracy = sum(np.argmax(avg_predictions,1)==np.argmax(yva,1))/len(avg_predictions)
        toc = time.time()
        file1.write('-> Child accuracy: %.5f (elaspsed time: %ds)\n' % (accuracy, (toc-tic)))
        print('-> Child accuracy: %.5f (elaspsed time: %ds)' % (accuracy, (toc-tic)))

        mem_accuracies.append(accuracy)
        if(np.max(mem_accuracies)==accuracy):
          print('Saving best validation policy.....')
          file1.write('Saving best validation policy.....\n')
          np.save('best_validation_policy.npy', subpolicies)

        if len(mem_softmaxes) > 5:
            # Let some epochs pass, so that the normalization is more robust
            controller.fit(mem_softmaxes, mem_accuracies)
        print()

    print()
    file1.write('Best policies found:\n')
    print('Best policies found:')
    print()

    tic = time.time()
    if(TEST_POLICY):
      _, subpolicies = controller.predict(N_SUBPOLICIES)
      print('Saving best test policy.....')
      file1.write('Saving best test policy.....\n')
      np.save('best_validation_policy.npy', subpolicies)
    else:
      # Load best validation or test policy
      subpolicies = np.load('best_test_policy.npy', allow_pickle=True); # or load best_test_policy.npy

    predictions_test_copy = predictions_test.copy()
    avg_predictions = tt_augment_with_policy(child_baseline, subpolicies, predictions_test_copy, Xts, yts, save_images = SAVE_IMAGES)
    
    # Save new correctly classified images
    
    if(SAVE_IMAGES): 
      save_dir = "augmented_images/new_correct/"
      if(os.path.isdir(save_dir)):
        shutil.rmtree(save_dir)
      os.mkdir(save_dir) 
      for k in range(len(Xts)):
          if((np.argmax(yts[k]) != np.argmax(predictions_test[k]))):
            if((np.argmax(yts[k]) == np.argmax(avg_predictions[k]))):
              x = PIL.Image.fromarray((Xts[k]*255).astype(np.uint8))
              x.save(save_dir+ "" + str(k) + "_t" + str(np.argmax(yts[k])) + "_p" + str(np.argmax(predictions_test[k])) + "_pen" + str(np.argmax(avg_predictions[k])) + ".png")
              
    get_accuracy_per_class(avg_predictions, yts)
    test_accuracy = sum(np.argmax(avg_predictions,1)==np.argmax(yts,1))/len(avg_predictions)
    toc = time.time()
    file1.write('-> Child test accuracy: %.5f (elaspsed time: %ds)\n' % (test_accuracy, (toc-tic)))
    print('-> Child test accuracy: %.5f (elaspsed time: %ds)' % (test_accuracy, (toc-tic)))
    file1.close()
