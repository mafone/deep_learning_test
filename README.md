# deep_learning_test
First install some deppendencies like csv, sys, numpy, cv2, PIL, glob and keras.

Then run: CIFAR10.py <train_folder> <test_folder> destination_folder>, 
                                  destination_folder is where the outputs will be sent;
                                  
Then use: eval.py <truth.csv> <test.preds.csv> to evaluale <test.preds.csv> 
                                              according to the ground truth <truth.csv>.
