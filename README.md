# ID3 and C4.5 decision tree learning algorithm implementation
Decision Tree Implementation by using ID3 & C4.5 Algorithms and generate F1 scores.

# Test on UCI Machine learning Mushroom Data Set

Getting Started:
	Download the "Project1_N01412075_Resubmission" folder to a local drive. 
	
	This folder has 
	1) Project1_Mushroom_DT_N01412075.py - A file that contains source code for the implementation.
	2) Mushroom folder that has 10 smaller training files(used for cross validation), 1 larger training file (which is a concatenation of all the smaller files) and a final testing file.
	3) A PDF document containing the project report.

Pre-requisites::
	1) Below packages are required for the program to run without issues:
		- pandas
		- numpy
	2) The input files should be used as is. The header information should not be included in the files.
	3) Python 3.7 should be installed on the system on which the application will be used

Running the program:
	1) Execute the 'P1_DT_N01412075_resubmission.py' file
	2) The program should provide output in the below format.

Output:	
	Decision tree implementation using ID3 and C4.3 algorithms for Mushroom data. 

	************************************************************
	Currently ID3 decision tree learning algorithm is running...

	For ID3 and fold_count = 0 :
	    Confusion matrix values: TP = 322 , FP = 0 , TN : 328 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 1 :
	    Confusion matrix values: TP = 327 , FP = 0 , TN : 323 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 2 :
	    Confusion matrix values: TP = 331 , FP = 0 , TN : 319 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 3 :
	    Confusion matrix values: TP = 328 , FP = 0 , TN : 322 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 4 :
	    Confusion matrix values: TP = 361 , FP = 0 , TN : 289 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 5 :
	    Confusion matrix values: TP = 331 , FP = 0 , TN : 319 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 6 :
	    Confusion matrix values: TP = 345 , FP = 0 , TN : 305 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 7 :
	    Confusion matrix values: TP = 350 , FP = 0 , TN : 300 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 8 :
	    Confusion matrix values: TP = 337 , FP = 0 , TN : 313 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For ID3 and fold_count = 9 :
	    Confusion matrix values: TP = 342 , FP = 0 , TN : 308 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0

	The max F1 score for ID3 implementation is 1.0 for the model corresponding to fold_count = 0
	Implementing the selected model with fold_count = 0 , algorithm as ID3 , on the final testing set...
	Below is the decision tree for ID3 algorithm:

	{'odor': {'a': 'e',
		  'c': 'p',
		  'f': 'p',
		  'l': 'e',
		  'm': 'p',
		  'n': {'spore-print-color': {'b': 'e',
					      'h': 'e',
					      'k': 'e',
					      'n': 'e',
					      'o': 'e',
					      'r': 'p',
					      'w': {'habitat': {'d': {'gill-size': {'b': 'e',
										    'n': 'p'}},
								'g': 'e',
								'l': {'cap-color': {'c': 'e',
										    'n': 'e',
										    'w': 'p',
										    'y': 'p'}},
								'p': 'e',
								'w': 'e'}},
					      'y': 'e'}},
		  'p': 'p',
		  's': 'p',
		  'y': 'p'}}

	For ID3 and fold_count = 0 results on the final testing dataset are:
	    Confusion matrix values: TP = 834 , FP = 0 , TN : 790 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	************************************************************
	Currently C4.5 decision tree learning algorithm is running...

	For C4.5 and fold_count = 0 :
	    Confusion matrix values: TP = 322 , FP = 0 , TN : 328 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 1 :
	    Confusion matrix values: TP = 327 , FP = 0 , TN : 323 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 2 :
	    Confusion matrix values: TP = 331 , FP = 0 , TN : 319 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 3 :
	    Confusion matrix values: TP = 328 , FP = 0 , TN : 322 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 4 :
	    Confusion matrix values: TP = 361 , FP = 0 , TN : 289 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 5 :
	    Confusion matrix values: TP = 331 , FP = 0 , TN : 319 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 6 :
	    Confusion matrix values: TP = 345 , FP = 0 , TN : 305 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 7 :
	    Confusion matrix values: TP = 350 , FP = 0 , TN : 300 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 8 :
	    Confusion matrix values: TP = 337 , FP = 0 , TN : 313 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	For C4.5 and fold_count = 9 :
	    Confusion matrix values: TP = 342 , FP = 0 , TN : 308 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0

	The max F1 score for C4.5 implementation is 1.0 for the model corresponding to fold_count = 0
	Implementing the selected model with fold_count = 0 , algorithm as C4.5 , on the final testing set...
	Below is the decision tree for C4.5 algorithm:

	{'odor': {'a': 'e',
		  'c': 'p',
		  'f': 'p',
		  'l': 'e',
		  'm': 'p',
		  'n': {'spore-print-color': {'b': 'e',
					      'h': 'e',
					      'k': 'e',
					      'n': 'e',
					      'o': 'e',
					      'r': 'p',
					      'w': {'veil-color': {'w': {'gill-size': {'b': 'e',
										       'n': {'gill-spacing': {'c': 'p',
													      'w': {'bruises?': {'f': 'e',
																 't': 'p'}}}}}},
								   'y': 'p'}},
					      'y': 'e'}},
		  'p': 'p',
		  's': 'p',
		  'y': 'p'}}

	For C4.5 and fold_count = 0 results on the final testing dataset are:
	    Confusion matrix values: TP = 834 , FP = 0 , TN : 790 , FN : 0
	    Recall = 1.0 , Precision = 1.0 , F1 score =  1.0
	************************* End of program! *************************	
	
	