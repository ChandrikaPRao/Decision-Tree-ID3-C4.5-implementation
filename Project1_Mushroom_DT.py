# -*- coding: utf-8 -*-
"""
This program is a decision tree implementation for ID3 and C4.3 algorithms. The
algorithms are then tested on UCI Mushroom Data Set. For both ID3 and C4.5, the
 program implements a 10-fold cross validation process over the training set,
and picks the best model with the highest F1 score, and runs it
finally on the testing set and reports the Precision,Recall and F1 scores. 
UCI Mushroom Data Set - https://archive.ics.uci.edu/ml/datasets/mushroom
"""
##############################################################################
#Import the necessary python packages
import pandas as pd
import numpy as np
from pprint import pprint

#target_attribute_name will hold the target value/value to be predicted col name
target_attribute_name = "class-label"
  
#features_list list holds the header information of the input file
features_list = ['class-label','cap-shape','cap-surface','cap-color','bruises?',\
                         'odor','gill-attachment','gill-spacing','gill-size',\
                         'gill-color','stalk-shape','stalk-root',\
                         'stalk-surface-above-ring','stalk-surface-below-ring'\
                         ,'stalk-color-above-ring','stalk-color-below-ring',\
                         'veil-type','veil-color','ring-number','ring-type',\
                         'spore-print-color','population','habitat'] 

##############################################################################
def entropy(target_feature):
    """
    Calculates the entropy of a dataset.
    Input:
        1. target_feature: This specifies the target column
    Output:
        1. entropy: Entropy value for the input target feature
    """    
    
    elements,counts = np.unique(target_feature,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])    
    return entropy
##############################################################################
def information_gain(data,split_attribute_name, target_attribute_name):
    """
    Calculates the information gain of a dataset.    
    Input:
        1. data: The dataset for whose feature the information gain is to be 
            calculated
        2. split_attribute_name: the feature for which the information gain is 
            to be calculated
        3. target_attribute_name: name of the target feature(defined globally)
    Output:
        1. Information_Gain: information gain of a dataset w.r.t. a feature
    """    

    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_attribute_name])  

    #Calculate the values and the corresponding counts for the split attribute 
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    
    #Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data[data[split_attribute_name]==vals[i]][target_attribute_name]) for i in range(len(vals))])
    
    #Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain    
##############################################################################    
def gain_ratio(data,split_attribute_name, target_attribute_name):
    """
    Calculates the gain ratio of a dataset.
    Input:
        1. data: The dataset for whose feature the information gain is to be
            calculated
        2. split_attribute_name: the feature for which the information gain is
            to be calculated
        3. target_attribute_name: name of the target feature(defined globally)
    Output:
        1. gain_ratio_val: gain ratio of a dataset w.r.t. a feature
    """       
    #Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_attribute_name])  

     
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    sum_of_counts = sum(counts)
    
    #Calculate the weighted entropy    
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data[data[split_attribute_name]==vals[i]][target_attribute_name]) for i in range(len(vals))])

    #Calculate the IV(intrinsic value) required for the gain ratio calculation
    intrinsic_value = 0
    for y in range(len(vals)):
        i1 = int(counts[y]) / sum_of_counts
        intrinsic_value = -(i1 * np.log2(i1)) + intrinsic_value
    
    #Calculate the gain ratio
    Information_Gain = total_entropy - Weighted_Entropy
    if (intrinsic_value == 0):
        gain_ratio_val = Information_Gain
    else:
        gain_ratio_val = Information_Gain / intrinsic_value     
#    print("Gain_ratio_matrix = ",gain_ratio_val)
    return round(gain_ratio_val, 3)  
##############################################################################   
def create_decision_tree(data,originaldata,features,target_attribute_name, algorithm, parent_node_class = None):     
    """
    This function creates a decision tree taking the input data("data" 
    parameter) following the algorithm specified("algorithm" parameter).
    Input:
        1. data: the training data for which the decision tree algorithm 
        should be run
        2. originaldata: This is the original training dataset
        3. features: feature space of the dataset. This is needed for the 
            recursive call since during the tree growing process we have to 
            remove features from our dataset i.e; splitting at each node
        4. target_attribute_name = name of the target attribute
        5. parent_node_class = This is the value or PLURALITY-VALUE for parent
            examples
    Output:
        1. tree: decision tree created by the specified algorithm(ID3 or C4.3
            in our case)
    """  
    
    #Define the stopping criteria
    #If all examples have the same classification then return the classification
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #If the dataset is empty, return the PLURALITY-VALUE / mode target feature 
    #value in the original dataset
    elif len(data)==0:
        empty_dataset_val = np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
#        print("Dataset is empty, hence returning - ",empty_dataset_val)
        return empty_dataset_val
    
    #If the feature space is empty, return the mode target feature value of the
    #direct parent node(parent_node_class)
    elif len(features) ==0:
        return parent_node_class
    
    #If none of the above is true, grow the tree.
    else:    
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        if (algorithm == "ID3"):
            info_gain_or_ratio_matrix = [information_gain(data, feature, target_attribute_name) for feature in features]
        elif (algorithm == "C4.5"):
            info_gain_or_ratio_matrix = [gain_ratio(data, feature, target_attribute_name) for feature in features]
        
        #Select the feature which best splits the dataset on basis of 
        #information gain for ID3 and gain ration for C4.3
        best_feature_index = np.argmax(info_gain_or_ratio_matrix)
        best_feature = features[best_feature_index]
#        print("best_feature is",best_feature)
    
        #Create the tree structure using tuple data structure. The root is the
        #best_feature, calculated in the previous steps.
        tree = {best_feature:{}}
        
        #Remove the best_feature from the feature space
        features = [i for i in features if i != best_feature]    
    
        #Grow a branch under the root node for each possible value of the root
        #node feature
        for value in np.unique(data[best_feature]):
            value = value
            
            #Split the dataset along the value of the feature with the largest 
            #information gain or gain ration and therwith create sub_datasets
            sub_data = data[data[best_feature] == value]
            
            #Recursion step: calls the create_decision_tree algorithm for each 
            #of those sub_datasets with the new parameters
            subtree = create_decision_tree(sub_data,originaldata,features,target_attribute_name,algorithm,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree      
            
        return(tree)  
############################################################################## 
def predict(test_data,tree,default = 'p'):
    """
    Predicts target value for a new/unseen test data instance.
    Input:
        1. test_data: test data sample for which a target value is to be
            predicted
        2. tree: Decision tree that is to be considered for coming up with the 
            prediction.
    Output:
        1. result: Predicted value for the input test data.    
    """ 
    for key in list(test_data.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][test_data[key]] 
            except:
                return default
            result = tree[key][test_data[key]]
            
            if isinstance(result,dict):
                return predict(test_data,result)
            else:
                return result                  
##############################################################################       
def test(data,tree):
    """
    This function runs the test data("data" parameter) on the decision tree & 
    outputs the performance measures - precision, recall, F1_score,TP,FP,TN,FN
    Input:
        1. data: test data for which target values are to be predicted
        2. tree: The decision tree using which the target values are to be
            predicted
    Output:
        1. precision: It is the fraction of relevant instances among the retrieved instances
        2. recall: It is the fraction of the total amount of relevant instances that were actually retrieved
        3. F1_score: It is a measure of a test's accuracy
        4. TP: Count of True positive values on comparing predicted values against actual values
        5. FP: Count of False positive values on comparing predicted values against actual values
        6. TN: Count of True negative values on comparing predicted values against actual values
        7. FN: Count of False negative values on comparing predicted values against actual values      
    """
    
    #Create new test data instances by simply removing the target feature 
    #column from the original dataset and convert it to a dict
    global target_attribute_name
    queries = data.iloc[:,1:].to_dict(orient = "records")
    
    #Create a empty list in whose columns the prediction of the tree are stored
    predicted_list = []
    actual_list = []
    for i in range(len(data)):
        predicted_val = predict(queries[i],tree,1.0) 
        predicted_list.append(predicted_val)
    
    # actual_list is a list containing the actual values of the target feature
    actual_list = data[target_attribute_name].tolist()

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(predicted_list)): 
        if actual_list[i]==predicted_list[i]=='e':
           TP += 1
        if predicted_list[i]=='e' and actual_list[i]=='p':
           FP += 1
        if actual_list[i]==predicted_list[i]=='p':
           TN += 1
        if predicted_list[i]=='p' and actual_list[i]=='e':
           FN += 1

    precision = round(TP / (TP + FP),3)
    recall = round(TP / (TP + FN),3)
    F1_score = round(2 * (precision * recall) / (precision + recall),3)    
    
    return (precision, recall, F1_score,TP,FP,TN,FN)
##############################################################################
def main():
    print("Decision tree implementation using ID3 and C4.3 algorithms for", \
              "Mushroom data.","\n")
    target_attribute_name = "class-label"

    #Import the training dataset
    file_names = ['training_aa.data','training_ab.data','training_ac.data',
                  'training_ad.data','training_ae.data','training_af.data',
                  'training_ag.data','training_ah.data','training_ai.data',
                  'training_aj.data']     
    training_data = pd.read_csv('Mushroom/training.data',
                      names=features_list)   
    final_testing_data = pd.read_csv('Mushroom/testing.data',
                      names=features_list)     
    
    #Algorithms for the decision tree creation
    algorithms_to_test = ["ID3","C4.5"]
    
    # For cross validation purpose n is set to 10
    n = 10

    #For each algorithm decision tree is created using the training data and
    #eventually run on the testing set to calculate the performance measures.
    #10-fold cross validation is performed using the 10 files that were 
    #provided.A best model is selected on basis of the max F1 score. This model
    #is then run on the final testing dataset.
    for algorithm_under_test in algorithms_to_test: 
        print("************************************************************")
        print("Currently",algorithm_under_test,"decision tree learning algorithm is running...\n")
        F1_score_list = []
        for fold_count in range(0,n):
            test_file_name = []
            training_file_names = []
            for file_count in range(0,n):
                if (fold_count == file_count):
                    test_file_name.append(file_names[file_count])
                else:
                    training_file_names.append(file_names[file_count])
            
            merged_train_data = pd.concat([pd.read_csv('Mushroom/'+f,names=features_list) for f in training_file_names], sort='False')
            test_data = pd.read_csv('Mushroom/'+test_file_name[0],names=features_list)

            dt = create_decision_tree(merged_train_data, training_data, features_list[1:],target_attribute_name,algorithm_under_test)  
            test_perf_measures = test(test_data,dt)   
            
            precision = test_perf_measures[0]
            recall = test_perf_measures[1]
            F1_score = test_perf_measures[2]
            TruPos = test_perf_measures[3]
            FalPos = test_perf_measures[4]
            TruNeg = test_perf_measures[5]
            FalNeg = test_perf_measures[6]
            
            F1_score_list.append(F1_score)
            
            print("For",algorithm_under_test,"and fold_count =",fold_count,":")            
            print("    Confusion matrix values:","TP =",TruPos,", FP =",FalPos,\
                                              ", TN :",TruNeg,", FN :",FalNeg)
            print('    Recall = ' + str(recall),', Precision = ' + \
                  str(precision),", F1 score = ",F1_score)
            
        max_F1_score = max(F1_score_list)
        max_F1_foldcount = F1_score_list.index(max_F1_score)
        print("\nThe max F1 score for",algorithm_under_test,"implementation is"\
              ,max_F1_score,"for the model corresponding to fold_count =",\
              max_F1_foldcount)   
        print("Implementing the selected model with fold_count ="\
              ,max_F1_foldcount,",algorithm =",algorithm_under_test\
              ,",on the final testing set...")
        
        training_file_names = []
        test_data = pd.read_csv('Mushroom/testing.data', names=features_list)  

        for file_count in range(0,n):
            if (max_F1_foldcount != file_count):
                training_file_names.append(file_names[file_count])
        merged_train_data = pd.concat([pd.read_csv('Mushroom/'+f,names=features_list) for f in training_file_names], sort='False')
        
           
        dt = create_decision_tree(merged_train_data, training_data, features_list[1:],target_attribute_name,algorithm_under_test)  
        
        print("Below is the decision tree for",algorithm_under_test,"algorithm:\n")
        pprint(dt)
#        precision, recall, F1_score = test(final_testing_data,dt) 
      
        test_perf_measures = test(final_testing_data,dt)   
        
        precision = test_perf_measures[0]
        recall = test_perf_measures[1]
        F1_score = test_perf_measures[2]
        TruPos = test_perf_measures[3]
        FalPos = test_perf_measures[4]
        TruNeg = test_perf_measures[5]
        FalNeg = test_perf_measures[6]        
        
        
        print("\nFor",algorithm_under_test,"and fold_count =",max_F1_foldcount,\
                                  "results on the final testing dataset are:")            
        print("    Confusion matrix values:","TP =",TruPos,", FP =",FalPos,\
                                              ", TN :",TruNeg,", FN :",FalNeg)        
        print('    Recall = ' + str(recall),', Precision = ' + \
              str(precision),", F1 score = ",F1_score)
    print("\n************************* End of program! *************************")

if __name__ == '__main__':
    main()  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    