def rfPredictionIntervals(rf, xTrain, yTrain, xVal, percentile=90):
    """
    Defind the function to get the confidence interval, by interrogating the individual trees in the RF, see:https://blog.datadive.net/prediction-intervals-for-random-forests/

    Parameters
    ----------
    rf : sklearn.ensemble.RandomForestRegressor
        The random forest model to be evaluated 

    xTrain : pandas.DataFrame
        The features of the training set

    yTrain : pandas.DataFrame
        The target of the training set

    xVal : pandas.DataFrame
        The features of the validation set - to be predicted
        
    percentile : float, optional
        The percentile to be used for the confidence interval. The default is 90.

    Returns
    -------
    y_mean : pandas.Series
        The mean of the predictions from the random forest model

    y_uncert : pandas.Series
        The uncertainty of the predictions from the random forest model

    y_lower : pandas.Series
        The lower bound of the confidence interval

    y_upper : pandas.Series
        The upper bound of the confidence interval
    """

    # import dependencies
    import numpy as np
    import pandas as pd

    # fit the model
    rf_fitted = rf.fit(xTrain, yTrain)
    # initialize a list to hold the predictions from each tree
    y_preds = []
    # loop through the trees in the random forest
    for tree in rf_fitted.estimators_:
        # get the predictions from each tree
        y_pred = tree.predict(xVal.values)
        # append the predictions to the list
        y_preds.append(y_pred)
    # Convert to np.array by stacking list of arrays along the column axis with each column being the prediction from a different tree
    y_preds = np.stack(y_preds, axis=1)           
    # get the quantiles for the confidence interval
    q_down = (100 - percentile) / 2.
    q_up = 100 - q_down

    # get the mean, uncertainty, lower bound, and upper bound
    y_lower = pd.Series(np.percentile(y_preds, q_down, axis=1),index=xVal.index)
    y_upper = pd.Series(np.percentile(y_preds, q_up, axis=1)  ,index=xVal.index)  
    y_mean = pd.Series(rf_fitted.predict(xVal) ,index=xVal.index)  
    y_uncert = pd.Series(y_upper - y_lower,index=xVal.index)
    
    return y_mean, y_uncert, y_lower, y_upper

def splitTrainVal (xTrainVal, yTrainVal, lstTrainIndices):
    '''
    Split the training+validation set (non-holdout) into training and validation sets

    Parameters
    ----------
    xTrainVal : pandas.DataFrame
        Features of the training+validation set.

    yTrainVal : pandas.DataFrame
        Target of the training+validation set.

    lstTrainIndices : list
        List of indices to use for the training set.

    Returns
    -------
    xTrain : pandas.DataFrame
        Features of the training set.

    xVal : pandas.DataFrame
        Features of the validation set.

    yTrain : pandas.DataFrame
        Target of the training set.

    yVal : pandas.DataFrame
        Target of the validation set.
    '''

    # import dependencies
    import pandas as pd

    # subsample the TrainVal set down to the training set based on the indicies passed
    xTrain = xTrainVal.loc[lstTrainIndices]
    yTrain = yTrainVal.loc[lstTrainIndices]

    # get the validation set 
    xVal = xTrainVal.drop(lstTrainIndices)
    yVal = yTrainVal.drop(lstTrainIndices)

    return xTrain, xVal, yTrain, yVal

def grow_random (model1, model2,
                 xTrainVal, yTrainVal,
                 xTest, yTest, 
                 strSaveDir = '',
                 strSaveName = 'growing_random',
                 strModel1Name = 'model1',
                 strModel2Name = 'model2',
                 fltStepFrac = 0.01,
                 fltTrainFracStop = 1.0,
                 intRandomSeed = 0):
    '''
    Grow the training set from a randomly selected stepfrac number of points up to the full training+validation 
    set by iterativly adding a stepfrac number of points, which are selected from the validation set at random

    Parameters
    ----------
    model1 : regressor
        The first model to be evaluated

    model2 : regressor
        The second model to be evaluated

    xTrainVal : pandas.DataFrame
        Features of the training+validation set
        shape = (n_samples, n_features)

    yTrainVal : pandas.Series
        Target of the training+validation set
        shape = (n_samples,)

    xTest : pandas.DataFrame
        Features of the test set
        shape = (n_samples, n_features)

    yTest : pandas.Series
        Target of the test set
        shape = (n_samples,)

    strSaveDir : str, optional
        The path to save the results to. The default is '' which means the results will be saved in the same
        directory as the script.

    strSaveName : str, optional
        The name of the file to save the results to. The default is 'growing_xgbMaxUncertainty_ibug'.

    strModel1Name : str, optional
        The name of the first model. The default is 'model1'.

    strModel2Name : str, optional
        The name of the second model. The default is 'model2'.

    fltStepFrac : float, optional
        The fraction of the training+validation set to add to the training set at each step. The default is 0.01
        which means that the training set will grow by 1% at each step.

    fltTrainFracStop : float, optional
        The fraction of the training+validation set to stop growing the training set at. The default is 1.0,
        which means that the training set will grow to the full training+validation set.
    
    intRandomSeed : int, optional
        The random state to use for the random selection of points from the validation set. The default is 0.

    Returns
    -------
    dfGrowing_random_model1: pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using model1 on the test set

    dfGrowing_random_model2: pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using model2 on the test set

    '''

    # import dependencies
    import random
    import numpy as np
    import pandas as pd
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME 
    import time

    # check that the path given ends with a '/'
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # initialize dataframe to hold results
    # will have 'train_ratio', 'maes', 'rmse','r2', 'train_size' columns
    dfGrowing_random_model1 = pd.DataFrame()
    dfGrowing_random_model2 = pd.DataFrame()

    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # initialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # initialize the step size
    intStepSize = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()

    while intTempTrainSize < (len(xTrainVal)+1):
        # start time
        timeStart = time.time()
        # --- CHECK STOP CRITERIA ---
        # check if the training set has met the stop criteria
        if intTempTrainSize >= len(xTrainVal)*fltTrainFracStop:
            # if so, break the loop
            break

        # --- SUBSAMPLE THE TRAININGVAL SET ---
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal, lstTrainIndices)

        # --- CALCULATE, PRINT AND STORE RESULTS ---
        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print('Training set percentage: ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # get scores using function for the first model
        maes_1, rmse_1, r2_1 = get_scores(model1,
                                          xTrain_temp,yTrain_temp,
                                          xTest,yTest)
        
        # get scores using function for the second model
        maes_2, rmse_2, r2_2 = get_scores(model2,
                                          xTrain_temp,yTrain_temp,
                                          xTest,yTest)
        
        # store the results for the first model predictions
        dfGrowing_random_model1 = pd.concat([dfGrowing_random_model1,
                                             pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                           'maes': [round(maes_1,4)],
                                                           'rmse': [round(rmse_1,4)],
                                                           'r2': [round(r2_1,4)],
                                                           'train_size': [len(xTrain_temp)]})],
                                             ignore_index=True)
        # store the results for the second model predictions
        dfGrowing_random_model2 = pd.concat([dfGrowing_random_model2,
                                             pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                           'maes': [round(maes_2,4)],
                                                           'rmse': [round(rmse_2,4)],
                                                           'r2': [round(r2_2, 4)],
                                                           'train_size': [len(xTrain_temp)]})],
                                             ignore_index=True)
        
        # --- SAVE THE RESULTS ---
        # save the results to a csv file
        dfGrowing_random_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
        dfGrowing_random_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
        # make a dataframe with the indices of the training set
        dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
        # save the indices of the training set to a csv file
        dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')

        # --- GROWING CRITERIA ---
        # check if the training set is the full training+validation set
        if intTempTrainSize == len(xTrainVal):
            # if so, break the loop
            intTempTrainSize+=10
            break

        if intTempTrainSize + intStepSize > len(xTrainVal):
            # if the next step would be larger than the full training+validation set, set the step size to the
            # difference between the current training set size and the full training+validation set size
            intStepSize = len(xTrainVal) - intTempTrainSize
    
        # randomly select the next stepfrac indices from xVal_temp
        lstTempIndices = np.random.choice(xVal_temp.index, intStepSize, replace=False).tolist()

        # --- UPDATE THE TRAINING SET SIZE ---
        # add the indices to the list of training indices
        lstTrainIndices = lstTrainIndices + lstTempIndices
        # update the training set size
        intTempTrainSize = len(lstTrainIndices)

        # end time
        timeEnd = time.time()
        # print the time it took to run the loop
        print('Time to run loop: ', round(timeEnd-timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')

    # --- SAVE THE RESULTS ---
    # save the results to a csv file
    dfGrowing_random_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
    dfGrowing_random_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
    # make a dataframe with the indices of the training set
    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    # save the indices of the training set to a csv file
    dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
    return

def grow_rfMaxUncertainty (model1, model2,
                           xTrainVal, yTrainVal,
                           xTest, yTest,
                           strSaveDir = '',
                           strSaveName = 'growing_rfMaxUncertainty',
                           strModel1Name = 'model1',
                           strModel2Name = 'model2',
                           fltStepFrac = 0.01,
                           fltTrainFracStop = 1.0,
                           intRandomSeed = 0):
    '''
    Grow the training set from a randomly selected stepfrac number of points up to the full training+validation 
    set by iterativly adding a stepfrac number of points, which are selected from the validation set with the
    criteria that they have the highest rf uncertainty

    Parameters
    ----------
    model1 : sklearn.ensemble.RandomForestRegressor
        The random forest model to be evaluated

    model2 : regressor
        The second model to be evaluated

    xTrainVal : pandas.DataFrame
        Features of the training+validation set
        shape = (n_samples, n_features)

    yTrainVal : pandas.Series
        Target of the training+validation set
        shape = (n_samples,)

    xTest : pandas.DataFrame
        Features of the test set
        shape = (n_samples, n_features)

    yTest : pandas.Series
        Target of the test set
        shape = (n_samples,)

    strSaveDir : str, optional
        The path to save the results to. The default is '' which means the results will be saved in the same
        directory as the script.

    strSaveName : str, optional
        The name of the file to save the results to. The default is 'growing_xgbMaxUncertainty_ibug'.

    strModel1Name : str, optional
        The name of the first model. The default is 'model1'.

    strModel2Name : str, optional
        The name of the second model. The default is 'model2'.

    fltStepFrac : float, optional
        The fraction of the training+validation set to add to the training set at each step. The default is 0.01
        which means that the training set will grow by 1% at each step.

    fltTrainFracStop : float, optional
        The fraction of the training+validation set to stop growing the training set at. The default is 1.0,
        which means that the training set will grow to the full training+validation set.

    intRandomSeed : int, optional
        The random state to use for the random selection of points from the validation set. The default is 0.

    Returns
    -------
    dfGrowing_rfMaxUncertainty_model1: pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using the model1 on the test set

    dfGrowing_rfMaxUncertainty_model2: pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using the model2 on the test set
    '''

    # import dependencies
    import random
    import numpy as np
    import pandas as pd
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME
    import time

    # check that the path given ends with a '/'
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # initialize dataframe to hold results
    # will have 'train_ratio', 'maes', 'rmse','r2', 'train_size' columns
    dfGrowing_rfMaxUncertainty_model1 = pd.DataFrame()
    dfGrowing_rfMaxUncertainty_model2 = pd.DataFrame()

    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # intialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # intialize the step size
    intStepSize = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()

    while intTempTrainSize < (len(xTrainVal)+1):
        # start time
        timeStart = time.time()
        # --- CHECK STOP CRITERIA ---
        # check if the training set has met the stop criteria
        if intTempTrainSize >= len(xTrainVal)*fltTrainFracStop:
            # if so, break the loop
            break
        # --- SUBSAMPLE THE TRAININGVAL SET ---
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal, lstTrainIndices)

        # --- CALCULATE, PRINT AND STORE RESULTS ---
        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print('Training set percentage: ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # get scores using function for the random forest model
        maes_1, rmse_1, r2_1 = get_scores(model1,
                                          xTrain_temp,yTrain_temp,
                                          xTest,yTest)
        # get scores using function for the extra model
        maes_2, rmse_2, r2_2 = get_scores(model2,
                                          xTrain_temp,yTrain_temp,
                                          xTest,yTest)
        
        # store the results for the random forest model predictions
        dfGrowing_rfMaxUncertainty_model1 = pd.concat([dfGrowing_rfMaxUncertainty_model1,
                                                       pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                                     'maes': [round(maes_1, 4)],
                                                                     'rmse': [round(rmse_1, 4)],
                                                                     'r2': [round(r2_1, 4)],
                                                                     'train_size': [len(xTrain_temp)]})], 
                                                      ignore_index=True)
        # store the results for the extra model predictions
        dfGrowing_rfMaxUncertainty_model2 = pd.concat([dfGrowing_rfMaxUncertainty_model2,
                                                       pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                                     'maes': [round(maes_2, 4)],
                                                                     'rmse': [round(rmse_2, 4)],
                                                                     'r2': [round(r2_2, 4)],
                                                                     'train_size': [len(xTrain_temp)]})],
                                                       ignore_index=True)
        # --- SAVE THE RESULTS ---
        # save the results to a csv file
        dfGrowing_rfMaxUncertainty_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
        dfGrowing_rfMaxUncertainty_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
        # make a dataframe with the indices of the training set
        dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
        # save the indices of the training set to a csv file
        dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
         
        # --- GROWING CRITERIA ---
        # check if the training set is the full training+validation set
        if intTempTrainSize == len(xTrainVal):
            # if so, break the loop
            intTempTrainSize+=10
            break

        # use rfPredictionIntervals to calculate the uncertainty of the validation set
        _, yUncert_val, _, _ = rfPredictionIntervals(model1, xTrain_temp, yTrain_temp, xVal_temp)
        # get the indices of the intStepSize points with the highest uncertainty
        lstMaxUncertIndices = yUncert_val.nlargest(intStepSize).index.tolist()

        # --- UPDATE THE TRAINING SET ---
        # add the indices to the training set
        lstTrainIndices = lstTrainIndices + lstMaxUncertIndices
        # update the size of the training set
        intTempTrainSize = len(lstTrainIndices)

        # end time
        timeEnd = time.time()

        # print the time it took to grow the training set
        print('Time to grow training set: ', round(timeEnd-timeStart, 2), ' seconds', flush=True)
        print('--------------------------------------------------------------------------------------------\n')

    # --- SAVE THE RESULTS ---
    # save the results to a csv file
    dfGrowing_rfMaxUncertainty_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
    dfGrowing_rfMaxUncertainty_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
    # make a dataframe with the indices of the training set
    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    # save the indices of the training set to a csv file
    dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
    return

def grow_QBC (model1, model2,
              xTrainVal, yTrainVal,
              xTest, yTest,
              strSaveDir = '',
              strSaveName = 'growing_QBC',
              strModel1Name = 'model1',
              strModel2Name = 'model2',
              fltStepFrac = 0.01,
              fltTrainFracStop = 1.0,
              intRandomSeed = 0):
    '''
    Grow the training set from a randomly selected stepfrac number of points up to the full training+validation 
    set by iterativly adding a stepfrac number of points, which are selected from the validation set by querying 
    the 'committee' of model1 and model2 for the points with the highest difference in the two model's predictions

    Parameters
    ----------
    model1 : regressor
        The first model to be evaluated

    model2 : regressor
        The second model to be evaluated

    xTrainVal : pandas.DataFrame
        Features of the training+validation set
        shape = (n_samples, n_features)

    yTrainVal : pandas.Series
        Target of the training+validation set
        shape = (n_samples,)

    xTest : pandas.DataFrame
        Features of the test set
        shape = (n_samples, n_features)

    yTest : pandas.Series
        Target of the test set
        shape = (n_samples,)

    strSaveDir : str, optional
        The path to save the results to. The default is '' which means the results will be saved in the same
        directory as the script.

    strSaveName : str, optional
        The name of the file to save the results to. The default is 'growing_xgbMaxUncertainty_ibug'.

    strModel1Name : str, optional
        The name of the first model. The default is 'model1'.

    strModel2Name : str, optional
        The name of the second model. The default is 'model2'.

    fltStepFrac : float, optional
        The fraction of the training+validation set to add to the training set at each step. The default is 0.01
        which means that the training set will grow by 1% at each step.

    fltTrainFracStop : float, optional
        The fraction of the training+validation set to stop growing the training set at. The default is 1.0,
        which means that the training set will grow to the full training+validation set.
    
    intRandomSeed : int, optional
        The random seed to use. The default is 0.

    Returns
    -------
    dfGrowing_QBC_model1 : pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using the model1 on the test set

    dfGrowing_QBC_model2 : pandas.DataFrame
        Dataframe containing the results of the growing training set, evaluated using the model2 on the test set
    '''

    # import dependencies
    import random
    import numpy as np
    import pandas as pd
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME
    import time 

    # check that the path given ends with a '/'
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # initialize dataframe to hold results
    # will have 'train_ratio', 'maes', 'rmse','r2', 'train_size' columns
    dfGrowing_QBC_model1 = pd.DataFrame()
    dfGrowing_QBC_model2 = pd.DataFrame()

    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # intialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # intialize the step size
    intStepSize = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()

    while intTempTrainSize < (len(xTrainVal)+1):
        # start time
        timeStart = time.time()
        # --- CHECK STOP CRITERIA ---
        # check if the training set has met the stop criteria
        if intTempTrainSize >= len(xTrainVal)*fltTrainFracStop:
            # if so, break the loop
            break

        # --- SUBSAMPLE THE TRAININGVAL SET ---
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal, lstTrainIndices)

        # --- CALCULATE, PRINT AND STORE RESULTS ---
        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print('Training set percentage: ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # get scores using function for the first model
        maes_1, rmse_1, r2_1 = get_scores(model1,
                                          xTrain_temp,yTrain_temp,
                                          xTest,yTest)
        
        # get scores using function for the second model
        maes_2, rmse_2, r2_2 = get_scores(model2,
                                          xTrain_temp,yTrain_temp,
                                          xTest,yTest)
        
        # store the results for the first model predictions
        dfGrowing_QBC_model1 = pd.concat([dfGrowing_QBC_model1,
                                          pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                        'maes': [round(maes_1,4)],
                                                        'rmse': [round(rmse_1,4)],
                                                        'r2': [round(r2_1,4)],
                                                        'train_size': [len(xTrain_temp)]})],
                                          ignore_index=True)
        # store the results for the second model predictions
        dfGrowing_QBC_model2 = pd.concat([dfGrowing_QBC_model2,
                                          pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                        'maes': [round(maes_2,4)],
                                                        'rmse': [round(rmse_2,4)],
                                                        'r2': [round(r2_2, 4)],
                                                        'train_size': [len(xTrain_temp)]})],
                                          ignore_index=True)
        # --- SAVE THE RESULTS ---
        # save the results to a csv file
        dfGrowing_QBC_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
        dfGrowing_QBC_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
        # make a dataframe with the indices of the training set
        dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
        # save the indices of the training set to a csv file
        dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
        
        # --- GROWING CRITERIA ---
        # check if the training set is the full training+validation set
        if intTempTrainSize == len(xTrainVal):
            # if so, break the loop
            intTempTrainSize+=10
            break

        # get the predictions of model 1 and 2 on the validation set
        xVal_temp_pred1 = model1.predict(xVal_temp)
        xVal_temp_pred2 = model2.predict(xVal_temp)
        # get the difference between the two models' predictions
        yVal_temp_diff = abs(xVal_temp_pred1 - xVal_temp_pred2)
        yVal_temp_diff = pd.Series(yVal_temp_diff, index = yVal_temp.index) 
        # get the indices of the intStepSize points with the highest difference
        lstMaxDiffIndices = yVal_temp_diff.nlargest(intStepSize).index.tolist()

        # --- UPDATE THE TRAINING SET ---
        # add the indices to the training set
        lstTrainIndices = lstTrainIndices + lstMaxDiffIndices
        # update the size of the training set
        intTempTrainSize = len(lstTrainIndices)

        # end time
        timeEnd = time.time()
        # print the time it took to run the loop
        print('Time to run loop: ', round(timeEnd-timeStart, 2), ' seconds', flush=True)
        print('\n--------------------------------------------------------------------------------------------')

    # --- SAVE THE RESULTS ---
    # save the results to a csv file
    dfGrowing_QBC_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
    dfGrowing_QBC_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
    # make a dataframe with the indices of the training set
    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    # save the indices of the training set to a csv file
    dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
    return

def grow_xgbMaxUncertainty_ibug (xgbModel4Uncert,
                                 model1, model2,
                                 xTrainVal, yTrainVal,
                                 xTest, yTest,
                                 strSaveDir = '',
                                 strSaveName = 'growing_xgbMaxUncertainty_ibug',
                                 strModel1Name = 'model1',
                                 strModel2Name = 'model2',
                                 fltStepFrac=0.01,
                                 fltTrainFracStop = 1.0,
                                 fltTreeSubsample = 1.0,
                                 intRandomSeed = 0):
    '''
    Grow the training set from a randomly selected stepfrac number of points up to the full training+validation 
    set by iterativly adding a stepfrac number of points, which are selected from the validation set with the
    criteria that they have the highest xgb uncertainty
        - predict the uncertainty of the xgb model with ibug

    Parameters
    ----------
    xgbModel4Uncert : regressor.xgb
        XGBoost regression model to be used to calculate the uncertainty of the predictions 
        (preferably numParallelTree=1 for speed)

    model1 : regressor
        The first model to be evaluated

    model2 : regressor
        The second model to be evaluated

    xTrainVal : pandas.DataFrame
        Features of the training+validation set
        shape = (n_samples, n_features)

    yTrainVal : pandas.Series
        Target of the training+validation set
        shape = (n_samples,)

    xTest : pandas.DataFrame
        Features of the test set
        shape = (n_samples, n_features)

    yTest : pandas.Series
        Target of the test set
        shape = (n_samples,)

    strSaveDir : str, optional
        The path to save the results to. The default is '' which means the results will be saved in the same
        directory as the script.

    strSaveName : str, optional
        The name of the file to save the results to. The default is 'growing_xgbMaxUncertainty_ibug'.

    strModel1Name : str, optional
        The name of the first model. The default is 'model1'.

    strModel2Name : str, optional
        The name of the second model. The default is 'model2'.

    fltStepFrac : float, optional
        The fraction of the training+validation set to add to the training set at each step. The default is 0.01
        which means that the training set will grow by 1% at each step.

    fltTrainFracStop : float, optional
        The fraction of the training+validation set to stop growing the training set at. The default is 1.0,
        which means that the training set will grow to the full training+validation set.

    fltTreeSubsample : float, optional
        The fraction of the trees to use for the uncertainty calculation. The default is 1.0, which means that
        all trees will be used.

    intRandomSeed : int, optional
        The random seed to use for the initial random selection of the training set. The default is 0.

    Returns
    -------
    dfGrowing_xgbMaxUncertainty_model1 : pandas.DataFrame
        Dataframe containing the results of the growing the training set, evaluating model1 on the test set

    dfGrowing_xgbMaxUncertainty_model2 : pandas.DataFrame
        Dataframe containing the results of the growing the training set, evaluating model2 on the test set

    '''
    # import libraries
    import numpy as np
    import pandas as pd
    import random
    from myfunc import get_scores                                               # CHANGE DEPENDING ON KANGMING'S FILE NAME
    from ibug import IBUGWrapper                                                    # v0.0.9
    import time

    # check that the path given ends with a '/'
    if strSaveDir != '':
        if strSaveDir[-1] != '/':
            strSaveDir += '/'

    # initialize the dataframe to store the results
    # will have 'train_ratio', 'maes', 'rmse', 'r2', 'train_size' columns
    dfGrowing_xgbMaxUncertainty_model1 = pd.DataFrame()
    dfGrowing_xgbMaxUncertainty_model2 = pd.DataFrame()

    # set the random state
    np.random.seed(intRandomSeed)
    random.seed(intRandomSeed)

    # initialize the size of the training set
    intTempTrainSize = int(len(xTrainVal)*fltStepFrac)
    # initialize the step size
    intStepSize = int(len(xTrainVal)*fltStepFrac)

    # initialize a list with the indices of the first randomly selected training set
    lstTrainIndices = np.random.choice(xTrainVal.index, intTempTrainSize, replace=False).tolist()
    
    while intTempTrainSize < (len(xTrainVal)+1):
        # start time
        timeStart = time.time()
        # --- CHECK STOP CRITERIA ---
        # check if the training set has met the stop criteria
        if intTempTrainSize >= len(xTrainVal)*fltTrainFracStop:
            # if so, break the loop
            break

        # --- SUBSAMPLE THE TRAINING SET ---
        xTrain_temp, xVal_temp, yTrain_temp, yVal_temp = splitTrainVal(xTrainVal, yTrainVal, lstTrainIndices)

        # --- CALCULATE, PRING AND STORE RESULTS ---
        print('\n--------------------------------------------------------------------------------------------')
        print('Training set size: ', len(xTrain_temp))
        print('Training set percentage: ', '{:.1%}'.format(len(xTrain_temp)/len(xTrainVal)), '\n')

        # # get scores using function for the first model
        maes_1, rmse_1, r2_1 = get_scores(model1,
                                          xTrain_temp, yTrain_temp,
                                          xTest, yTest)
        
        # # get scores using function for the second model
        maes_2, rmse_2, r2_2 = get_scores(model2,
                                          xTrain_temp, yTrain_temp,
                                          xTest, yTest)
        
        # store the results for the xgb model predictions
        dfGrowing_xgbMaxUncertainty_model1 = pd.concat([dfGrowing_xgbMaxUncertainty_model1,
                                                        pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                                      'maes': [round(maes_1,4)],
                                                                      'rmse': [round(rmse_1,4)],
                                                                      'r2': [round(r2_1,4)], 
                                                                      'train_size': [len(xTrain_temp)]})],
                                                        ignore_index=True)
        # store the results for the extra model predictions
        dfGrowing_xgbMaxUncertainty_model2 = pd.concat([dfGrowing_xgbMaxUncertainty_model2,
                                                        pd.DataFrame({'train_ratio': [round(len(xTrain_temp)/len(xTrainVal), 3)],
                                                                      'maes': [round(maes_2,4)],
                                                                      'rmse': [round(rmse_2,4)],
                                                                      'r2': [round(r2_2, 4)],
                                                                      'train_size': [len(xTrain_temp)]})],
                                                        ignore_index=True)
        # --- SAVE THE RESULTS ---
        # save the results
        dfGrowing_xgbMaxUncertainty_model1.to_csv(strSaveDir + strSaveName + '_' + strModel1Name + '.csv')
        dfGrowing_xgbMaxUncertainty_model2.to_csv(strSaveDir + strSaveName + '_' + strModel2Name + '.csv')
        # make a dataframe with the indices of the training set
        dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
        # save the indices of the training set
        dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
        
        # --- GROWING CRITERA ---
        # check if the training set is the full training+validation set
        if intTempTrainSize == len(xTrainVal):
            # if yes, break the loop
            intTempTrainSize+=10
            break

        # fit the xgb model for uncertainty calculation to the training set
        xgbModel4Uncert_fitted = xgbModel4Uncert.fit(xTrain_temp.to_numpy(), yTrain_temp.to_numpy())
        # extend xgb to a probabilistic estimator
        xgbModel4Uncert_ibug = IBUGWrapper().fit(xgbModel4Uncert_fitted, xTrain_temp.to_numpy(), yTrain_temp.to_numpy())
        xgbModel4Uncert_ibug.set_tree_subsampling(fltTreeSubsample, 'random')
        # predict the mean and variance for validation set
        yVal_temp_pred, yVal_temp_var = xgbModel4Uncert_ibug.pred_dist(xVal_temp.to_numpy())
        # make yVal_temp_var a dataframe with the same index as xVal_temp
        yVal_temp_var = pd.DataFrame(yVal_temp_var, index=xVal_temp.index, columns=['var'])
        # get the indices of the intStepSize points with the highest variance(uncertainty)
        lstMaxUncertIndices = yVal_temp_var.nlargest(intStepSize, 'var').index.tolist()
        
        # --- UPDATE THE TRAINING SET ---
        # add the indices of the intStepSize points with the highest variance(uncertainty) to the training set
        lstTrainIndices = lstTrainIndices + lstMaxUncertIndices
        # update the size of the training set
        intTempTrainSize = len(lstTrainIndices)

        # end time
        timeEnd = time.time()
        # print the time it took to run the loop
        print('Time to run this iteration: ', timeEnd-timeStart, ' seconds', flush=True) 
        print('\n--------------------------------------------------------------------------------------------')

    # --- SAVE THE RESULTS ---
    # save the results
    dfGrowing_xgbMaxUncertainty_model1.to_csv(strSaveDir + strSaveName + '_'+ strModel1Name + '.csv')
    dfGrowing_xgbMaxUncertainty_model2.to_csv(strSaveDir + strSaveName + '_'+ strModel2Name + '.csv')
    # make a dataframe with the indices of the training set
    dfTrainIndices = pd.DataFrame(lstTrainIndices, columns=['trainIndices'])
    # save the indices of the training set
    dfTrainIndices.to_csv(strSaveDir + strSaveName + '_trainIndices.csv')
    return
