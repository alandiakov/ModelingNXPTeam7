import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
import pandas as pd

# RMSE LOO Hat-sample

def correlationCoefficient(setA, setB):
    meanA = np.mean(setA)
    meanB = np.mean(setB)
    diffA = setA - meanA
    diffB = setB - meanB
    top = np.sum(diffA * diffB)
    bottom = np.sqrt(np.sum(np.square(diffA)) * np.sum(np.square(diffB)))
    return top / bottom

def correlationAnalysis(data):
    outputs = data.iloc[-1][2:]
    inputs = data.iloc[0:-4]
    r = np.zeros(len(inputs.index))
    for rowIndex in inputs.index:
        inputValues = inputs.loc[rowIndex][2:]
        r[rowIndex] = correlationCoefficient(inputValues, outputs)
    
    result = pd.DataFrame(columns=["Feature", "Correlation coefficient", "Sensitivity (%)"])
    result["Feature"] = data[0][:-4] + " " + data[1][:-4]
    result["Correlation coefficient"] = r
    result["Sensitivity (%)"] = 100 * (np.square(r) / np.sum(np.square(r)))
    result = result.transpose()
    lastInput = result.shape[1]
    result[lastInput] = ["Number of points", inputs.shape[1] - 2, ""]
    result = result.transpose()
    return result

def intervalCorrelation(data, timeStamps, printOutput=False):
    timeStamps.insert(0, 0)
    timeStamps.append(np.inf)
    transposedData = data.transpose()
    
    if printOutput:
        print("[" + ("-" * 100) + "] 0% Fitting data...", end='\r')
    for timeStampIndex in range(len(timeStamps) - 1):
        df = transposedData[2:]
        df = df[(df[len(df.columns) - 3] >= timeStamps[timeStampIndex]) & (df[len(df.columns) - 3] < timeStamps[timeStampIndex + 1])]
        if len(df.index) == 0:
            continue
        df = pd.concat([transposedData.iloc[[0, 1]], df]).transpose()
        newResult = correlationAnalysis(df)
        if timeStampIndex == 0:
            result = pd.DataFrame(newResult["Feature"])
        additionName = str(timeStamps[timeStampIndex]) + "h-" + str(timeStamps[timeStampIndex + 1]) + "h"
        if timeStampIndex == len(timeStamps) - 2:
            additionName = str(timeStamps[timeStampIndex]) + "h+"
        result = pd.concat([result, newResult.rename(columns={
            "Correlation coefficient": ("Correlation coefficient " + additionName),
            "Sensitivity (%)": ("Sensitivity (%) " + additionName)
        }).drop("Feature", axis=1)], axis=1)
        if printOutput:
            print("[" + ("=" * (round(100 * (timeStampIndex + 1) / len(timeStamps)))) + ("-" * (100 - round(100 * (timeStampIndex + 1) / len(timeStamps)))) + "] " + str(round(100 * (timeStampIndex + 1) / len(timeStamps))) + "% Fitting data...", end='\r')
    
    if printOutput:
        print("[" + ("=" * 100) + "] 100% Data fitted:        \n" + str(result))
        print("=" * 100)
    return result

def redoRegression(data, filterThreshhold=0, printOutput=False):
    filteredData = data.copy()
    filteredLength = len(filteredData.index) + 1
    while len(filteredData.index) < filteredLength:
        result = regression(filteredData, filterThreshhold, False)
        filteredLength = len(filteredData.index)
        filteredData = filteredData[:-4][(np.abs(result["Coefficient"][:-4]) >= filterThreshhold).to_numpy()].reset_index(drop=True)
        filteredData = pd.concat([filteredData, data[-4:]])
    if printOutput:
        print(result)
    return result

def intervalRegression(data, timeStamps, printOutput=False):
    timeStamps.insert(0, 0)
    timeStamps.append(np.inf)
    transposedData = data.transpose()
    
    if printOutput:
        print("[" + ("-" * 100) + "] 0% Fitting data...", end='\r')
    for timeStampIndex in range(len(timeStamps) - 1):
        df = transposedData[2:]
        df = df[(df[len(df.columns) - 3] >= timeStamps[timeStampIndex]) & (df[len(df.columns) - 3] < timeStamps[timeStampIndex + 1])]
        if len(df.index) == 0:
            continue
        df = pd.concat([transposedData.iloc[[0, 1]], df]).transpose()
        newResult = regression(df)
        if timeStampIndex == 0:
            result = pd.DataFrame(newResult["Feature"])
        additionName = str(timeStamps[timeStampIndex]) + "h-" + str(timeStamps[timeStampIndex + 1]) + "h"
        if timeStampIndex == len(timeStamps) - 2:
            additionName = str(timeStamps[timeStampIndex]) + "h+"
        result = pd.concat([result, newResult.rename(columns={
            "Coefficient": ("Coefficient " + additionName),
            "Sensitivity (%)": ("Sensitivity (%) " + additionName)
        }).drop("Feature", axis=1)], axis=1)
        if printOutput:
            print("[" + ("=" * (round(100 * (timeStampIndex + 1) / len(timeStamps)))) + ("-" * (100 - round(100 * (timeStampIndex + 1) / len(timeStamps)))) + "] " + str(round(100 * (timeStampIndex + 1) / len(timeStamps))) + "% Fitting data...", end='\r')
    
    if printOutput:
        print("[" + ("=" * 100) + "] 100% Data fitted:        \n" + str(result))
        print("=" * 100)
    return result

def regression(data, printOutput=False):
    #times = data["Times"].to_numpy()
    #outputs = np.log(data["GHz"].to_numpy())
    #inputs = data.drop(["Times", "GHz", "GHz Offset"], axis=1).to_numpy()
    times = data.iloc[-3][2:].to_numpy()
    #outputs = np.log(data.iloc[-1][2:].astype("float64").to_numpy())
    outputs = data.iloc[-1][2:].to_numpy()
    inputs = data.iloc[0:-4].drop([0, 1], axis=1).to_numpy().transpose()
    clf = Ridge(alpha=1)

    loo = LeaveOneOut()
    looScore = 0
    for i, (train_index, test_index) in enumerate(loo.split(inputs)):
        clf.fit(inputs[train_index], outputs[train_index])
        looScore += (clf.predict(inputs[test_index]) - outputs[test_index]) ** 2
    looScore = looScore[0] / len(outputs)

    clf.fit(inputs, outputs)
    R2score = clf.score(inputs, outputs)
    predictions = clf.predict(inputs)
    RMSEscore = np.sqrt(np.mean((predictions - outputs) ** 2))

    if printOutput:
        print("Data fitted: Intercept " + str(clf.intercept_) + ", coefficients " + str(clf.coef_))
        print("R^2 value: " + str(R2score))
        print("Total LOO score: " + str(looScore))
        print("RMSE value: " + str(RMSEscore))

        predictions = clf.predict(inputs)
        plt.scatter(outputs, predictions)
        #plt.show()

    result = pd.DataFrame(columns=["Feature", "Coefficient", "Sensitivity (%)"])
    result["Feature"] = data[0][:-4] + " " + data[1][:-4]
    result["Coefficient"] = np.array(clf.coef_)
    result["Sensitivity (%)"] = 100 * (np.abs(clf.coef_) / np.sum(np.abs(clf.coef_)))
    result = result.reset_index(drop=True).transpose()
    lastInput = result.shape[1]
    result[lastInput] = ["Intercept", clf.intercept_, ""]
    result[lastInput + 1] = ["R^2 value", R2score, ""]
    result[lastInput + 2] = ["LOO value", looScore, ""]
    result[lastInput + 3] = ["RMSE value", RMSEscore, ""]
    result[lastInput + 4] = ["Number of points", inputs.shape[0], ""]
    result = result.transpose()
    if printOutput:
        print(result)
    return result