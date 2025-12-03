import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

def parameterOverFrequency():
    parameterNames = list(set(inputs["Parameter name"].unique()) - set(["hcidamage", "pbtidamage", "nbtidamage"]))
    parameterNames.sort()
    
    for name in parameterNames:
        temp = inputs[inputs["Parameter name"] == name][["Value at t=0", "Value at t=720h", "Value at t=3120h", "Value at t=10920h", "Value at t=12000h"]].transpose()
        summation = temp.sum(axis=1)[1:] / len(temp.columns)
        plt.plot([720, 3120, 10920, 12000] + 100*np.random.rand(4), summation.to_numpy() / outputs["Output"].to_numpy(), marker='o')
    plt.legend(parameterNames)
    plt.title("Average parameter / frequency vs time")
    plt.xlabel("Time")
    plt.ylabel("Average parameter / frequency")
    plt.show()

def simplePercentage(data, groupBy):
    degradations = data[["Value at t=720h", "Value at t=3120h", "Value at t=10920h", "Value at t=12000h"]]
    percentage = pd.DataFrame(degradations.diff(axis=1).drop("Value at t=720h", axis=1).to_numpy() / degradations.drop("Value at t=12000h", axis=1).to_numpy())
    
    #percentage = pd.concat([data[["Instance", "Parameter name"]], percentage], axis=1, join="inner")
    totals = percentage.sum(axis = 0, numeric_only=True)

    result = 100 * percentage / totals
    result["Instance"] = data["Instance"]
    result["Parameter name"] = data["Parameter name"]
    result = result.groupby([groupBy], as_index=False).sum()
    
    #print(result.sort_values(by=[0], ascending=False).head())
    #print(result.sort_values(by=[1], ascending=False).head())
    #print(result.sort_values(by=[2], ascending=False).head())

    plt.bar(np.arange(result.shape[0]) - 0.2, result[0], width = 0.2)
    plt.bar(np.arange(result.shape[0]), result[1], width = 0.2)
    plt.bar(np.arange(result.shape[0]) + 0.2, result[2], width = 0.2)
    plt.xticks(ticks=np.arange(result.shape[0]), labels=result[groupBy], rotation=45)
    plt.ylabel("Influence (%)")
    plt.title("Influence percentage per " + groupBy)
    plt.legend(["Interval 2", "Interval 3", "Interval 4"])
    plt.show()

def covariance(data):
    times = np.array([720, 3120, 10920, 12000])
    times = times - np.mean(times)
    rescaled = np.log((data[data.select_dtypes(include=['number']).columns] * 10 ** 7) + 1).to_numpy()
    rescaled = np.log(data[data.select_dtypes(include=['number']).columns].to_numpy())
    rescaled = rescaled - np.mean(rescaled, axis=1).reshape((rescaled.shape[0], 1))
    covariance = np.sum(rescaled * times, axis=1) / rescaled.shape[1]
    print(covariance)
        

