import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import DataAnalysis

def correlationHeatMap(data, groupBy):
    if groupBy == "Instance":
        groupBy = 0
    elif groupBy == "Parameter name":
        groupBy = 1
    
    r = DataAnalysis.correlationAnalysis(data).reshape(data[0].nunique() - 4, data[1].nunique())
    fig, ax = plt.subplots(1, 2)
    sns.heatmap(r, linewidth=0.5, ax=ax[0])
    sns.heatmap(np.square(r), linewidth=0.5, ax=ax[1])
    ax[0].set_title("Correlation coefficients")
    ax[0].set_xticks(np.arange(data[1][:-4].nunique()), labels=data[1][:-4].unique())
    ax[0].set_yticks(np.arange(data[0][:-4].nunique()), labels=data[0][:-4].unique(), rotation=90)
    ax[0].tick_params("y", rotation=0)
    ax[1].set_title("Correlation coefficients squared")
    ax[1].set_xticks(np.arange(data[1][:-4].nunique()), labels=data[1][:-4].unique())
    ax[1].set_yticks(np.arange(data[0][:-4].nunique()), labels=data[0][:-4].unique(), rotation=90)
    ax[1].tick_params("y", rotation=0)
    plt.show()

    coefficientFrame = data[[0, 1]][:-4]
    coefficientFrame[2] = r.flatten()
    for label, df in coefficientFrame.groupby(groupBy):
        barValues = np.square(df.drop([0, 1], axis=1).transpose().to_numpy()).flatten()
        plt.bar(df[0], barValues)
        plt.title("Correlation coefficient of " + label + " with output frequency")
        plt.ylabel("Correlation coefficient of" + label + " (Absolute value)")
        plt.xticks(ticks=np.arange(df.shape[0]), labels=df[0], rotation=90)
        plt.show()
    return

def timeGraphs(data, groupBy):
    if groupBy == "Instance":
        groupBy = 0
    elif groupBy == "Parameter name":
        groupBy = 1

    times = data.iloc[-3][2:]
    outputs = data.iloc[-1][2:]
    plt.title("Output frequency over time")
    plt.scatter(times, outputs, marker='o')
    plt.ylabel("Frequency (GHz)")
    plt.xlabel("Time (h)")
    plt.show()

    inputs = data.iloc[0:-4]
    for label, df in inputs.groupby(1):
        inputValues = df.drop([0, 1], axis=1).transpose()
        plt.plot(times, inputValues, linestyle='', marker='o')
        plt.title(label + " over time")
        plt.ylabel(label)
        plt.xlabel("Time (h)")
        plt.legend(df[0], bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.show()

def frequencyVSparameter(data, groupBy):
    if groupBy == "Instance":
        groupBy = 0
    elif groupBy == "Parameter name":
        groupBy = 1

    outputs = data.iloc[-1][2:]
    inputs = data.iloc[0:-4]
    for label, df in inputs.groupby(groupBy):
        inputValues = df.drop([0, 1], axis=1).transpose()
        average = inputValues.sum(axis=1) / len(inputValues.columns)
        plt.scatter(average, outputs, marker='o', alpha=0.7)
        plt.title("Output frequency vs " + label)
        plt.ylabel("Frequency (GHz)")
        plt.xlabel("Average parameter value")
        plt.show()

def parameterComparison(data, groupBy):
    if groupBy == "Instance":
        groupBy = 0
    elif groupBy == "Parameter name":
        groupBy = 1

    inputs = data.iloc[0:-4].groupby(groupBy)
    fig, axes = plt.subplots(inputs.ngroups, inputs.ngroups)
    correlations = np.zeros((inputs.ngroups, inputs.ngroups))
    (index1, index2) = (0, 0)
    for labelA, dfA in inputs:
        xvalues = dfA.drop([0, 1], axis=1).to_numpy().flatten()
        for labelB, dfB in inputs:
            yvalues = dfB.drop([0, 1], axis=1).to_numpy().flatten()
            correlations[index1, index2] = DataAnalysis.correlationCoefficient(xvalues, yvalues)
            axes[index1, index2].scatter(xvalues, yvalues, marker='o', alpha=0.7)
            axes[index1, index2].plot([0, max(xvalues)], [0, max(xvalues)], color='r')
            axes[index1, index2].set_xlabel(labelA, fontsize=5)
            axes[index1, index2].set_ylabel(labelB, fontsize=5)
            axes[index1, index2].set_xticks([])
            axes[index1, index2].set_yticks([])
            index2 += 1
        index1 += 1
        index2 = 0
    print(correlations)
    plt.show()