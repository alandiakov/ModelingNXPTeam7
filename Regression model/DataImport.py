import pandas as pd

def loadMultiTab():
    numberOfTabs = 17
    result = pd.read_excel("TUe_data (1).xlsx", sheet_name="DeviceDeg1")[["Instance", "Aging model", "Parameter type", "Parameter name"]].transpose().reset_index(drop=True).transpose()
    length = result.shape[0]
    result.loc[length, 0] = "Instance"
    result.loc[length + 1, 0] = "Time"
    result.loc[length + 2, 0] = "Temperature"
    result.loc[length + 3, 0] = "Frequency"
    print("[" + ("-" * 100) + "] 0% Loading data...", end='\r')
    tabSelects = range(1, numberOfTabs + 1)
    tabSelects = [1]
    for i in tabSelects:
        newInputs = pd.read_excel("TUe_data (1).xlsx", sheet_name="DeviceDeg" + str(i)).drop(["Instance", "Aging model", "Parameter type", "Parameter name", "Value at t=0"], axis=1).transpose().reset_index(drop=True).transpose()
        newOutputs = pd.read_excel("TUe_data (1).xlsx", sheet_name="Output" + str(i)).transpose().reset_index(drop=True)
        newOutputs.loc[0] = i
        newOutputs.loc[1] = newOutputs.loc[1].str[:-1]
        newOutputs.loc[3] = newOutputs.loc[3].str[:-1]
        newOutputs = newOutputs.astype("float64")
        newOutputs = newOutputs.fillna(0)
        newOutputs.loc[1] = newOutputs.cumsum(axis=1).loc[1]
        result = pd.concat([result, pd.concat([newInputs, newOutputs]).reset_index(drop=True)], axis=1)
        print("[" + ("=" * (round(100 * i / numberOfTabs))) + ("-" * (100 - round(100 * i / numberOfTabs))) + "] " + str(round(100 * i / numberOfTabs)) + "% Loading data...", end='\r')
    result = result.transpose().reset_index(drop=True).transpose()
    print("[" + ("=" * 100) + "] 100% Data loaded:        \n" + str(result.describe()))
    print("=" * 100)
    return result

def cluster(data, groupBy):
    if len(groupBy) == 0:
        return data
    
    groupBy = [0 if x == "Instance" else x for x in groupBy]
    groupBy = [1 if x == "Parameter name" else x for x in groupBy]

    groups = data[:-4].copy()
    groups["Chopped name"] = groups[0].str[:-1]
    groups["Subcircuit"] = groups[0].str[:2]
    groups = groups.groupby(groupBy)

    result = pd.DataFrame(columns=data.columns).reset_index(drop=True)
    counter = 0
    for label, df in groups:
        averages = df.sum()[2:-2] / len(df.index)
        result.loc[counter] = [df.iloc[0][groupBy].sum(), "Average value", *averages]
        counter += 1
    result = pd.concat([result, data[-4:]]).reset_index(drop=True)
    print("Clustered data:")
    print(result)
    print("=" * 100)
    return result

def filterData(data, printOutput=False):
    def compare(table, i, j):
        if (not i in table.index) or (not j in table.index) or i == j or i >= table.shape[0] or j >= table.shape[0]:
            return False
        return table.loc[i, "Instance"] == table.loc[j, "Instance"] and table.loc[i, "Value at t=0"] == table.loc[j, "Value at t=0"] and table.loc[i, "Value at t=720h"] == table.loc[j, "Value at t=720h"] and table.loc[i, "Value at t=3120h"] == table.loc[j, "Value at t=3120h"] and table.loc[i, "Value at t=10920h"] == table.loc[j, "Value at t=10920h"] and table.loc[i, "Value at t=12000h"] == table.loc[j, "Value at t=12000h"]

    size = data.size
    result = data.drop([1, 2], axis=1).transpose().reset_index(drop=True).transpose()
    if printOutput:
        print("Removed aging model and parameter type: " + str(size - result.size) + " entries (2 columns)")
    size = result.size
    result[1] = result[1].str[1:]
    if printOutput:
        print("Removed first letter from parameter name")
    result = result[(result[1] != "cidamage") & (result[1] != "btidamage")].reset_index(drop=True)
    if printOutput:
        print("Removed damage parameters: " + str(size - result.size) + " entries (" + str(len(data.index) - len(result.index)) + " rows)")
    size = result.size
    result = result[(result[1] != "btivthdegr") & (result[1] != "cibetdegrfac") & (result[1] != "civthdegr")].reset_index(drop=True)
    if printOutput:
        print("Removed duplicate parameters: " + str(size - result.size) + " entries (" + str(len(data.index) - len(result.index)) + " rows)")
    
    print("Filtered data: \n" + str(result.describe()))
    print("=" * 100)
    return result

def exportDataframe(data, name="Exported data"):
    data.to_excel(name + ".xlsx")
    print("Dataframe exported as '" + name + ".xlsx'")