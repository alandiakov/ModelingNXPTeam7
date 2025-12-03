import DataImport
import Visualization
import DataAnalysis
import pandas as pd

#dataFrame = DataImport.cluster(DataImport.filterData(DataImport.loadMultiTab()), [])
#dataFrame = DataImport.cluster(DataImport.filterData(DataImport.loadMultiTab()), ["Instance"])
#dataFrame = DataImport.cluster(DataImport.filterData(DataImport.loadMultiTab()), ["Chopped name"])
dataFrame = DataImport.cluster(DataImport.filterData(DataImport.loadMultiTab()), [])

#Visualization.timeGraphs(dataFrame, "Instance")
#Visualization.timeGraphs(dataFrame, "Parameter name")
#Visualization.frequencyVSparameter(dataFrame, "Instance")
#Visualization.frequencyVSparameter(dataFrame, "Parameter name")
#Visualization.parameterComparison(dataFrame, "Instance")
#Visualization.parameterComparison(dataFrame, "Parameter name")
#Visualization.correlationHeatMap(dataFrame, "Parameter name")
#DataImport.exportDataframe(DataAnalysis.correlationAnalysis(dataFrame), "Fit data")
#DataImport.exportDataframe(DataAnalysis.intervalCorrelation(dataFrame, [8760 * i for i in range(1, 10)]), "Fit data")
DataImport.exportDataframe(DataAnalysis.regression(dataFrame, True), "Fit data")
#DataImport.exportDataframe(DataAnalysis.intervalRegression(dataFrame, [8760 * i for i in range(1, 10)], True), "Fit data")
#DataImport.exportDataframe(DataAnalysis.redoRegression(dataFrame, 0.25, True), "Fit data")