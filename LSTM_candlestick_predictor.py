import os
import glob
from enum import Enum
import re
import pandas as P
import numpy 
import torch 
import torch.nn as N
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plot
import plotly.graph_objects as plty

# -----------------------------------------------------------------------------------------------
                            # LSTM MODEL CLASS DECLARATION 
# -----------------------------------------------------------------------------------------------
class LSTM(L.LightningModule):
    
    def __init__(self, aFeaturesAmount, aHiddenSize, aLayers ,aLearningRate, aLogFolder):
        super(LSTM, self).__init__()
        self.n_features = aFeaturesAmount
        self.n_hidden = aHiddenSize
        self.n_layers = aLayers
        self.pLearningRate = aLearningRate
        self.pLogFolder = aLogFolder
        self.pLstm = N.LSTM(input_size=self.n_features, hidden_size = self.n_hidden, num_layers = self.n_layers, batch_first = True)
        self.pOutputLayer = N.Linear(self.n_hidden, 4)
        self.save_hyperparameters('aFeaturesAmount', 'aHiddenSize', 'aLayers', 'aLogFolder', 'aLearningRate')
        
    def forward(self, aInput):
        pLstmOutput, pHidden = self.pLstm(aInput)
        return self.pOutputLayer(pLstmOutput[:,-1,:])
    
    def configure_optimizers(self):
        return Adam(self.parameters(), self.pLearningRate)
    
    def training_step(self, aBatch, aBatchIndx):
        pFeatures, pLabels = aBatch
        pPredictions = self.forward(pFeatures)
        pLoss = F.mse_loss(pPredictions, pLabels)
        self.log('train_loss', pLoss)
        customTrainLossLogger(pLoss, self.global_step, self.pLogFolder);
        return pLoss
    
#------------------------------------------------------------------------------------------------
                               # DATA PREPARATION- FUNCTIONS
# -----------------------------------------------------------------------------------------------
def normalizeValues(aT_SeriesData, aMaxVal, aMinVal):
            
    return (aT_SeriesData - float(aMinVal)) / (float(aMaxVal) - float(aMinVal))
#----------------------------------------------------------------------------
def denormalizeValues(aT_SeriesNormalizedData, pMaxVal, pMinVal):

    return aT_SeriesNormalizedData * (float(pMaxVal) - float(pMinVal)) + float(pMinVal)
#----------------------------------------------------------------------------
def getDataFromCsv(aDataPercentForTrain):

    pData = P.read_csv('Time-series.csv')
    pFeatures = pData[['Open', 'High', 'Low', 'Close']].dropna().values

    pTrainiT_SeriesData = pFeatures[:int(len(pFeatures) * (aDataPercentForTrain/100))]
    pEvalT_SeriesData = pFeatures
    
    return pTrainiT_SeriesData, pEvalT_SeriesData
#----------------------------------------------------------------------------
def prepareDataForTraining(aTrainT_SeriesData, aSequenceLength, aBatchSize, aMaxVal, aMinVal):
    
    pTrainiT_SeriesData = normalizeValues(aTrainT_SeriesData, aMaxVal, aMinVal)
    
    pInputSequences = []
    pLabels = []
        
    for pTimeStep in range(0, len(pTrainiT_SeriesData), aSequenceLength):
        if pTimeStep + aSequenceLength < len(pTrainiT_SeriesData):
            pInputSequences.append(pTrainiT_SeriesData[pTimeStep:pTimeStep + aSequenceLength])
            pLabels.append(pTrainiT_SeriesData[pTimeStep + aSequenceLength])
            
    pInputSequences_tensor = torch.tensor(pInputSequences).float()
    pLabels_tensor = torch.tensor(pLabels).float()

    pSamples_tensor = TensorDataset(pInputSequences_tensor, pLabels_tensor)

    pDataLoader = DataLoader(pSamples_tensor, aBatchSize, shuffle=True)
    
    return pDataLoader
#----------------------------------------------------------------------------
def prepareDataForEvaluation(aEvalT_SeriesData, aMaxVal, aMinVal):

    pEvalT_SeriesData = normalizeValues(aEvalT_SeriesData, aMaxVal, aMinVal)

    pEvalT_SeriesTensor = torch.Tensor(pEvalT_SeriesData).float()

    return pEvalT_SeriesTensor.unsqueeze(1)
#----------------------------------------------------------------------------
def getPredictedPrices(aPredictedValues, aLogFolder):
    
    pFileNameWithMinMaxVals = "MaxMinValues.txt"
    pMaxVal, pMinVal = extractSavedMinMaxValsFromFile(pFileNameWithMinMaxVals, aLogFolder)
    
    pPredcitedValues = aPredictedValues.numpy()
    
    return denormalizeValues(pPredcitedValues, pMaxVal, pMinVal)
#------------------------------------------------------------------------------------------------
                       # LOGGING/ EXTRACTING TRAINING/ EVAL METRICS- FUNCTIONS
# -----------------------------------------------------------------------------------------------
def createLogFolder(aFolderPath):
    
    try:
        os.makedirs(aFolderPath)
    except FileExistsError:
        pass
#----------------------------------------------------------------------------
def getLogFolderName(aFolderVersion):

    pFolderVersion = 1
    
    try:
        pDirectory = "Logs"

        pFolders = os.listdir(pDirectory)

        pFoldersName = r'Log_Version(\d+)'
    
        pFolders = [pFolder for pFolder in pFolders if os.path.isdir(os.path.join(pDirectory, pFolder)) and re.match(pFoldersName, pFolder)]

        if aFolderVersion == None:       
            for pFolder in pFolders:
                pFolderNum = int(re.search(pFoldersName, pFolder).group(1))
                if pFolderNum > pFolderVersion:
                    pFolderVersion = pFolderNum
                pFolderVersion += 1
        else:
            pFolderVersion = aFolderVersion
    except:
        pass
    
    return "Logs\Log_Version" + str(pFolderVersion)
#----------------------------------------------------------------------------
def saveMinMaxValsOfTrainT_SeriesData(aTrainTSeriesData, aFileNameWithMinMaxVals, aLogFolder):
    
    pFlattenedDataSet = [pFeature for pTimestep in aTrainTSeriesData for pFeature in pTimestep]
        
    try:
        pMaxVal, pMinVal = extractSavedMinMaxValsFromFile(aFileNameWithMinMaxVals, aLogFolder)
        if max(pFlattenedDataSet) > pMaxVal:
            pMaxVal = max(pFlattenedDataSet)
        if min(pFlattenedDataSet) < pMinVal:
            pMinVal = min(pFlattenedDataSet)
    except:
        pMaxVal = max(pFlattenedDataSet)
        pMinVal = min(pFlattenedDataSet)
    
    with open(aLogFolder + "/" + aFileNameWithMinMaxVals, "w") as pFileWithMinMaxVals:
        pFileWithMinMaxVals.write(f"Max: {pMaxVal}\n")
        pFileWithMinMaxVals.write(f"Min: {pMinVal}\n")

    return pMaxVal, pMinVal
#----------------------------------------------------------------------------
def extractSavedMinMaxValsFromFile(aFileNameWithMinMaxVals, aLogFolder):
    
    with open(aLogFolder + "/" + aFileNameWithMinMaxVals, "r") as pFileWithMinMaxVals:
        pLines = pFileWithMinMaxVals.readlines()

    pMaxVal = pLines[0].split(": ")[1].strip()
    pMinVal = pLines[1].split(": ")[1].strip()

    return pMaxVal, pMinVal
#----------------------------------------------------------------------------
def saveLstmModel(aLogFolder):
     
     pCheckpoint = ModelCheckpoint(
        dirpath=aLogFolder,
        filename=f"Trained_LSTM_Model",
        save_top_k=1,
        verbose=False,
        monitor='train_loss',
        mode='min'
    )
     return pCheckpoint
#----------------------------------------------------------------------------
def getTrainedLstmModel(aLogFolder):

    pFilesList = glob.glob(os.path.join(aLogFolder, '*.ckpt')) 
    
    if not pFilesList:
        return None

    pTrainedLstmModel = max(pFilesList, key=os.path.getctime)

    return pTrainedLstmModel
#----------------------------------------------------------------------------
def getLossLogNum(aLossLogDir, aLossLogFlag):
    
    pExistingFiles = os.listdir(aLossLogDir)
    
    pMaxFileNum = 0
    
    try:        
        for pFile in pExistingFiles:
            if aLossLogFlag and pFile.startswith("My_Train_Loss_Log") and pFile.endswith(".csv"):
                pFileNum = int(pFile.split('My_Train_Loss_Log')[1].split('.csv')[0])
                pMaxFileNum = max(pMaxFileNum, pFileNum)
            elif not aLossLogFlag and pFile.startswith("My_Eval_Loss_Log") and pFile.endswith(".csv"):
                pFileNum = int(pFile.split('My_Eval_Loss_Log')[1].split('.csv')[0])
                pMaxFileNum = max(pMaxFileNum, pFileNum)       
    except:
        pass         
    
    return pMaxFileNum
#----------------------------------------------------------------------------
def customTrainLossLogger(aLoss, aTrainingStep, aLogFolder):
    
    pLossLogDir = f"{aLogFolder}/Loss_Log"  

    pLossLogNum = getLossLogNum(pLossLogDir, True)
    
    pLossLogFilePath = f"{pLossLogDir}/My_Train_Loss_Log{pLossLogNum}.csv"
   
    if aTrainingStep == 0:
         pLossLogFilePath = f"{pLossLogDir}/My_Train_Loss_Log{pLossLogNum+1}.csv" 
         P.DataFrame(columns=["Loss", "Training Step"]).to_csv(pLossLogFilePath, index=False)
        
    pNewRow = P.DataFrame({"Loss": [aLoss.item()], "Training Step": [aTrainingStep]})
    pNewRow.to_csv(pLossLogFilePath, mode='a', header=False, index=False)

#----------------------------------------------------------------------------
def customEvalLossLogger(aActualT_SeriesData, aPredictedT_SeriesData, aLogFolder):
    
    pLossLogDir = f"{aLogFolder}/Loss_Log"  

    pLossLogNum = getLossLogNum(pLossLogDir, False)
    
    pLosses = []
    for pPred, pActual in zip(aPredictedT_SeriesData, aActualT_SeriesData):
        pLoss = F.mse_loss(pPred, pActual)
        pLosses.append(pLoss.item())

    pTimeSteps = numpy.arange(1, len(pLosses) + 1)
    
    pLossLogFilePath = f"{pLossLogDir}/My_Eval_Loss_Log{pLossLogNum+1}.csv" 
    P.DataFrame(columns=["Loss", "Time-step"]).to_csv(pLossLogFilePath, index=False)
    P.DataFrame({"Loss": pLosses, "Time-step": pTimeSteps}).to_csv(pLossLogFilePath, mode='a', header=False, index=False)
#----------------------------------------------------------------------------
def savePredictions(aPredictedT_SeriesData, aLogFolder):    
    
    pTimeSteps = numpy.arange(1, len(aPredictedT_SeriesData) + 1)
    
    pLossLogFilePath = f"{aLogFolder}/Predictions.csv" 
    P.DataFrame(columns=["Time-step", "Open", "High", "Low", "Close"]).to_csv(pLossLogFilePath, index=False)
    P.DataFrame({"Time-step": pTimeSteps, "Open": aPredictedT_SeriesData[:, 0], "High": aPredictedT_SeriesData[:, 1], "Low": aPredictedT_SeriesData[:, 2], "Close": aPredictedT_SeriesData[:, 3]}).to_csv(pLossLogFilePath, mode='a', header=False, index=False)
#------------------------------------------------------------------------------------------------
                              # RUNNING LSTM MODEL- FUNCTIONS
# -----------------------------------------------------------------------------------------------
def trainLstm(aLstmModel, aTrainTSeriesData, aSampleSize, aBatchSize, aEpochsAmount, aLogFolder):
    
    pFileNameWithMinMaxVals = "MaxMinValues.txt"
    pMaxVal, pMinVal = saveMinMaxValsOfTrainT_SeriesData(aTrainTSeriesData, pFileNameWithMinMaxVals, aLogFolder)
    
    pTrainTSeriesDataLoader = prepareDataForTraining(aTrainTSeriesData, aSampleSize, aBatchSize, pMaxVal, pMinVal)

    pCheckpoint = saveLstmModel(aLogFolder)
    
    pTrainer = L.Trainer(max_epochs=aEpochsAmount, callbacks=[pCheckpoint])
    pTrainer.fit(aLstmModel, train_dataloaders=pTrainTSeriesDataLoader) 

#----------------------------------------------------------------------------
def runLstmModelEval_ForwardPass(aLstmModel, aEvalT_SeriesData, aLogFolder):
    
    pFileNameWithMinMaxVals = "MaxMinValues.txt"
    pMaxVal, pMinVal = extractSavedMinMaxValsFromFile(pFileNameWithMinMaxVals, aLogFolder)
    
    pEvalData = prepareDataForEvaluation(aEvalT_SeriesData, pMaxVal, pMinVal)

    aLstmModel.eval()
    
    with torch.no_grad():
        pPredictedValues = aLstmModel(pEvalData)
    
    customEvalLossLogger(pEvalData, pPredictedValues, aLogFolder)
        
    return pPredictedValues
#------------------------------------------------------------------------------------------------
                                    # VISULIZATION- FUNCTIONS
# -----------------------------------------------------------------------------------------------
def createCandlestickChart(aActualT_SeriesData, aPredictedT_SeriesData):
    
    pIndices = numpy.arange(1, len(aActualT_SeriesData) + 1)
   
    pDf1 = P.DataFrame(aActualT_SeriesData, columns=['Open', 'High', 'Low', 'Close'])
    pDf2 = P.DataFrame(aPredictedT_SeriesData, columns=['Open', 'High', 'Low', 'Close'])
    
    pDf1['Time-step'] = pIndices
    pDf2['Time-step'] = pIndices

    pCandlestickChart = plty.Figure()

    pCandlestickChart.add_trace(
        plty.Candlestick(x=pDf1['Time-step'],
        open=pDf1['Open'],
        high=pDf1['High'],
        low=pDf1['Low'],
        close=pDf1['Close'],
        increasing_line_color='green',  
        decreasing_line_color='green',  
        name='Actual Candlesticks')
    )
    
    pCandlestickChart.add_trace(
        plty.Candlestick(x=pDf2['Time-step'],
        open=pDf2['Open'],
        high=pDf2['High'],
        low=pDf2['Low'],
        close=pDf2['Close'],
        increasing_line_color='red', 
        decreasing_line_color='red',  
        name='Prediced Candlesticks')
    )
    
    pCandlestickChart.add_trace(
        plty.Scatter(x=[pDf1['Time-step'][int(len(pDf1)*0.8)], pDf1['Time-step'][int(len(pDf1)*0.8)]],
        y=[pDf1['High'].max(), 0], 
        mode='lines',
        line=dict(color='blue', dash='dash'),
        name='Right of Line: Out-of-Sample Predictions')
    )
    
    pCandlestickChart.update_layout(
        title='Predicted vs Actual Candlestick Chart', font=dict(size=25),
        xaxis_title='n Time-steps',
        yaxis_title='Price',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=25)),  
        xaxis=dict(title_font=dict(size=25)),  
        yaxis=dict(title_font=dict(size=25))
    )
    
    pCandlestickChart.show()
#------------------------------------------------------------------------------    
def graphLoss(aLogFolder):
    
    pLossLogDir = f"{aLogFolder}/Loss_Log"  

    pLossLogNum = getLossLogNum(pLossLogDir, True)
    
    pLossLogFilePath = f"{pLossLogDir}/My_Train_Loss_Log{pLossLogNum}.csv"
    
    pLossLog = P.read_csv(pLossLogFilePath)
    
    pLossData = pLossLog[['Loss', 'Training Step']].dropna()
    pLossData = pLossData.values
    
    plot.figure()
    plot.plot(pLossData[:, 1], pLossData[:, 0], label=f"Loss")
    plot.title(f'Training Loss Log {pLossLogNum}')
    plot.xlabel('n Batches')
    plot.ylabel('Loss')
    plot.yscale('log')
    plot.legend() 
    plot.show()

#------------------------------------------------------------------------------------------------
                                    # USER INPUT
# -----------------------------------------------------------------------------------------------
class mode(Enum):
    new_training = 1
    resume_training = 2
    evaluation = 3
   
def getMode():
    
    print("Program Modes:\n1. Train New Model (1)\n2. Resume Training an Existing Trained Model (2)\n3. Evaluate a Trained Model (3)")
    
    while True:
        pMode = int(input("\nSelect Program Mode (1-3): "))
        
        if pMode >= 1 and pMode <= 3:
            break
              
    return pMode
#------------------------------------------------------------------------------
def getTrainingSettings():
    
    pSequenceLength = int(input("\nEnter Sequence Length: "))
    pBatchSize = int(input("\nEnter Batch Size: "))
    pLearningRate =  float(input("\nEnter the Learning Rate: "))
    pEpochs = int(input("\nEnter Number of Epochs: "))
    
    return pSequenceLength, pBatchSize, pLearningRate, pEpochs

#------------------------------------------------------------------------------------------------
                                    # MAIN FUNCTION
# -----------------------------------------------------------------------------------------------
def performOperations():
    
    pTrainiT_SeriesData, pEvalT_SeriesData = getDataFromCsv(80)
    
    pMode = getMode()
    
    if pMode == mode.new_training.value:
        pLogFolder = getLogFolderName(None)
        createLogFolder(pLogFolder)
        createLogFolder(pLogFolder + "\Loss_Log")
        pSequenceLength, pBatchSize, pLearningRate, pEpochs = getTrainingSettings()
        pLstmModel = LSTM(aFeaturesAmount=4, aHiddenSize=400, aLayers=1, aLearningRate=pLearningRate, aLogFolder=pLogFolder)
    elif pMode == mode.resume_training.value:
        while True:
            try:
                pTrainingLogVersionNum = input("\nEnter the Log Version: ")
                pLogFolder = getLogFolderName(pTrainingLogVersionNum)
                pSequenceLength, pBatchSize, pLearningRate, pEpochs = getTrainingSettings()
                pLstmModel = LSTM.load_from_checkpoint(checkpoint_path=getTrainedLstmModel(pLogFolder), aLearningRate=pLearningRate)
                break
            except:
                print("\nWrong Log Version, Try Again.")
                pass
    elif pMode == mode.evaluation.value:
         while True:
            try:
               pTrainingLogVersionNum = input("\nEnter the Log Version: ")
               pLogFolder = getLogFolderName(pTrainingLogVersionNum)
               pLstmModel = LSTM.load_from_checkpoint(checkpoint_path=getTrainedLstmModel(pLogFolder))
               break
            except:
                print("\nWrong Log Version, Try Again.")
                pass
     
    if pMode == mode.new_training.value or pMode == mode.resume_training.value:
        trainLstm(pLstmModel, pTrainiT_SeriesData, pSequenceLength, pBatchSize, pEpochs, pLogFolder)
     
    pPredcitedValues_Tensor = runLstmModelEval_ForwardPass(pLstmModel, pEvalT_SeriesData, pLogFolder)

    pPredictedPrices = getPredictedPrices(pPredcitedValues_Tensor, pLogFolder)
    
    savePredictions(pPredictedPrices, pLogFolder);
    
    createCandlestickChart(pEvalT_SeriesData, pPredictedPrices)
    
    graphLoss(pLogFolder)
    
#------------------------------------------------------------------------------------------------    
performOperations()





