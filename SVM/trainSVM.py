import csv
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd
import pprint as pp
import itertools
import pickle
from pandas import Series, DataFrame
from sklearn.svm import LinearSVC

class TwoStream_SVM():
    def __init__(self,attributePath,labelPath):
        self.attribute_data = []
        self.feature_names = []
        self.TwoStrem_feature_names = ["RGB:Background_Predicted","RGB:Bear_Predicted","RGB:Boar_Predicted",
                                        "MHI:Background_Predicted","MHI:Bear_Predicted","MHI:Boar_Predicted"]
        self.load_attribute(attributePath,labelPath)
        self.max_iteration = 100000 
        self.model = LinearSVC(max_iter = self.max_iteration)
    def load_attribute(self,attributePath,labelPath):
        with open(attributePath) as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC,delimiter=' ')
            self.attribute_data = [row for row in reader]
        with open(labelPath) as f:
            reader = csv.reader(f,quoting=csv.QUOTE_NONNUMERIC,delimiter=' ')
            self.feature_names = [row for row in reader]
        self.feature_names = [int(i) for i in list(itertools.chain.from_iterable(self.feature_names))]
    def calc_Accuracy(self,modelpath):
        dataframe = pd.DataFrame(self.attribute_data,columns = self.TwoStrem_feature_names)
        dataframe["target"] = self.feature_names
        explanatory_variable = dataframe.drop('target', axis=1)
        objective_variable = dataframe['target']
        model = pickle.load(open(modelpath, 'rb'))
        print("train score:",model.score(explanatory_variable,objective_variable))
        #print(model.predict(特徴量のリスト))
        #x = [[2.718307, 5.892713489, 0.999994158744812, 0.02142390049994, 0.002837722655386, 0.97573846578598]]
        #print(model.predict(x))
    def SVM_main(self):
        dataframe = pd.DataFrame(self.attribute_data,columns = self.TwoStrem_feature_names)
        dataframe["target"] = self.feature_names
        train_explanatory_variable = dataframe.drop('target', axis=1)
        train_objective_variable = dataframe['target']
        self.model.fit(train_explanatory_variable,train_objective_variable)
        filename = 'SVM_MobileNet_Wild-Life_part1_MobileNet_MHI_WildLife_log5_part1.sav'
        pickle.dump(self.model, open(filename, 'wb'))
    def __call__(self):
        self.SVM_main()

SVM = TwoStream_SVM(attributePath = r"E:\Research\TwoStreamCNN_2nd-Season\SVM\models\MobileNet_RGB_Wild-Life_part1_MobileNet_MHI_WildLife_log5_part1\explanatory_test.txt"
                    ,labelPath = r"E:\Research\TwoStreamCNN_2nd-Season\SVM\models\MobileNet_RGB_Wild-Life_part1_MobileNet_MHI_WildLife_log5_part1\objective_test.txt")
#SVM()
SVM.calc_Accuracy(modelpath=r"E:\Research\TwoStreamCNN_2nd-Season\SVM\models\MobileNet_RGB_Wild-Life_part1_MobileNet_MHI_WildLife_log5_part1\SVM_MobileNet_Wild-Life_part1_MobileNet_MHI_WildLife_log5_part1.sav")

