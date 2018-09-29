# coding: utf-8
import plotly.offline as py
import plotly.graph_objs as go
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.utils.testing import ignore_warnings
import operator
import csv
import time
import mlpy
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from textwrap import wrap
import statsmodels.api as sm
from itertools import cycle
from toolz import interleave
import datetime
import ast
from ast import literal_eval
import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import sys
import scipy as sp
import scipy.stats
import psycopg2
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from wpca import PCA
import sklearn.metrics as metric
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import scipy.stats.mstats as mst

desired_width = 320
pd.set_option('display.width', desired_width)


#Seed="S-7307926_7553318_7947782_10368799_13086631\Train-7307926_7553318_7947782_10368799_13086631_new_new.csv"
#Seed="Paris_zone_weekday_May19-Sep29.csv"
#Seed="Marseille_zone_weekday_May19-Jul26_stepwise.csv"
Seed="dataset.csv"
#Seed="S-7792928_8773192\Train-7792928_8773192_new_new.csv.csv"
#Seed="18_sections\\18_sections.csv"

f = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/Difference_prediction_average.csv", "w")
#f = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/summary_novar.csv", "w")
#f.write(',MSE,R2,CI_X,CI_y,NanX_train,Nany_train,NanX_test,Nany_test'+'\n')
#fSpeeds = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/Speed_summary_.csv", "w")
#fPoints = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/Point_summary_.csv", "w")
#fStds = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/Std_summary_.csv", "w")

i = 0
TimeStep=15
TimeMin="06:30:00"
TimeMax="08:30:00"
SectionListOfInterest = 'S-8154551,S-8788682,S-8788686,S-7743814,S-7792965,S-13278556,S-10191592,S-8006261,S-7743817,S-8086814,S-11352412,S-14656876'

# 'S-8154551,S-8788682,S-8788686,S-7743814,S-7792965,S-13278556,S-9298893,S-9328169,S-10191592,S-8006261,S-7743817,S-8086814,' \
#                         'S-7763383,S-7787975,S-8211050,S-8707005,S-10858299,S-11352412,S-11881467,S-12986706,S-13620236,S-14656876,S-15196092,S-15354055,' \
#                         'S-15566884,S-8132794,S-13620265,S-9332553,S-8722904,S-8705304,S-9103111,S-13620600,S-8525566,S-8477777,S-13108190,S-7984999,S-13325970,'\
#                         'S-7987611,S-7976319,S-7975642,S-7985046,S-9332464,S-9664840,S-7984958,S-14908185,S-13641599,S-12821860,S-14465136,S-7977364,S-13471750'


SectionListInterest = SectionListOfInterest.split(',')

def getSectionSpeedDic(Seed,TimeStep,TimeMin,TimeMax):
    FichierLecture = open(Seed, "rU")
    reader = csv.reader(FichierLecture)#load data file
    TimeStampMin = 3600 * int(TimeMin[0:2]) + 60 * int(TimeMin[3:5]) + int(TimeMin[6:8])#lower bound timestamp
    TimeStampMax = 3600 * int(TimeMax[0:2]) + 60 * int(TimeMax[3:5]) + int(TimeMax[6:8])#upper bound timestamp
    NumTimeSteps = int(float(TimeStampMax - TimeStampMin) / (60 * TimeStep)) #get timestep number between two timestamp
    TimeFlow = []
    ListeInit = []
    ListeListeInit = []
    for i in range(NumTimeSteps):#initial timeflow and list of timestamps
        TimeFlow.append(i)
        ListeInit.append(0)
        ListeListeInit.append([])

    Count = 0
    Min = 1
    IdField = 0
    LatField = 1
    LonField = 2
    HeadingField = 3
    SpeedField = 4
    TimeField = 5
    SectionField = 6

    SectionNumDic = {}
    SectionCarDic = {}
    SectionSpeedDic = {}
    SectionSeparateSpeedDic = {}
    SectionSeparateSpeedDicWithDate = {}
    SectionCoordinateDic = {}
    dateArray = []

    for e in reader:
        if Count < Min:
            Count = Count + 1
        else:
            Id = e[IdField]
            longitude = e[LonField]
            latitude = e[LatField]
            Speed = int(e[SpeedField])
            heading = int(e[HeadingField])
            #if Speed > 130:
                #Speed = heading
            Time = e[TimeField]
            Section = e[SectionField]
            if Section in SectionListInterest:
                Date=Time[:-5]
                dateArray.append(Date)
                Heure=int(Time[-4:-3])
                Minutes=int(Time[-2:])
                TimeStamp=(3600*Heure+60*Minutes)
                TempsStep=int(float(TimeStamp-TimeStampMin)/(60*TimeStep))
                Id_date = Id + '_' + Date
                id_coordinate = Id+ '_' +latitude+ '_' +longitude
                if Section in SectionNumDic:
                    SectionNumDic[Section][TempsStep]=SectionNumDic[Section][TempsStep]+1
                    if Id_date in SectionCarDic[Section][TempsStep]:
                        pass #as there should be only one appearance of a specific car(Id) in a timestep of a day(date)
                    else:
                        SectionCarDic[Section][TempsStep].append(Id_date)
                    if(SectionSpeedDic[Section][TempsStep] == ""):
                        SectionSpeedDic[Section][TempsStep] = 0
                    SectionSpeedDic[Section][TempsStep]=SectionSpeedDic[Section][TempsStep]+Speed
                    SectionSeparateSpeedDic[Section][TempsStep]= SectionSeparateSpeedDic[Section][TempsStep]+str(Speed)+","
                    SectionSeparateSpeedDicWithDate[Section][TempsStep]= SectionSeparateSpeedDicWithDate[Section][TempsStep] \
                    + Date + "_" + str(Speed)+","
                    SectionCoordinateDic[Section][TempsStep].append(id_coordinate)
                    #SectionSpeed2Dic[Section][TempsStep]=SectionSpeed2Dic[Section][TempsStep]+Speed*Speed
                else:
                    SectionNumDic[Section]=[]
                    SectionCarDic[Section]=[]
                    SectionSpeedDic[Section]=[]
                    #SectionSpeed2Dic[Section]=[]
                    SectionSeparateSpeedDic[Section] = []
                    SectionSeparateSpeedDicWithDate[Section] = []
                    #SectionFlowDic[Section]=[]
                    SectionCoordinateDic[Section] = []

                    for k in range(NumTimeSteps):
                        SectionNumDic[Section].append(0)
                        SectionCarDic[Section].append([])
                        SectionSpeedDic[Section].append("")
                        #SectionSpeed2Dic[Section].append(0)
                        #SectionFlowDic[Section].append(0)
                        SectionSeparateSpeedDic[Section].append("")
                        SectionSeparateSpeedDicWithDate[Section].append("")
                        SectionCoordinateDic[Section].append([])

                    #print("0",Section,Indice,TempsStep,type(SectionNumDic[Section]),type(SectionCarDic[Section]),type(SectionSpeedDic[Section]))
                    SectionNumDic[Section][TempsStep]=1
                    #print("1",Section,Indice,TempsStep,SectionNumDic[Section])
                    SectionCarDic[Section][TempsStep].append(Id_date)
                    #print("2",Section,Indice,TempsStep,SectionCarDic[Section])
                    SectionSpeedDic[Section][TempsStep]=Speed
                    #print("3",Section,Indice,TempsStep,SectionSpeedDic[Section])
                    #SectionSpeed2Dic[Section][TempsStep]=Speed*Speed
                    #print("4",Section,Indice,TempsStep,SectionSpeed2Dic[Section])
                    SectionCoordinateDic[Section][TempsStep].append(id_coordinate)
            Count = Count + 1

    for S in SectionNumDic:
        for i in range(NumTimeSteps):
            if SectionNumDic[S][i] == 0:
                # f.write(str(i)+',0,0,0'+'\n')
                pass
            else:
                SectionSpeedDic[S][i] = float(SectionSpeedDic[S][i]) / SectionNumDic[S][i]
                #SectionSpeed2Dic[S][i] = sqrt(float(SectionSpeed2Dic[S][i]) / SectionNumDic[S][i] - SectionSpeedDic[S][i] * SectionSpeedDic[S][i])
                #temp = float(SectionSpeedDic[S][i])
                #SectionFlowDic[S][i] = SectionNumDic[S][i] * float(SectionSpeedDic[S][i])
                #f.write(str(i)+','+str(SectionSpeedDic[S][i])+','+str(std)+','+str(SectionNumDic[S][i])+'\n')
        #print(S,SectionNumDic[S],SectionCarDic[S],SectionSpeedDic[S],SectionSpeed2Dic[S],SectionFlowDic[S][i])
        #Num = np.array(SectionNumDic[S])
        #Cars =
        #np.array(SectionCarDic[S])
        #Speed = np.array(SectionSpeedDic[S])
        #Speed2 = np.array(SectionSpeed2Dic[S])
        #Flow = np.array(SectionFlowDic[S])
        SeparateSpeed = np.array(SectionSeparateSpeedDic[S])
        #print()
        #TimeTable = np.array(TimeFlow)
    #plotting(SectionSpeedDic, S, TimeFlow, Date, TimeMin, TimeMax, Speed)
    dateArray = sorted(set(dateArray))
    return SectionSpeedDic,SectionSeparateSpeedDic,SectionCarDic,SectionCoordinateDic,SectionSeparateSpeedDicWithDate,dateArray

def getConnectedSectionList(centerSection):
    hostname = 'localhost'  # 'srv044.it4pme.fr'
    username = 'postgres'  # 'geo4cast'
    password = '111111'  # LY+tpLRQA5lmX//'
    database = 'postgres'  # geo4cast'
    conn = psycopg2.connect(host=hostname, user=username, password=password, dbname=database)
    cur = conn.cursor()
    s = centerSection[2:]

    SQLquery = """SELECT g2.ogc_fid, g2.other_tags, g1.other_tags 
                  FROM zone_sections As g1, zone_sections As g2 
                  WHERE g1.ogc_fid = %(s)s 
                  and ST_Touches(g1.wkb_geometry::geometry, g2.wkb_geometry::geometry) IS TRUE
                  and g1.other_tags like '%maxspeed%' and g2.other_tags like '%maxspeed%'""" % {'s': s}

    # s1 = i[2:]
    # s2 = j[2:]
    #
    # SQLquery = """SELECT ST_Touches(g1.wkb_geometry::geometry,g2.wkb_geometry::geometry),
    #                          ST_distance(st_transform(st_setsrid(g1.wkb_geometry::geometry,4326),2100),st_transform(st_setsrid(g2.wkb_geometry::geometry,4326),2100)),
    #                          g1.other_tags,g2.other_tags,g1.highway,g2.highway
    #                    FROM postgis.zone_sections As g1, postgis.zone_sections As g2
    #                    WHERE g1.ogc_fid = %(s1)s and g2.ogc_fid = %(s2)s""" % {'s1': s1,
    #                                                                            's2': s2}

    cur.execute(SQLquery)
    result = cur.fetchall()  # list of corresponding sections of the point

# def countInvalidItemPair(list_avg_speed_1,list_avg_speed_2):
#     count = 0
#     index = []
#     for i in range (0,len(list_avg_speed_2)):
#         if(list_avg_speed_1[i]=="" or list_avg_speed_2[i]=="" ):
#             count = count + 1
#             index.append(i)
#             #del list_avg_speed_1[i]
#             #del list_avg_speed_2[i]
#     list1 = [k for j, k in enumerate(list_avg_speed_1) if j not in index]
#     list2 = [k for j, k in enumerate(list_avg_speed_2) if j not in index]
#     return count,list1,list2#index#list_avg_speed_1,list_avg_speed_2

# def removeBlankItem(item_collection):
#     index = []
#     for i in range(0, len(item_collection)):
#         if (item_collection[i] == ""):
#             index.append(i)
#     list = [k for j, k in enumerate(item_collection) if j not in index]
#     return list

SQLquery = """SELECT ogc_fid, other_tags FROM zone_sections WHERE other_tags like '%maxspeed%'"""

# def outputFileCongestion(SectionCoordinateDic,dictSectionSpeedLimit,SectionSpeedDic,timestep):
#     SectionCoordinateSet = set(SectionCoordinateDic)
#     SetSectionSpeedLimit = set(dictSectionSpeedLimit)
#     max = 1
#     min = 1
#     f = open("C:/Users/vluu\Dropbox/TELECOM-PARISTECH/Coyote_traffic_monitoring/outputCongestion"+str(timestep)+".csv", "w")
#     f.write('Id,Latitude,Longitude,Section,Ratio,Avg.Speed,MaxSpeed'+'\n')
#     for name in SectionCoordinateSet.intersection(SetSectionSpeedLimit):
#         x = SectionCoordinateDic[name][timestep]
#         if(len(x)>0):#if there is vehicle in the section at the timestamp
#             for i in x:
#                 temp = i.split('_')
#                 ratio = SectionSpeedDic[name][timestep] / float(dictSectionSpeedLimit[name])
#                 f.write(temp[0] + ',' +str(temp[1]) + ',' +str(temp[2]) + ',' + name + ',' + str(ratio) + ',' + str(SectionSpeedDic[name][timestep]) + ',' + dictSectionSpeedLimit[name] + '\n')
#                 if ratio > max:
#                     max = ratio
#                 if ratio < min:
#                     min = ratio
#
#     f.close()
#     print ("max: " + str(max) + "    "+"min: " + str(min))


def plotting(SectionSpeedDic,S,TimeFlow,Date,TimeMin,TimeMax,Speed):
    Titre = "Trafic for section " + S + "\n on " + Date + " from " + TimeMin + ' to ' + TimeMax
    plt.title(Titre)
    plt.ylabel('Traffic Data')
    plt.xlabel("Time (periods)")
    plt.ylim(0, 130)
    plt.xlim(0, 8)
    # pX=plt.plot(TimeFlow,Num,'b',marker='o')
    # pY=plt.plot(TimeFlow,Cars,'r',marker='v')
    pZ = plt.plot(TimeFlow, Speed, 'g', marker='x')
    # pD=plt.plot(TimeFlow,Speed2,'y',marker='o')
    # pF=plt.plot(TimeFlow,Flow,'m',marker='x')
    # plt.legend([pX,pY,pZ,PD],["Points","Cars","Av Speed","SD speed"])
    # LegNum = mlines.Line2D([], [], color='blue', marker='o',markersize=5, label='Trafic points')
    # LegCars = mlines.Line2D([], [], color='red', marker='v',markersize=5, label='# Cars')
    # LegSpeed = mlines.Line2D([], [], color='green', marker='x',markersize=5, label='Av speed')
    # LegSpeed2 = mlines.Line2D([], [], color='yellow', marker='o',markersize=5, label='Speed St. Dev')
    # LegFlow = mlines.Line2D([], [], color='magenta', marker='o',markersize=5, label='Flow')
    # plt.legend(handles=[LegNum,LegCars,LegSpeed,LegSpeed2])#,LegFlow])
    # plt.show()
    # fig = plt.gcf()
    # plt.savefig("D:/Speed_Output/" + S + ".png", dpi=None, orientation='landscape')
    # plt.clf()
    # plt.show()
    # print("result")
    return 0


def confidenceInterval(data, confidence):
    a = 1.0*np.array(data)
    n = len(a)
    se = np.std(a,ddof = 1)
    h = se * scipy.stats.norm.ppf(1-(1 - confidence)/2.) / np.sqrt(n)
    return h

def getAvgSpeedForEachTimestampAndDay(SectionSeparateSpeedDicWithDate):
    for k, v in SectionSeparateSpeedDicWithDate.items():#for each section k, get list of day_speed
        tempList = v
        newList = []
        for x in tempList:#for each timestamp
            tempArray = x[:-1].split(',')#get list of  day_speed
            result = getAvgSpeedForEachDayOfATimestamp(tempArray)
            if result != 'null':
                newList.append(result)
        SectionSeparateSpeedDicWithDate[k] = newList
    print()

def getAvgSpeedForEachDayOfATimestamp(tempArray):#for list day_speed of each timestamp
    if tempArray[0]=='':#if there is no data for that timestamp (of the section), even of 1 day
        return 'null'#there will be no data in dataset for that section timestamp
    else:
        tempDict = {}
        for i in tempArray:
            tempVar = i.split('_')#separate day and speed
            if tempVar[0] in tempDict:#if day is in the dictionary as key
                tempDict[tempVar[0]] += ','+tempVar[1]#add speed to the value of that key
            else:
                tempDict[tempVar[0]] = tempVar[1]#create item
        for k,v in tempDict.items():
            avg = getAvgOfArray(v)
            tempDict[k] = avg#avg. speed of the day
        return tempDict# so what we have here is, for the section timestamp, days and their corresponding average speeds

def getAvgOfArray(valueString):
    valueArray = np.array(valueString.split(',')).astype(np.float)
    return np.average(valueArray)

def getColumnIndexArray(SectionSpeedDic):
    columnIndexArray = []
    for k, v in SectionSeparateSpeedDicWithDate.items():
        for i in range(1,9):
            columnIndexArray.append(k + '_' + str(i))
    return columnIndexArray

def createObservationPredictorMatrix(dateArray,columnIndexArray,SectionSeparateSpeedDicWithDate):
    #cnt = 0
    PCAdataframe = pd.DataFrame(index=dateArray, columns=columnIndexArray)
    for k, v in SectionSeparateSpeedDicWithDate.items():
        timestamp = 1
        tempList = v
        for x in tempList:
            for date, avgSpeed in x.items():
                rowIndex = date
                columnIndex = k + '_' + str(timestamp)
                #cnt+=1
                #print(str(cnt))
                PCAdataframe.at[rowIndex,columnIndex] = avgSpeed
            timestamp = timestamp + 1

    #PCAdataframe.to_csv("originalDatframe.csv")
    #nullList = PCAdataframe.isnull().sum()
    #PCAdataframe.isnull().sum().to_csv("null.csv")
    #PCAdataframe.to_csv("TrafficData.csv", sep=',')
    #x = PCAdataframe.mean()
    return PCAdataframe.dropna(axis=1, how='all')#,nullList#.fillna(PCAdataframe.mean()),nullList#for ones contaning missing values, replace them by mean value

def data_to_plotly(coefs):#for LassoCV prediction
    y_ = []
    for col in range(0, len(coefs[0])):
        y_.append([])
        for row in range(0, len(coefs)):
            y_[col].append(coefs[row][col])
    return y_

# def PCAeigenPrediction(X_train):
#     std_matrix = StandardScaler().fit_transform(X_train)
#     cov_mat = np.cov(std_matrix.T)
#     # cov_mat_2 = np.dot(std_matrix_2,std_matrix_2.T)
#     eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#     idx = eig_vals.argsort()[::-1]
#     eigenValues = eig_vals[idx]
#     eigenVectors = eig_vecs[:, idx]
#     print("Eigen vectors:")
#     x = eigenVectors[0]
#     for v in eigenVectors:
#         print(v)
#         print()
#     # print("Eigen values:")
#     # print('Eigenvectors \n%s' %eigenVectors)
#     print('\nEigenvalues \n%s' % eigenValues)
#     # sympy_matrix = Matrix(cov_mat)
#     # print(sympy_matrix.eigenvects())
#     # print(sympy_matrix.eigenvals())
#     xx = np.linspace(0, 126, 126)#1473, 1473)  # consider x values 0, 1, .., 100
#     plt.plot(xx, eigenValues)
#     plt.xlabel('number of eigenvectors')
#     plt.ylabel('eigen values')
#     # plt.plot(xx, eigenValues[:,1],label="medium")
#     # plt.plot(xx, eigenValues[:,0],label="smallest")
#     plt.legend()
#     plt.show()
#     # eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
#     # print('Eigenvalues in descending order:')
#     # for i in eig_pairs:
#     # print(i[0],i[1])
#     print("PCA components:")
#     pca = PCA().fit(std_matrix)
#     for c in pca.components_:
#         print(c)
#         print()
#     print("----------------")
#     print(pca.explained_variance_)
#     print("----------------")
#     print(pca.explained_variance_ratio_)
#     print("----------------")
#     print(pca.explained_variance_ratio_.cumsum())
#
#     # plotting
#     plt.plot(pca.explained_variance_ratio_.cumsum())
#     plt.xlabel('number of components')
#     plt.ylabel('cumulative explained variance')
#     plt.show()

# def OLS(SectionListInterest,trainSet,testSet):
#     print ('OLS:')
#     for e in SectionListInterest:
#         X_train, y_train = getInterestSection(trainSet, str(e),5)
#         X_test, y_test = getInterestSection(testSet, str(e),5)
#         std_matrix = StandardScaler().fit_transform(X_train)
#         std_matrix_test = StandardScaler().fit_transform(X_test)
#         #if y_train == nan ----> to be added
#         reg = linear_model.LinearRegression()
#         reg.fit(std_matrix,y_train)
#         y_pred = reg.predict(std_matrix_test)
#         #plt.plot( y_true, y_pred , 'yo', color = "orange" )
#         #plt.show()


def LassoCVCoordinateDescent(X_train,y_train,cv):
    print("Figuring path of regularization using the coordinate descent lasso...")
    t1 = time.time()
    model = linear_model.LassoCV(cv=cv).fit(X_train,y_train)
    t_lasso_cv = time.time() - t1

    # Display results
    m_log_alphas = -np.log10(model.alphas_)

    plt.figure()
    ymin, ymax = 0, 2
    plt.plot(m_log_alphas, model.mse_path_, ':')
    plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
             label='Average across folds', linewidth=2)
    plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
                label='α: CV estimate')

    plt.legend()

    plt.xlabel('-log(α)')
    plt.ylabel('MSE')
    plt.title('MSE on each fold: coordinate descent '
              '(train time: %.2fs)' % t_lasso_cv)
    plt.axis('tight')
    plt.ylim(ymin, ymax)
    plt.show()

def LassoCVPrediction(X_train,y_train,cv):
    std_matrix = StandardScaler().fit_transform(X_train)#standardize predictors
    model = linear_model.LassoCV(cv=cv).fit(X_train,y_train)
    m_log_alphas = -np.log10(model.alphas_)
    t1 = time.time()
    t_lasso_cv = time.time() - t1
    data = []
    y_ = data_to_plotly(model.mse_path_)
    #model.alphas_ = np.append(model.alphas_,5)
    for i in range(0, len(y_)):
        p1 = go.Scatter(x=model.alphas_, y=y_[i], mode='lines', line=dict(dash='dot', width=1), showlegend=False)
        data.append(p1)
    p2 = go.Scatter(x=model.alphas_, y=model.mse_path_.mean(axis=-1), mode='lines', line=dict(color='black'), name='AVG. of MSE across folds')
    p3 = go.Scatter(x=2 * [model.alpha_], y=[0, 350], mode='lines', line=dict(color='black', dash='dashdot'), name='Estimated alpha')
    data.append(p2)
    data.append(p3)
    arr = np.where(model.alphas_==model.alpha_)[0]
    alphaMSE = np.mean(model.mse_path_[arr[0]])
    #alphaMSE = np.mean(model.mse_path_[int(str(*np.where(model.alphas_==model.alpha_)[0]))])
    ZeroMSE = np.mean(model.mse_path_[-1])
    layout = go.Layout(title='CV=%.2fs - MSE on each fold using coordinate descent (Training time: %.2fs)' % (cv, t_lasso_cv), hovermode='closest',
                       xaxis=dict(title='Lambda', zeroline=False), yaxis=dict(title='MSE', zeroline=False, range=[0, 350]))
    fig = go.Figure(data=data, layout=layout)
    #py.plot(fig)
    return model.alpha_,alphaMSE,model.alphas_[-1],ZeroMSE

def LassoCVCoefficient(X_train,y_train,X_test,y_test,alpha):
    #std_matrix = StandardScaler().fit_transform(X_train)
    #std_matrix_test = StandardScaler().fit_transform(X_test)
    lasso = linear_model.Lasso(alpha)
    clf = lasso.fit(X_train, y_train)
    predictionArray = clf.predict(X_test)
    #score = clf.score(std_matrix,y_train)
    score = clf.score(X_test,y_test)
    header = list(X_train)
    #model = SelectFromModel(lasso, prefit=True)
    #temp = model.get_support()
    covariateNumber = lasso.coef_.size
    #plt.xticks(range(step+1,covariateNumber+step+1))

    ###########################################################################################################
    #plt.xticks(range(len(header)),header,size='small',rotation=90)
    #plt.plot(lasso.coef_)
    #plt.xlabel("Timestamp")
    #plt.ylabel("Regression Coefficient")
    #title = "(α=" + str(alpha) + ") y = "
    #for i in lasso.coef_:
        #if (i!=0):
            #index = str(*np.where(lasso.coef_ == i)[0])
            #if(i>0):
                #title = title + "+" + str(i) + "(" + header[int(index)] + ")" #str(*temp[0]+step+1)+"x" + "+"
            #if (   i<0):
                #title = title + str(i) + "(" + header[int(index)] + ")"
    #title = title + ")"
    #if lasso.intercept_==0:
        #title = "\n".join(wrap(title[:-1],200))
    #if lasso.intercept_<0:
        #title = "\n".join(wrap(title[:-1]+str(lasso.intercept_),200))
    #if lasso.intercept_>0:
        #title = "\n".join(wrap(title[:-1]+"+"+ str(lasso.intercept_),200))
    #print(title)
    #plt.suptitle(title,size='small')
    #plt.show()
    ##############################################################################################################
    return lasso.coef_,lasso.intercept_,predictionArray,score


def getInterestSection(X_train,interestSection,h):
    vars = []
    for i in range(1,h):
        if interestSection+'_'+str(i) in X_train.columns:
            vars.append(interestSection+'_'+str(i))
        else:#if any section timestamp is not in the train set (missing data)
            flag=0
            count = 1
            while flag==0 and count < len(X_train.columns)/2.0 :#go left and right from the missing column to get the nearest avalable data until done or exceeded the series
                if interestSection+'_'+str(i+count) in X_train.columns:#if there is avialable data on left
                    vars.append(interestSection + '_' + str(i+count))#assign it to the missing column
                    flag = 1#done
                if interestSection+'_'+str(i-count) in X_train.columns:#if there is avialable data on right
                    vars.append(interestSection + '_' + str(i-count))#assign it to the missing column
                    flag = 1
                count+=1
    X_train_updated = X_train.filter(vars, axis=1)
    result = ''
    if interestSection+'_'+str(h) in X_train.columns:
        y_train = X_train[interestSection+'_'+str(h)]
        #print(confidenceInterval(y_train[interestSection+'_'+str(h)].tolist(), 0.95))
    else:
        y_train = nan
        print("Section without y_train:" + interestSection)
    return X_train_updated, y_train

space = {
        'learning_rate': hp.uniform('learning_rate', 0.005, 0.05),
        'max_depth': hp.quniform('max_depth', 8, 15, 1),
        'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
        'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
        'gamma': hp.quniform('gamma', 0.9, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 0.7, 0.05)
    }

def XGBoostEnhanced(SectionListInterest,dataSet,index,space):
    start = time.time()
    df_result_hyperopt = pd.DataFrame(columns=['score', 'estimators'] + list(space.keys()))
    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled = datasetFilled.iloc[0:20]
    output = 'XGB'
    for e in SectionListInterest:
        X_train, y_train = getInterestSection(trainSetFilled, str(e), index)
        X_test, y_test = getInterestSection(testSetFilled, str(e), index)

        def objective(space):
            global i
            i += 1
            print('>>')
            print(space)
            clf = xgb.XGBRegressor(n_estimators=10000,
                                    max_depth=int(space['max_depth']),
                                    learning_rate=space['learning_rate'],
                                    min_child_weight=space['min_child_weight'],
                                    subsample=space['subsample'],
                                    colsample_bytree=space['colsample_bytree'],
                                    gamma=space['gamma'])

            clf.fit(X_train, y_train,
                    eval_set=[(X_train, y_train), (X_test, y_test)],
                    eval_metric="auc", early_stopping_rounds=100)

            pred = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, pred)
            print("SCORE: %s" % auc)
            df_result_hyperopt.loc[i, ['score', 'estimators'] + list(space.keys())] = \
                [auc, clf.best_iteration] + list(space.values())
            return {'loss': 1. - auc, 'status': STATUS_OK}

        trials = Trials()
        best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
        print(df_result_hyperopt.sort(['score'], ascending=True).iloc[0])
    end = time.time()
    print(end - start)


def XGBoostForCategoricalData(SectionListInterest,dataSet,index):
    start = time.time()
    for column in dataSet.columns:
        if dataSet[column].dtype == type(object):
            le = LabelEncoder()
            dataSet[column] = le.fit_transform(dataSet[column].fillna('0'))

    trainSetFilled = dataSet.iloc[20:100]
    testSetFilled  = dataSet.iloc[0:20]
    output = 'XGB'
    for e in SectionListInterest:
        X_train, y_train = getInterestSection(trainSetFilled, str(e), index)
        X_test, y_test = getInterestSection(testSetFilled, str(e), index)
        svmModel = svm.SVC(kernel='linear', C=1, gamma=1)
        svmModel.fit(X_train, y_train)
        print(svmModel.score(X_train, y_train))
        # Predict Output
        y_pred = svmModel.predict(X_test)
        print(metric.accuracy_score(y_test, y_pred, normalize=False))
        mse = str(np.sqrt(metric.mean_squared_error(y_test, y_pred)))
        output += '\t' + str(mse)
    print(output)
    end = time.time()
    print(end - start)

def XGBoost(SectionListInterest,dataSet,index):
    start = time.time()
    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled  = datasetFilled.iloc[0:20]
    output = 'XGB'
    for e in SectionListInterest:
        X_train, y_train = getInterestSection(trainSetFilled, str(e), index)
        X_test, y_test = getInterestSection(testSetFilled, str(e), index)
        model = xgboost.XGBRegressor(max_depth=1,n_estimators=40,learning_rate=0.1)#(max_depth=5, learning_rate=0.1, n_estimators=2000)
        #clf = GridSearchCV(model, {'max_depth': [2, 4, 6], 'n_estimators': [50, 100, 200]}, verbose=1, scoring='neg_log_loss')
        #clf.fit( X_train, y_train)
        temp = model.fit(X_train, y_train)
        y_pred = temp.predict(X_test)
        #print(clf.best_score_, clf.best_params_)
        mse = str(np.sqrt(metric.mean_squared_error(y_test, y_pred)))
        output += '\t' + str(mse)
    print(output)
    end = time.time()
    print(end - start)


def XGBoostOnLasso(SectionListInterest,dataSet,index,cv):
    start = time.time()
    vars = []
    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled  = datasetFilled.iloc[0:20]
    output = 'XGB_LASSO'
    for e in SectionListInterest:
        X_train, y_train = getInterestSection(trainSetFilled, str(e), index)
        X_test, y_test = getInterestSection(testSetFilled, str(e), index)
        significantFeatures,sfm = LassoXGB(X_train, y_train,cv)
        significantFeaturesNumber = significantFeatures.shape[1]
        trainFeatures = X_train.columns.values
        if significantFeaturesNumber > 0:
            idx = trainFeatures[sfm.get_support()]
            for kid in idx:
                vars.append(kid)
            X_train_updated = X_train.filter(vars, axis=1)
            X_test_updated = X_test.filter(vars, axis=1)
            y_pred = xgboost.XGBRegressor(max_depth=1,n_estimators=40).fit(X_train_updated,y_train).predict(X_test_updated)
        else:
            y_pred = xgboost.XGBRegressor(max_depth=1,n_estimators=40).fit(X_train, y_train).predict(X_test)
        mse = str(np.sqrt(metric.mean_squared_error(y_test, y_pred)))
        output+= '\t' + str(mse)
    print(output)
    end = time.time()
    print(end - start)

def RandomForest(SectionListInterest,dataSet,index):
    start = time.time()
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled = datasetFilled.iloc[0:20]
    output = 'RF'
    for e in SectionListInterest:
        X_train, y_train = getInterestSection(trainSetFilled, str(e), index)
        X_test, y_test = getInterestSection(testSetFilled, str(e), index)
        model_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 10, cv = 3,
                                          verbose=2, random_state=42, n_jobs = -1)
        RFbestPr = model_random.best_params_
        print("Best RF param: " + RFbestPr)
        model = RandomForestRegressor(n_jobs=-1).set_params(**RFbestPr)
        #y_pred = model_random.fit(X_train, y_train)#.predict(X_test)

        #mse = str(round(metric.mean_squared_error(y_test, y_pred), 2))
        #output+= '\t' + str(mse)
    #print(output)
    #end = time.time()
    #print(end - start)

def Polynomial (SectionListInterest,dataSet,index):
    start = time.time()
    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled = datasetFilled.iloc[0:20]
    output = 'Poly'
    for e in SectionListInterest:
        print(e)
        X_train,y_train = getInterestSection(trainSetFilled,str(e),index)
        X_test,y_test = getInterestSection(testSetFilled,str(e),index)
        model = make_pipeline(PolynomialFeatures(2, interaction_only=True),linear_model.LassoCV(cv=30))#,max_iter=20000))
        y_pred = np.array(model.fit(X_train, y_train).predict(X_test))
        #RMSE = np.sqrt(np.sum(np.square(test_pred - y_test)))
        mse = str(round(metric.mean_squared_error(y_test, y_pred), 2))
        output += '\t' + str(mse)
    print(output)
    end = time.time()
    print(end - start)

def PolynomialNoPipeline(SectionListInterest,dataSet,index):
    #with ignore_warnings(category=ConvergenceWarning):
    start = time.time()
    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled = datasetFilled.iloc[0:20]

    output = ''
    for e in SectionListInterest:
        X_train,y_train = getInterestSection(trainSetFilled,str(e),index)
        X_test,y_test = getInterestSection(testSetFilled,str(e),index)
        poly = PolynomialFeatures(2,interaction_only=True)  # generate a polynomial object
        X_train_ = poly.fit_transform(X_train)
        X_test_ = poly.fit_transform(X_test)
        #print(poly.get_feature_names(X_train.columns))
        model = linear_model.LassoCV(cv=30,normalize=False)#,max_iter=20000))
        alpha = model.fit(X_train_, y_train).alpha_
        model2 = linear_model.Lasso(alpha)
        #columns = ['_'.join(['x{var}^{exp}'.format(var=var, exp=exp) for var, exp in enumerate(a[i, :])]) for i in range(a.shape[0])
        #zip(columns, model2.coef_)
        y_pred = np.array(model2.fit(X_train_, y_train).predict(X_test_))
        mse = str(round(metric.mean_squared_error(y_test, y_pred), 2))
        output += '\t' + str(mse)
    print(output)
    end = time.time()
    print(end - start)

def LassoXGB(X_train,y_train,cv):
    #std_matrix = StandardScaler().fit_transform(X_train)
    #y_train_std = StandardScaler().fit_transform(y_train)
    model = linear_model.LassoCV(cv=cv)
    sfm = SelectFromModel(model, threshold=0.25)
    sfm.fit(X_train,y_train)
    significantFeatures = sfm.transform(X_train)
    return significantFeatures,sfm

def LASSO(SectionListInterest,dataSet,index,correlationDf,cv):
    start = time.time()
    trainSet = dataSet.iloc[20:100]
    testSet = dataSet.iloc[0:20]
    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled = datasetFilled.iloc[0:20]
    output = 'LASSO'
    columnArray = []
    for e in SectionListInterest:
        X_train,y_train = getInterestSection(trainSet,str(e),index)
        X_test,y_test = getInterestSection(testSet,str(e),index)
        NullCountX_train = str(X_train.isnull().sum()[0]) #null values by count
        NullCounty_train = y_train.isnull().sum()
        NullCountX_test = str(X_test.isnull().sum()[0])
        NullCounty_test = y_test.isnull().sum()

        NullXPct = (100 - (X_train.isnull().sum()[0]+X_test.isnull().sum()[0]))/100#null values by percentage
        NullyPct = (100 - (y_train.isnull().sum() + y_test.isnull().sum()))/100

        X_train, y_train = getInterestSection(trainSetFilled, str(e), index)
        X_test, y_test = getInterestSection(testSetFilled, str(e), index)
        X_entire = pd.concat([X_train,X_test])
        y_entire = pd.concat([y_train,y_test])
        cI_X = str(round(confidenceInterval(X_entire.values, 0.95),2))
        cI_y = str(round(confidenceInterval(y_entire.values, 0.95),2))
        #X_train = X_train.fillna(X_train.mean())
        alpha, alphaMSE, minAlpha, ZeroMSE = LassoCVPrediction(X_train,y_train,cv)
        coef, intercept, y_pred, score = LassoCVCoefficient(X_train,y_train,X_test,y_test,alpha)
        mse = str(round(metric.mean_squared_error(y_test, y_pred), 2))
        output+= '\t' + str(mse)
    print(output)
    end = time.time()
    print(end - start)

def Average(SectionListInterest,dataSet,index):
    start = time.time()
    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled = datasetFilled.iloc[0:20]
    output = 'AVG'
    for e in SectionListInterest:
        X_train,y_train = getInterestSection(trainSetFilled,str(e),index)
        X_test,y_test = getInterestSection(testSetFilled,str(e),index)
        y_pred_avg = []
        for i in range(20):
            y_pred_avg.append(y_train.mean())
        mse = str(np.sqrt(metric.mean_squared_error(y_test, y_pred_avg)))
        output += '\t' + str(mse)
    print(output)
    end = time.time()
    print(end - start)

def getSignificantVarByCoefficient(coef):
    temp = []
    for idx, val in enumerate(coef):
        if val!=0:
            temp.append(idx)
    return temp#[:-1]

def carCountForSection(SectionCarDic):
    for key, value in SectionCarDic.items():
        #print(key)
        sum = 0
        for list in value:
            sum+=len(list)
        print(sum)

def getSpeedPointAndStdSummary(SectionSeparateSpeedDicWithDate):
    points = ''
    speeds = ''
    stds = ''
    for k, v in SectionSeparateSpeedDicWithDate.items():
        points+=k+','#for each section k, get list of day_speed
        speeds+=k+','
        stds += k + ','
        tempList = v
        list = []
        for x in tempList:#for each timestamp
            tempArray = x[:-1].split(',')#get list of  day_speed
            points += str(len(tempArray)) + ','
            sum=0
            for e in tempArray:
                day_speed = e.split('_')
                sum+=int(day_speed[1])
                list.append(int(day_speed[1]))
            avg = sum/len(tempArray)
            speeds += str(avg)+ ','
            stds += str(np.array(list).std())+ ','
        #fPoints.write(points[:-1]+'\n')
        #fSpeeds.write(speeds[:-1]+'\n')
        #fStds.write(stds[:-1]+'\n')
        points = ''
        speeds = ''
        stds = ''
    #fPoints.close()
    #fSpeeds.close()
    #fStds.close()

def getDiffOfXtestValsAndXtrainMean(a,b):
    sum = 0
    for ts in a:
        sum += np.square(ts - b)
    return sum

def LASSOAnomalyDetection(SectionListInterest, dataSet, index, correlationDf):
    datasetFilled = dataSet.fillna(dataSet.mean())
    trainSetFilled = datasetFilled.iloc[20:100]
    testSetFilled = datasetFilled.iloc[0:20]
    columnArray = []
    for e in SectionListInterest:
        X_train, y_train = getInterestSection(trainSetFilled, str(e), index)
        X_test, y_test = getInterestSection(testSetFilled, str(e), index)
        alpha, alphaMSE, minAlpha, ZeroMSE = LassoCVPrediction(X_train, y_train,cv)
        coef, intercept, y_pred, score = LassoCVCoefficient(X_train, y_train, X_test, y_test, alpha)
        list1 = []
        list2 = []
        for i in range(len(y_pred)):
            diff1 = abs(y_pred[i] - y_test[i])#diff threshold to define outliers will be set up later, this is primarily for visualization
            X_train_mean = float(X_train.values.mean())
            #diff2 = abs((X_test.values[i][0]) - X_train_mean) #for one timestamp to predict the previous one (h-1,h for getInterestSection) using loop
            diff2 = getDiffOfXtestValsAndXtrainMean(X_test.values[i],X_train_mean)  #for 7 timestamp to predict the last one (1,h for getInterestSection)
            list1.append(diff1)
            list2.append(diff2)
        columnArray.append(np.corrcoef(list1, list2)[0][1])
    correlationDf[index] = columnArray

def SetNullCount(SectionListInterest,dataSet,index):
    trainSet = dataSet.iloc[20:100]
    testSet = dataSet.iloc[0:20]
    output = ''
    for e in SectionListInterest:
        X_train,y_train = getInterestSection(trainSet,str(e),index)
        X_test,y_test = getInterestSection(testSet,str(e),index)
        NullCountX_train = X_train.isnull().sum()[0]
        NullCounty_train = y_train.isnull().sum()
        NullCountX_test = X_test.isnull().sum()[0]
        NullCounty_test = y_test.isnull().sum()
        NullXPct = (100 - (X_train.isnull().sum()[0]+X_test.isnull().sum()[0]))/100
        NullyPct = (100 - (y_train.isnull().sum() + y_test.isnull().sum()))/100
        sum = NullCountX_train+NullCounty_train+NullCountX_test+NullCounty_test
        output += '\t' + str(NullCounty_test)
    print(output)

SectionSpeedDic, SectionSeparateSpeedDic, SectionCarDic, SectionCoordinateDic, SectionSeparateSpeedDicWithDate, dateArray = \
getSectionSpeedDic(Seed, TimeStep, TimeMin, TimeMax)
#highCorrelationSectionList = getListCorrelationCoefficientAndStuff(SectionSpeedDic,SectionCarDic,1)
#outputFileConfidenceInterval(highCorrelationSectionList,SectionSeparateSpeedDic)
#carCountForSection(SectionCarDic)
#getSpeedPointAndStdSummary(SectionSeparateSpeedDicWithDate)
#---------------------
getAvgSpeedForEachTimestampAndDay(SectionSeparateSpeedDicWithDate)#after this function, there will be avg. speed of each section timestamp for each day, not speed values of section tomestamps of that day
columnIndexArray = getColumnIndexArray(SectionSpeedDic)#get header
observationPredictorMatrix = createObservationPredictorMatrix(dateArray,columnIndexArray,SectionSeparateSpeedDicWithDate)#create the appropriate dataframe for mining
#----------------------
trainSet = observationPredictorMatrix.iloc[20:100]
testSet = observationPredictorMatrix.iloc[0:20]
#getAvgCI(SectionListInterest,trainSet,testSet)
#Ridge(SectionListInterest,trainSet,testSet)
#OLS(SectionListInterest,trainSet,testSet)

correlationDf = pd.DataFrame(index=SectionListInterest, columns=[2, 3, 4, 5, 6, 7, 8])
#Average(SectionListInterest, observationPredictorMatrix,8)
#XGBoostOnLasso(SectionListInterest, observationPredictorMatrix,8)
#XGBoostEnhanced(SectionListInterest,observationPredictorMatrix,8,space)
XGBoost(SectionListInterest, observationPredictorMatrix,8,cv)
SetNullCount(SectionListInterest,observationPredictorMatrix,8)
#RandomForest(SectionListInterest, observationPredictorMatrix, 8)
#LASSO(SectionListInterest,observationPredictorMatrix,8,correlationDf,cv)
#Polynomial(SectionListInterest,observationPredictorMatrix,8)#,correlationDf)
#PolynomialNoPipeline(SectionListInterest,observationPredictorMatrix,8)
#----------------------------------------------------------
#LassoCVCoordinateDescent(X_train7,y_train7)
#----------------------
#X_train = pd.concat([X_train1,X_train2,X_train3],axis=1)
