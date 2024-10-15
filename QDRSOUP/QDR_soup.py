import os
import sys
import multiprocessing
import argparse
import numpy as np
from datetime import datetime
import random

from soup_utils import E_calculate, soup_generate, one_soup_generate, E_move, ratio_resample
from one_slack_ssvm_normal import OneSlackSSVM as OneSlackSSVM_normal


sys.path.insert(0, '..')
from SP import SP_USCO
from Steiner import Steiner_USCO
from StoGraph import StoGraph
from USCO_Solver import USCO_Solver
from Utils import Utils


class Object(object):
    pass


def main(soup_i):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--sourceT', default='Steiner',
        choices=['SP', 'Steiner'])

    parser.add_argument(
        '--targetT', default='Steiner',
        choices=['SP', 'Steiner'])

    parser.add_argument(
        '--graph', default='kro_2',
        choices=['col_2', 'kro_2', 'bay_2', 'col_3'])

    parser.add_argument(
        '--graphType', default='tru',
        choices=['tru', 'true'])

    parser.add_argument(
        '--featureNum', type=int, default=40,
        help='number of features (random subgraphs) used in StratLearn ')
    parser.add_argument(
        '--featureGenMethod', default='WeibullRandom',
        choices=['WeibullRandom'],
        help='the distribution used for generating features, the choices correspond phi_1^1, phi_0.01^1, phi_0.005^1, phi_+^+')

    parser.add_argument(
        '--trainNum', type=int, default=80, help='number of training data')

    parser.add_argument(
        '--testNum', type=int, default=960, help='number of testing data')

    parser.add_argument(
        '--testBatch', type=int, default=1, help='number of testing data')

    parser.add_argument(
        '--thread', type=int, default=64, help='number of threads')

    parser.add_argument(
        '--beta', type=float, default=1, help='number of threads')

    parser.add_argument(
        '--output', default=False, action="store_true", help='if output prediction')

    parser.add_argument(
        '--pre_train', default=True, action="store_true", help='if store a pre_train model')

    parser.add_argument(
        '--log_path', default='log', help='if store a pre_train model')

    args = parser.parse_args()

    sourceT = args.sourceT
    targetT = args.targetT

    graph = args.graph
    graphType = args.graphType

    trainNum = args.trainNum
    testNum = args.testNum
    testBatch = args.testBatch

    thread = args.thread

    verbose = 3
    C = 0.0001
    tol = 0.001
    max_iter = 10
    beta = args.beta
    featureNum = args.featureNum
    featureGenMethod = args.featureGenMethod



    if graph == "col_2":
        vNum = 512
        eNum = 551
        if sourceT == "SP":
            sourceSample_maxPair = 10000
            sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        else:
            if sourceT == "Steiner":
                sourceSample_maxPair = 10000
                sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
            else:
                sys.exit("wrong source task type")

        if targetT == "SP":
            targetSample_maxPair = 10000
            targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

        else:
            if targetT == "Steiner":
                targetSample_maxPair = 10000
                targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)
            else:
                sys.exit("wrong target task type")

    if graph == "kro_2":
        vNum = 1024
        eNum = 2745
        if sourceT == "SP":
            sourceSample_maxPair = 10000
            sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        else:
            if sourceT == "Steiner":
                sourceSample_maxPair = 10000
                sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
            else:
                sys.exit("wrong source task type")

        if targetT == "SP":
            targetSample_maxPair = 10000
            targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

        else:
            if targetT == "Steiner":
                targetSample_maxPair = 10000
                targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)
            else:
                sys.exit("wrong target task type")

    if graph == "col_3":
        vNum = 512
        eNum = 1551
        if sourceT == "SP":
            sourceSample_maxPair = 10000
            sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        else:
            if sourceT == "Steiner":
                sourceSample_maxPair = 10000
                sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
            else:
                sys.exit("wrong source task type")

        if targetT == "SP":
            targetSample_maxPair = 10000
            targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

        else:
            if targetT == "Steiner":
                targetSample_maxPair = 10000
                targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)
            else:
                sys.exit("wrong target task type")

    if graph == "bay_2":
        vNum = 1056
        eNum = 1570
        sourceSample_maxPair = 10000
        sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        targetSample_maxPair = 10000
        targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)
    if graph == "com_64":
        vNum = 64
        eNum = 4096
        sourceSample_maxPair = 10000
        sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        targetSample_maxPair = 10000
        targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

    maxFeatureNum = 10000

    pre_train = args.pre_train
    preTrainPathResult = None

    path = os.getcwd()
    data_path = os.path.dirname(path) + "/data"
    sourceSample_path = "{}/{}/{}".format(data_path, graph, sourceSampleFileName)
    targetSample_path = "{}/{}/{}".format(data_path, graph, targetSampleFileName)
    stoGraphPath = "{}/{}/{}_{}_stoGraph".format(data_path, graph, graph, graphType)
    featurePath = "{}/{}/features/{}_{}".format(data_path, graph, featureGenMethod, maxFeatureNum)
    if args.log_path is not None:
        logpath = path + "/log"

    stoGraph = StoGraph(stoGraphPath, vNum, graphType)

    if sourceT == "SP":
        source_USCO = SP_USCO(stoGraph)
    if sourceT == "Steiner":
        source_USCO = Steiner_USCO(stoGraph)
    if targetT == "SP":
        target_USCO = SP_USCO(stoGraph)
    if targetT == "Steiner":
        target_USCO = Steiner_USCO(stoGraph)


    #

    TrainSamples, TrainQueries, TrainDecisions, = source_USCO.readSamples(sourceSample_path, trainNum, trainNum)

    TestSamples_s, TestQueries_s, TestDecisions_s = [], [], []
    for i in range(testBatch):
        TestSamples, TestQueries, TestDecisions = target_USCO.readSamples(targetSample_path, testNum,
                                                                          targetSample_maxPair)
        TestSamples_s.append(TestSamples)
        TestQueries_s.append(TestQueries)
        TestDecisions_s.append(TestDecisions)



    print("data fetched")

    Utils.writeToFile(logpath, "data fetched")

    realizations, realizationIndexes = source_USCO.readRealizations(featurePath, featureNum, maxNum=maxFeatureNum)


    train_start_time = datetime.now()

    usco_Solver = USCO_Solver()
    usco_Solver.initialize(realizations, source_USCO)


    trainMethod = "one_slack_normal"
    one_slack_svm = OneSlackSSVM_normal(usco_Solver, verbose=verbose, C=C, tol=tol, n_jobs=thread, max_iter=max_iter,
                                        log=logpath)
    one_slack_svm.fit(TrainQueries, TrainDecisions, beta, initialize=False)
    w = one_slack_svm.w


    train_end_time = datetime.now()
    train_time = train_end_time - train_start_time
    print(train_time)

    if pre_train is True:
        now = datetime.now()
        preTrainPath = path + "/pre_train" + str(featureNum) +"_" +str(trainNum) +"/" + str(soup_i) + "/"
        if not os.path.exists(preTrainPath):
            os.makedirs(preTrainPath)
        Utils.save_pretrain(preTrainPath, w, realizationIndexes, featurePath)
        preTrainEgraph = preTrainPath + "/Egraph"

    E_calculate(eNum,preTrainPath,featurePath,stoGraphPath)

    return parser


def soup_main(parser,soup_number):
    args = parser.parse_args()

    sourceT = args.sourceT
    targetT = args.targetT

    graph = args.graph
    graphType = args.graphType

    trainNum = args.trainNum
    testNum = args.testNum
    testBatch = args.testBatch

    thread = args.thread

    soup_number = soup_number
    verbose = 3
    C = 0.0001
    tol = 0.001
    max_iter = 10
    beta = args.beta
    featureNum = args.featureNum
    featureGenMethod = "soup"



    if graph == "col_2":
        vNum = 512
        eNum = 551
        if sourceT == "SP":
            sourceSample_maxPair = 10000
            sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        else:
            if sourceT == "Steiner":
                sourceSample_maxPair = 10000
                sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
            else:
                sys.exit("wrong source task type")

        if targetT == "SP":
            targetSample_maxPair = 10000
            targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

        else:
            if targetT == "Steiner":
                targetSample_maxPair = 10000
                targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)
            else:
                sys.exit("wrong target task type")

    if graph == "kro_2":
        vNum = 1024
        eNum = 2745
        if sourceT == "SP":
            sourceSample_maxPair = 10000
            sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        else:
            if sourceT == "Steiner":
                sourceSample_maxPair = 10000
                sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
            else:
                sys.exit("wrong source task type")

        if targetT == "SP":
            targetSample_maxPair = 10000
            targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

        else:
            if targetT == "Steiner":
                targetSample_maxPair = 10000
                targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)
            else:
                sys.exit("wrong target task type")

    if graph == "col_3":
        vNum = 512
        eNum = 1551
        if sourceT == "SP":
            sourceSample_maxPair = 10000
            sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        else:
            if sourceT == "Steiner":
                sourceSample_maxPair = 10000
                sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
            else:
                sys.exit("wrong source task type")

        if targetT == "SP":
            targetSample_maxPair = 10000
            targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

        else:
            if targetT == "Steiner":
                targetSample_maxPair = 10000
                targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)
            else:
                sys.exit("wrong target task type")

    if graph == "bay_2":
        vNum = 1056
        eNum = 1570
        sourceSample_maxPair = 10000
        sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        targetSample_maxPair = 10000
        targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)
    if graph == "com_64":
        vNum = 64
        eNum = 4096
        sourceSample_maxPair = 10000
        sourceSampleFileName = "{}_{}_{}_samples".format(graph, graphType, sourceT)
        targetSample_maxPair = 10000
        targetSampleFileName = "{}_{}_{}_samples".format(graph, graphType, targetT)

    maxFeatureNum = featureNum

    pre_train = args.pre_train
    preTrainPathResult = None

    path = os.getcwd()
    data_path = os.path.dirname(path) + "/data"
    sourceSample_path = "{}/{}/{}".format(data_path, graph, sourceSampleFileName)
    targetSample_path = "{}/{}/{}".format(data_path, graph, targetSampleFileName)
    stoGraphPath = "{}/{}/{}_{}_stoGraph".format(data_path, graph, graph, graphType)
    fpath = os.path.dirname(path)
    E_list = soup_generate(soup_number,fpath,graph,featureGenMethod,maxFeatureNum,featureNum,trainNum)
    featurePath = "{}/{}/features/{}_{}".format(data_path, graph, featureGenMethod, maxFeatureNum)
    print(featurePath)
    if args.log_path is not None:
        logpath = path + "/log"

    stoGraph = StoGraph(stoGraphPath, vNum, graphType)

    if sourceT == "SP":
        source_USCO = SP_USCO(stoGraph)
    if sourceT == "Steiner":
        source_USCO = Steiner_USCO(stoGraph)
    if targetT == "SP":
        target_USCO = SP_USCO(stoGraph)
    if targetT == "Steiner":
        target_USCO = Steiner_USCO(stoGraph)

    #

    TrainSamples, TrainQueries, TrainDecisions, = source_USCO.readSamples(sourceSample_path, trainNum, trainNum)

    TestSamples_s, TestQueries_s, TestDecisions_s = [], [], []
    for i in range(testBatch):
        TestSamples, TestQueries, TestDecisions = target_USCO.readSamples(targetSample_path, testNum,
                                                                          targetSample_maxPair)
        TestSamples_s.append(TestSamples)
        TestQueries_s.append(TestQueries)
        TestDecisions_s.append(TestDecisions)



    print("data fetched")

    Utils.writeToFile(logpath, "data fetched")

    realizations, realizationIndexes = source_USCO.readRealizations(featurePath, featureNum, maxNum=maxFeatureNum)


    train_start_time = datetime.now()

    usco_Solver = USCO_Solver()
    usco_Solver.initialize(realizations, source_USCO)


    trainMethod = "one_slack_normal"
    one_slack_svm = OneSlackSSVM_normal(usco_Solver, verbose=verbose, C=C, tol=tol, n_jobs=thread, max_iter=max_iter,
                                        log=logpath)
    one_slack_svm.fit(TrainQueries, TrainDecisions, beta, initialize=False)
    w = one_slack_svm.w


    train_end_time = datetime.now()
    train_time = train_end_time - train_start_time
    print(train_time)

    if pre_train is True:
        now = datetime.now()
        preTrainPath = path + "/pre_train_soup/" + now.strftime("%d-%m-%Y-%H-%M-%S") + "/"
        if not os.path.exists(preTrainPath):
            os.makedirs(preTrainPath)
        Utils.save_pretrain(preTrainPath, w, realizationIndexes, featurePath)
        Utils.writeToFile(logpath, preTrainPath, toconsole=True)
        preTrainPathResult = preTrainPath + "/soupresult1"
    Utils.writeToFile(logpath, "===============================================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, "Testing ...", toconsole=True, preTrainPathResult=preTrainPathResult)


    Utils.writeToFile(logpath, sourceT + " " + targetT, toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "{}_{}".format(graph, graphType), toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "featureNum: {}, featureGenMethod: {}, trainMethod: {}, beta: {}".format(featureNum,
                                                                                                        featureGenMethod,
                                                                                                        trainMethod,
                                                                                                        beta),
                      toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum: {} ".format(trainNum, ), toconsole=True,
                      preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "testNum:{} ".format(testNum), toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "maxIter: {}, c: {} ".format(max_iter, C), toconsole=True,
                      preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, "=================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)

    for TestSamples, TestQueries, TestDecisions in zip(TestSamples_s, TestQueries_s, TestDecisions_s):
        predDecisions = target_USCO.solve_R_batch(TestQueries, w, realizations, n_jobs=thread, offset=None, trace=False)
        ratio_pred = target_USCO.test(TestSamples, TestQueries, TestDecisions, predDecisions, thread, logpath=logpath,
                                      preTrainPathResult=preTrainPathResult)


    Utils.writeToFile(logpath, "=================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "All ones", toconsole=True, preTrainPathResult=preTrainPathResult)
    for TestSamples, TestQueries, TestDecisions in zip(TestSamples_s, TestQueries_s, TestDecisions_s):
        randomW = np.ones(featureNum)
        randPredDecisions = target_USCO.solve_R_batch(TestQueries, randomW, realizations, n_jobs=thread, offset=None,
                                                      trace=False)
        ratio_ones = target_USCO.test(TestSamples, TestQueries, TestDecisions, randPredDecisions, thread,
                                      logpath=logpath, preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, "=================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)





    E_move(E_list,path,graph,soup_number)

    featurePath = "{}/{}/features/{}_{}".format(data_path, graph, "EG", soup_number)
    realizations, realizationIndexes = source_USCO.readRealizations(featurePath, soup_number, maxNum=soup_number)

    train_start_time = datetime.now()

    usco_Solver = USCO_Solver()
    usco_Solver.initialize(realizations, source_USCO)

    trainMethod = "one_slack_normal"
    one_slack_svm = OneSlackSSVM_normal(usco_Solver, verbose=verbose, C=C, tol=tol, n_jobs=thread, max_iter=max_iter,
                                        log=logpath)
    one_slack_svm.fit(TrainQueries, TrainDecisions, beta, initialize=False)
    w = one_slack_svm.w

    train_end_time = datetime.now()
    train_time = train_end_time - train_start_time
    print(train_time)

    preTrainPathResult = preTrainPath + "/EG_result1"

    Utils.writeToFile(logpath, "===============================================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, "Testing ...", toconsole=True, preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, sourceT + " " + targetT, toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "{}_{}".format(graph, graphType), toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "featureNum: {}, featureGenMethod: {}, trainMethod: {}, beta: {}".format(featureNum,
                                                                                                        featureGenMethod,
                                                                                                        trainMethod,
                                                                                                        beta),
                      toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "trainNum: {} ".format(trainNum, ), toconsole=True,
                      preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "testNum:{} ".format(testNum), toconsole=True, preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "maxIter: {}, c: {} ".format(max_iter, C), toconsole=True,
                      preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, "=================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)

    for TestSamples, TestQueries, TestDecisions in zip(TestSamples_s, TestQueries_s, TestDecisions_s):
        predDecisions = target_USCO.solve_R_batch(TestQueries, w, realizations, n_jobs=thread, offset=None, trace=False)
        ratio_pred = target_USCO.test(TestSamples, TestQueries, TestDecisions, predDecisions, thread, logpath=logpath,
                                      preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, "=================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)
    Utils.writeToFile(logpath, "All ones", toconsole=True, preTrainPathResult=preTrainPathResult)
    for TestSamples, TestQueries, TestDecisions in zip(TestSamples_s, TestQueries_s, TestDecisions_s):
        randomW = np.ones(soup_number)
        randPredDecisions = target_USCO.solve_R_batch(TestQueries, randomW, realizations, n_jobs=thread, offset=None,
                                                      trace=False)
        ratio_ones = target_USCO.test(TestSamples, TestQueries, TestDecisions, randPredDecisions, thread,
                                      logpath=logpath, preTrainPathResult=preTrainPathResult)

    Utils.writeToFile(logpath, "=================================", toconsole=True,
                      preTrainPathResult=preTrainPathResult)

if __name__ == "__main__":
    soup_number = 10
    for soup_i in range(soup_number):
        print(soup_i)
        parser = main(soup_i)
    soup_main(parser,soup_number)
