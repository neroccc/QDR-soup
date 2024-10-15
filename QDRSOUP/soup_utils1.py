import math
import os
import random
import sys
import math
import scipy
import scipy.integrate as integrate
import scipy.optimize as optimize
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import laplacian_kernel
def gen_matrix(file,vNum):
    infile = open(file, 'r')
    # Create a 2D array with all values as -1
    array = -1 * np.ones((vNum, vNum), dtype=int)

    # Set the diagonal values to 0
    np.fill_diagonal(array, 0)
    while True:
        line = infile.readline()
        if not line:
            infile.close()
            break
        ints = line.split()
        i = int(ints[0])
        j = int(ints[1])
        w = float(ints[4])
        array[i, j] = w
    return array

def E_calculate(eNum,preTrainPath,featurePath,stoGraphPath):
    w = [0] * eNum
    infile = open(preTrainPath + "/featureIndex", 'r')
    outfolder = featurePath +"/"
    while True:
        line = infile.readline()
        if not line:
            infile.close()
            break
        ints = line.split()
        number = ints[0]
        weight = float(ints[1])
        inpath = "{}{}".format(outfolder, number)
        with open(inpath, 'r') as feature:
            i = 0
            while True:
                line = feature.readline()
                if not line:
                    feature.close()
                    break
                ints = line.split()
                node_weight = float(ints[2])
                w[i] = w[i] + weight * node_weight
                i = i + 1
    infile.close()
    infile = open(stoGraphPath, 'r')
    outfile = open(preTrainPath + "/Egraph", 'w')
    mini_vaule = min(w)
    nor_r = 3 / mini_vaule
    for i in range(len(w)):
        w[i] = w[i] * nor_r
    i = 0
    while True:
        line = infile.readline()
        if not line:
            infile.close()
            break
        ints = line.split()
        mean = math.ceil(float(w[i]))
        node1 = ints[0]
        node2 = ints[1]
        alpha = random.randint(1, 10)
        tag = 1
        while(tag != 0):
            try:
                beta = estimate_beta(alpha, mean)
                tag = 0
            except ValueError:
                print(mean)
                alpha = alpha + 1
        outfile.write(str(node1) + " ")
        outfile.write(str(node2) + " ")
        outfile.write(str(alpha) + " ")
        outfile.write(str(beta) + " ")
        outfile.write(str(mean) + "\n")
        i = i + 1
    outfile.close()

def estimate_beta(alpha, expected_time):
    # Define the distribution of time
    def time_dist(x, alpha, beta):
        return alpha * math.pow(-math.log(1 - x), beta)

    # Define the objective function to minimize
    def objective(beta):
        integrand = lambda x: time_dist(x, alpha, beta) * x ** beta * (-math.log(1 - x))
        estimated_time = integrate.quad(integrand, 0, 1)[0]
        return (estimated_time - expected_time) ** 2

    # Find the value of beta that minimizes the objective function
    result = optimize.minimize_scalar(objective, method='bounded', bounds=(0.01, 10))
    beta_estimate = result.x

    return math.ceil(beta_estimate)

def soup_generate(soup_number,path,graph,featureGenMethod,maxFeatureNum,featureNum,trainNum):
    folder_path = path + "/QRTS/pre_train" + str(featureNum) +"_" +str(trainNum)
    items = os.listdir(folder_path)
    folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    folders_value = len(folders)
    E_list = []
    folder_Nums = (np.random.permutation(folders_value))[0:soup_number]
    for item in folder_Nums:
        select_folder_path = os.path.join(folder_path, folders[item])
        Egraph = select_folder_path + "/Egraph"
        E_list.append(Egraph)
    data_path = path + "/data"
    featurePath = "{}/{}/features/{}_{}/".format(data_path, graph, featureGenMethod, maxFeatureNum)
    genMultiRealization(maxFeatureNum/soup_number, featurePath, E_list, startIndex=0)
    return E_list

def getWeibull_log_likelihood(alpha, beta, c):
    log_likelihood = math.log(beta) - beta * math.log(alpha) + (beta - 1) * math.log(c) - (c / alpha) ** beta
    return log_likelihood

def genNxGraph(file,vNum):
        g = nx.Graph()
        for i in range(vNum):
            g.add_node(str(i))
        infile = open(file, 'r')
        while True:
            line = infile.readline()
            if not line:
                infile.close()
                break
            ints = line.split()
            i = ints[0]
            j = ints[1]
            w = float(ints[4])
            g.add_edge(str(i), str(j), weight=w)
        return g

def file_compare(soupG_s,resG_s):
    soupG = open(soupG_s,"r")
    resG = open(resG_s,"r")
    lines1 = soupG.readlines()
    lines2 = resG.readlines()
    p1 = 0
    for i in range(len(lines1)):
        string1 = lines1[i].split(" ")
        string2 = lines2[i].split(" ")
        alpha = int(string1[2])
        beta = int(string1[3])
        center = float(string2[4])
        p1 = p1 + getWeibull_log_likelihood(alpha,beta,center)
    return p1

def compare( g_1, g_2):
    """Compute the kernel value (similarity) between two graphs.

    Parameters
    ----------
    g1 : networkx.Graph
        First graph.
    g2 : networkx.Graph
        Second graph.

    Returns
    -------
    k : The similarity value between g1 and g2.
    """
    # Diagonal superior matrix of the floyd warshall shortest
    # paths:
    fwm1 = np.array(nx.floyd_warshall_numpy(g_1))
    fwm1 = np.where(fwm1 == np.inf, 0, fwm1)
    fwm1 = np.where(fwm1 == np.nan, 0, fwm1)
    fwm1 = np.triu(fwm1, k=1)
    bc1 = np.bincount(fwm1.reshape(-1).astype(int))

    fwm2 = np.array(nx.floyd_warshall_numpy(g_2))
    fwm2 = np.where(fwm2 == np.inf, 0, fwm2)
    fwm2 = np.where(fwm2 == np.nan, 0, fwm2)
    fwm2 = np.triu(fwm2, k=1)
    bc2 = np.bincount(fwm2.reshape(-1).astype(int))

    # Copy into arrays with the same length the non-zero shortests
    # paths:
    v1 = np.zeros(max(len(bc1), len(bc2)) - 1)
    v1[range(0, len(bc1) - 1)] = bc1[1:]

    v2 = np.zeros(max(len(bc1), len(bc2)) - 1)
    v2[range(0, len(bc2) - 1)] = bc2[1:]

    return np.sum(v1 * v2)


def compare_normalized(g_1, g_2, verbose=False):
    return compare(g_1, g_2) / (np.sqrt(compare(g_1, g_1) *
                                             compare(g_2, g_2)))

def file_compare_v1(soupG_s,resG_s,vNum):
    graph1 = genNxGraph(soupG_s,vNum)
    graph2 = genNxGraph(resG_s,vNum)
    try:
        kernel_score = compare_normalized(graph1, graph2)
    except ValueError:
        kernel_score = 0.2
    print(kernel_score)
    return kernel_score

def file_compare_v2(soupG_s,resG_s,vNum):
    graph1 = gen_matrix(soupG_s,vNum)
    graph2 = gen_matrix(resG_s,vNum)
    try:
        kernel_score = laplacian_kernel(graph1, graph2, gamma=0.1).mean()
    except ValueError:
        kernel_score = 0.2
    print(kernel_score)
    return kernel_score


def ratio_resample(soup_number,path,graph,featureGenMethod,maxFeatureNum,featureNum,trainNum,preTrainPath,vNum):
    folder_path = path + "/pre_train" + str(featureNum) +"_" +str(trainNum)
    revistied_path = preTrainPath
    items = os.listdir(folder_path)
    folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item))]
    folders_value = len(folders)
    E_list = []
    ratio_list = []
    folder_Nums = (np.random.permutation(folders_value))[0:soup_number]
    for item in folder_Nums:
        select_folder_path = os.path.join(folder_path, folders[item])
        Egraph = select_folder_path + "/Egraph"
        com_Graph = revistied_path + "/Egraph"
        E_list.append(Egraph)
        ratio = file_compare_v2(Egraph,com_Graph,vNum)
        ratio_list.append(ratio)
    data_path = path[0:-5] + "/data"
    featurePath = "{}/{}/features/{}_{}/".format(data_path, graph, featureGenMethod, maxFeatureNum)
    genMultiRealizationratio(maxFeatureNum, ratio_list, featurePath, E_list, startIndex=0)
    return E_list

def ten_exp(maxFeatureNum,ratio_list):
    log_values = ratio_list
    pair = []
    for i in range(10):
        pair.append(i)
    # Calculate the maximum logarithmic value
    max_log = max(log_values)

    # Calculate the sum of exponential differences in logarithmic space
    exp_diff_sum_log = math.log(sum(math.exp(log_val - max_log) for log_val in log_values))

    sam = []
    # Calculate the approximate ratio in logarithmic space
    for i in range(10):
        approx_ratio_log = log_values[i] - max_log - exp_diff_sum_log

    # Calculate the approximate result by converting logarithmic space to actual value
        result = math.exp(approx_ratio_log)
        sam.append(math.ceil(result*maxFeatureNum))
    difference = sum(sam) - maxFeatureNum
    random.shuffle(pair)
    print(sam)
    for i in range(difference):
        num = pair[i]
        sam[num] = sam[num]-1
    return sam
def ten_ratio(maxFeatureNum,ratio_list):
    log_values = ratio_list
    pair = []
    max_log = max(log_values)
    for i in range(10):
        pair.append(i)
        log_values[i] = log_values[i] + max_log
    sum_log = sum(log_values)

    sam = []
    # Calculate the approximate ratio in logarithmic space
    for i in range(10):
        approx_ratio_log = log_values[i]/sum_log
    # Calculate the approximate result by converting logarithmic space to actual value
        result = approx_ratio_log
        sam.append(math.ceil(result*maxFeatureNum))
    difference = sum(sam) - maxFeatureNum
    random.shuffle(pair)
    print(sam)
    for i in range(difference):
        num = pair[i]
        sam[num] = sam[num]-1
    return sam

def genMultiRealizationratio(maxFeatureNum, ratio_list, outfolder, E_list, startIndex=0):
    r = len(E_list)
    #num = ten_exp(maxFeatureNum,ratio_list)
    num = ten_ratio(maxFeatureNum,ratio_list)
    startIndex = 0
    for index in range(r):
        for cout in range(num[index]):
            path = "{}{}".format(outfolder,startIndex)
            startIndex = startIndex + 1
            genOneRealizationTrue(path,E_list[index])

def getWeibull(alpha, beta):
    time = alpha * math.pow(-math.log(1 - random.uniform(0, 1)), beta)
    if time >= 0:
        return math.ceil(time) + 1
    else:
        sys.exit("time <0")
        return None

def genMultiRealization(num, outfolder, E_list, startIndex=0):
    r = len(E_list)
    num = int(num)
    for index in range(r):
        for cout in range(num):
            path = "{}{}".format(outfolder, num*index + cout + startIndex)
            genOneRealizationTrue(path,E_list[index])

def genOneRealizationTrue(outpath,index):
    with open(outpath, 'w') as outfile:
        infile = open(index, 'r')
        while True:
            line = infile.readline()
            if not line:
                infile.close()
                break
            ints = line.split()
            node1 = ints[0]
            node2 = ints[1]
            alpha = ints[2]
            beta = ints[3]
            mean = ints[4]
            weight = getWeibull(float(alpha), float(beta))
            outfile.write(node1 + " ")
            outfile.write(node2 + " ")
            outfile.write(str(weight) + "\n")
        outfile.close()

def one_soup_generate(preTrainPath,path,graph,featureNum):
    data_path = os.path.dirname(path) + "/data"
    featurePath = "{}/{}/features/{}_{}/".format(data_path, graph, "prior", featureNum)

    genMultiRealization_prior(featureNum, featurePath, preTrainPath, startIndex=0)

def genMultiRealization_prior(num, outfolder, E_graph, startIndex=0):
    for cout in range(num):
            path = "{}{}".format(outfolder,cout + startIndex)
            genOneRealizationTrue(path,E_graph)

def E_move(E_list,path,graph,soup_number):
    print(E_list)
    data_path = os.path.dirname(path) + "/data"
    featurePath = "{}/{}/features/{}_{}/".format(data_path, graph, "EG", soup_number)
    i = 0
    for file_number in E_list:
        outpath = featurePath + str(i)
        with open(outpath, 'w') as outfile:
            infile = open(file_number, 'r')
            while True:
                line = infile.readline()
                if not line:
                    infile.close()
                    break
                ints = line.split()
                node1 = ints[0]
                node2 = ints[1]
                alpha = ints[2]
                beta = ints[3]
                mean = ints[4]
                weight = getWeibull(float(alpha), float(beta))
                outfile.write(node1 + " ")
                outfile.write(node2 + " ")
                outfile.write(str(weight) + "\n")
            outfile.close()
        i = i + 1





