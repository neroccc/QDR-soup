import math
import random
import sys



def two_exp(log_a,log_b):

# Calculate the maximum logarithm value
    max_log = max(log_a, log_b)

# Calculate the difference in logarithms
    diff_log = log_a - max_log - math.log1p(math.exp(log_b - max_log))

# Calculate the result using the approximation
    result = math.exp(diff_log)

    print(result)

def ten_exp():
    log_values = []
    pair = []
    for i in range(10):
        v = random.uniform(-102,-100)
        log_values.append(v)
        pair.append(i)
    print(log_values)
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
        sam.append(math.ceil(result*10000))
    difference = sum(sam) - 10000
    random.shuffle(pair)
    print(sam)
    for i in range(difference):
        num = pair[i]
        sam[num] = sam[num]-1
    print(sam)
    print(sum(sam))

def getWeibull(alpha, beta):
    time = alpha * math.pow(-math.log(1 - random.uniform(0, 1)), beta)
    if time >= 0:
        return math.ceil(time) + 1
    else:
        raise ValueError("time < 0")

def getWeibull_log_likelihood(alpha, beta, c):
    log_likelihood = math.log(beta) - beta * math.log(alpha) + (beta - 1) * math.log(c) - (c / alpha) ** beta
    return log_likelihood

def file_compare():
    file1 = open("Egraph1","r")
    file2 = open("Egraph2","r")
    lines1 = file1.readlines()
    lines2 = file2.readlines()
    p1 = 0
    p2 = 0
    for i in range(len(lines1)):
        string1 = lines1[i].split(" ")
        string2 = lines2[i].split(" ")
        alpha = int(string1[2])
        beta = int(string1[3])
        center = float(string2[4])
        p1 = p1 + getWeibull_log_likelihood(alpha,beta,center)
        alpha = int(string2[2])
        beta = int(string2[3])
        center = float(string1[4])
        p2 = p2 + getWeibull_log_likelihood(alpha,beta,center)
    res = two_exp(p1,p2)
    print(p1)
    print(p2)
file_compare()