import math
import random
import sys
def getWeibull(alpha, beta):
    time = alpha * math.pow(-math.log(1 - random.uniform(0, 1)), beta)
    if time >= 0:
        return math.ceil(time) + 1
    else:
        sys.exit("time < 0")
        return None

for i in range(10000):
    print(i)
    infile = "DC_exp_stoGraph"
    outfile = "WeibullRandom_10000/" + str(i)
    with open(infile, "r") as file1:
        lines = file1.readlines()
        with open(outfile, "w") as file2:
            for line in lines:
                line_str = line.split(" ")
                a = line_str[0]
                b = line_str[1]
                alpha = random.randint(1,10)
                beta = random.randint(1,10)
                value = getWeibull(alpha,beta)
                file2.write(a + " "+ b +" " +str(value)+"\n")
    file2.close()



