import numpy as np
import matplotlib.pyplot as plt
from random import randint

num = 10
prob = np.array([randint(0, 100) for i in range(num)])

if __name__ == "__main__":

    mean = prob.mean()
    median = np.median(prob)

    print("Min: ", prob.min())
    print("Max: ", prob.max())
    print("Mean: ", mean)
    print("Median: ", median)
    print("Standard deviation: ", prob.std())

    plt.boxplot(prob, showmeans=True)
    plt.show()

    # Calculating weights depending on deviation from Mean

    deviation_mean = np.abs(prob - mean)
    # plt.boxplot(deviation_mean, showmeans=True)
    # plt.show()
    percentage_deviation_mean = deviation_mean/mean
    print("Deviation: ", deviation_mean)
    print("Percentage deviation from Mean", percentage_deviation_mean)
    print("Max: ", percentage_deviation_mean.max())
    print("Min: ", percentage_deviation_mean.min())
    print("Mean: ", percentage_deviation_mean.mean())
    weights_mean = 1 - percentage_deviation_mean
    print("Weights: ", weights_mean)

    # Calculating weights depending on deviation from Median

    deviation_median = np.abs(prob - median)
    # plt.boxplot(deviation_median, showmeans=True)
    # plt.show()
    # percentage_deviation_median = deviation_median/median
    # print("Percentage deviation from Median")
    # print("Max: ", percentage_deviation_median.max())
    # print("Min: ", percentage_deviation_median.min())
    # print("Mean: ", percentage_deviation_median.mean())
    # weights_median = 1 - percentage_deviation_median
    # print("Weights: ", weights_median)
