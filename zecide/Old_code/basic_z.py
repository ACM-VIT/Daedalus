import numpy as np
import pandas as pd

def bayes(x, y, z):

    return((x*y) / ((x*y) + z*(1-x)))


if __name__ == "__main__":

    df = pd.read_csv('Z NIFTY.csv')

    val = df.iloc[0].copy()
    print(val)

    for i in range(1, 7):

        print(val['x'])
        val['x'] = bayes(val['x'], val['y' + str(i)], val['z' + str(i)])

    print(val['x'])
