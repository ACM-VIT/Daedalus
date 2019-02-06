import numpy as np
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('Z NIFTY.csv')
    # print(df)

    y = df[['y1', 'y2', 'y3', 'y4', 'y5', 'y6']].copy()


    w1 = np.array([0.2, 0.2, 0.1, 0.1, 0.1, 0.3])
    print("Sum of weights: %s" % str(w1.sum()))

   
    weights = y.mul(w1, axis=1)
    print("Weight multiplied matrix: ")
    print(weights)

    wavg = weights.sum(axis=1) * w1.sum()
    print("Weighted average: ")
    print(wavg)
