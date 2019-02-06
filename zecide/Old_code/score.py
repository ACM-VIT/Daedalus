import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#  Test case 1
# x% change that is increases due to factor
# Actually increases
# Expected output
# Score increases at the same rate till max


# Test case 2
# x% change that it increases due to factor
# Actually deceases
# Expected output
# Score decreases at the same rate till min


# Test case 3
# x% chance that it increases or descreases due to factor
# Expected output
# Score decreases at the same rate till extreme values

# Factor results to 40%
increase_fact = 0.4
decrease_fact = 0.4

# Actual change (increase = 1, Decrease = -1)
actual = 1

# User score (default value 0.5 for new users)
score = 0.5

# Growth factor defines how much each prediction reflects in the actual score
growth_factor = 0.1

history = [0.5]

def Calculate_score(prob, inc_dec):

    global score
    global history

    mul = None
    if inc_dec == actual:
        mul = 1
    else:
        mul = -1

    score += prob * increase_fact * mul * growth_factor

    if score>1:
        score = 1
    elif score<0:
        score = 0

    history.append(score)

if __name__ == "__main__":

    num = 20
    val = np.random.random_sample(num)
    inc_dec = np.array(np.random.choice([1, -1], size=num))
    print(val)

    for i in range(num):
        Calculate_score(val[i], inc_dec[i])

    plt.plot([i for i in range(num+1)], history, marker='o', color='pink')
    plt.plot([i for i in range(num+1)], \
        [np.mean(history) for i in range(num+1)], color='grey', linestyle='-')
    plt.legend(['Score', 'Mean'], loc='best')
    plt.show()
