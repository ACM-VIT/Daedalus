import matplotlib.pyplot as plt

weights = [0.9, 0.54, 0.87, 0.38, 0.49, 0.68, 0.73, 0.79, 0.62, 0.51]
sm = 0.0
sums = []

def iter():

    global weights
    global sm
    sm = sum(weights)
    avg = sm/10
    weights = weights[1:9]
    weights.append(avg)

if __name__ == "__main__":


# Test case 1:
#   10 iterations
# Test case 2:
#   20 iterations
# Test case 3:
#   100 iterations
# Test case 4:
#   1000 iterations

    for i in range(10):
        iter()
        print("Weights: ", weights)
        print("Sum of weights: ", sm)
        sums.append(sm)

    print("Sums: ", sums)

    plt.scatter([i for i in range(len(sums))], sums, color='b', alpha=0.50)
    plt.show()


# Method needs correction as the weights keep dropping
# For a small sample eg. 10 iterations, the method feels useful as
# randomly generated weights get eliminated. After 10 iterations, as
# only the newew weights are left (20 iterations) the weights drop
# steadily. Over a large number of iterations (1000 iterations), the
# weights approach 0 and the method does not value the user over time.

# Required changes
# Maybe use a different way to calculate the final weights, after the
# predication and update the list using that weight. This give the user
# a chance to improve his score over time if he makes the correct predictions.

# It's better to use the user rating as weights as this will require only
# one algorithm to calculate the user score and the weights for the user
# input.
