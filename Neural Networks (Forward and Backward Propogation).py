import math

# sigmoid activation function
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# given inputs
x1 = 0.35
x2 = 0.9

# actual output
y_actual = 0.5


print("----- First Forward Propagation -----")

# initial weights
w13 = 0.1
w23 = 0.8
w14 = 0.4
w24 = 0.6
w35 = 0.3
w45 = 0.9


# hidden layer calculations
a3 = (x1 * w13) + (x2 * w23)
y3 = sigmoid(a3)

a4 = (x1 * w14) + (x2 * w24)
y4 = sigmoid(a4)

# output layer
a5 = (y3 * w35) + (y4 * w45)
y = sigmoid(a5)

print("Output y =", round(y,3))

# old error
old_error = y_actual - y
print("Old Error =", round(old_error,3))


print("\n----- After Backward Propagation (Updated Weights) -----")

# updated weights
w13 = 0.099
w23 = 0.797
w14 = 0.397
w24 = 0.593
w35 = 0.272
w45 = 0.873


print("\n----- Second Forward Propagation -----")

# hidden layer again
a3 = (x1 * w13) + (x2 * w23)
y3 = sigmoid(a3)

a4 = (x1 * w14) + (x2 * w24)
y4 = sigmoid(a4)

# output layer again
a5 = (y3 * w35) + (y4 * w45)
y_new = sigmoid(a5)

print("Output y =", round(y_new,3))

# new error
new_error = y_actual - y_new
print("New Error =", round(new_error,3))


# error difference
error_difference = new_error - old_error

print("\n----- Error Difference -----")
print("Error Difference =", round(error_difference,3))
