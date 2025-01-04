import math

p = 0.99
e = 0.2
n = 8

k = math.log(1 - p) / math.log(1 - (1 - e) ** n)
print("Estimated number of iterations (k):", k) 