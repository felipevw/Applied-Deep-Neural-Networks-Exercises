"""
# Single neuron in tensorflow, Part 1: Performance
Created on Mon Sep 16 18:26:24 2019

@author: felip
"""
# Testing single neuron architecture in tensorflow
import timeit

# Performance of loops and lists vs numpy

print("Test 1: normal sample multiplication")
print(timeit.timeit("""
import random 
import numpy as np
lst1 = random.sample(range(1, 10**3), 10**2)
lst2 = random.sample(range(1, 10**3), 10**2)
for i in range(len(lst1)):
    ab = lst1[i] * lst2[i]
""",
number = 100))

print("Test 2: numpy sample multiplication")
print(timeit.timeit("""
import random 
import numpy as np
lst1 = random.sample(range(1, 10**3), 10**2)
lst2 = random.sample(range(1, 10**3), 10**2)
list1_np = np.array(lst1)
list2_np = np.array(lst2)
out2 = np.multiply(list1_np, list2_np)
""",
number = 100))

print("Test 3: multiple relu function performance")
print("Method 1")
print(timeit.timeit("""
import random 
import numpy as np
x = np.random.random(10**4)
def relu1(x):
    return np.maximum(x, 0, x)
""",
number = 100))

print("Method 2")
print(timeit.timeit("""
import random 
import numpy as np
x = np.random.random(10**4)
np.maximum(x, 0)
""",
number = 100))

print("Method 3")
print(timeit.timeit("""
import random 
import numpy as np
x = np.random.random(10**4)
x * (x > 0)
""",
number = 100))

print("Method 4")
print(timeit.timeit("""
import random 
import numpy as np
x = np.random.random(10**4)
(abs(x) + x) / 2
""",
number = 100))

