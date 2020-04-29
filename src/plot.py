import json
import numpy as np
import matplotlib.pyplot as plt

with open("./ADL-hw2/predict.json") as f:
    f = json.load(f)
a_list = [0]*33
for i in f.values():
    a_list[len(i)]+=1

virgin = True
for i in range(len(a_list)):
    if virgin:
        virgin = False
        continue
    else:
        a_list[i]+=a_list[i-1]
x = []
for i in range(33):
    a_list[i]=(a_list[i]/a_list[32])*100
    x.append(i)
plt.plot(x,a_list)
plt.xlabel("Length")
plt.ylabel("Count %")
plt.show()