import csv 
import matplotlib.pyplot as plt

rewards = []
iters = []
with open("graphs/single_pred_perf.csv", 'r') as csvfile:
    r = csv.reader(csvfile)
    for x in r:
        rewards.append(float(x[0]))
        iters.append(float(x[1]))

plt.plot(range(len(rewards)), rewards)
# plt.show()
#plt.plot(range(len(iters)), iters)
plt.xlabel("Iteration 100x")
plt.ylabel("Average reward")
plt.show()