import csv
import matplotlib.pyplot as plt
import numpy as np


scores = np.genfromtxt('learn_data/20230226_scores.csv', delimiter=',')
losses = np.genfromtxt('learn_data/20230226_losses.csv', delimiter=',')

plt.plot(scores, 'b')
plt.plot(1.0 * np.ones_like(scores), 'gray') #Score 상한선
plt.plot(-1.0 * np.ones_like(scores), 'gray') #Score 하한선
plt.show()

plt.plot(losses, 'orange')
plt.plot(0.04 * np.ones_like(losses), 'gray') #Loss 상한선
plt.plot(np.zeros_like(losses), 'gray') #Loss 하한선
plt.show()