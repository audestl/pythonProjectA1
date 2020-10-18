
import numpy as np
import matplotlib.pyplot as plt
dataset = np.loadtxt("train_1.csv", delimiter=",")


# PLOTTING
character = []
characterClass = [0]*26
value = [0]*26
for cell in range(len(value)):
    value[cell] = cell


for cell in range(len(dataset)):
    character.append(dataset[cell][-1])

for cell in range(len(character)):
    num = int(character[cell])
    characterClass[num] = characterClass[num] + 1.0


plt.bar(value, characterClass)
plt.ylabel('Number of instances')
plt.xlabel('Letters index')
plt.show()
