import debugpy

debugpy.configure(qt="none")

import matplotlib
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

plt.ion()

fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5])

plt.show(block=False)

print("plot created")