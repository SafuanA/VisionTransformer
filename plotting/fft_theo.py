import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, 3 * np.pi)
sin = 6 * np.sin(2 * np.pi * x * 4)
cos = 5 * np.cos(2 * np.pi * x * 6)
cos_h = 7 * np.cos(2 * np.pi * x * 5)

fig = plt.figure(1)
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.99,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)
plt.subplot(211)
plt.title('Frequenzen')
plt.plot(x, sin)
plt.plot(x, cos)
plt.plot(x, cos_h)
plt.legend(['a(t)', 'b(t)', ' c(t)'])
plt.xlabel('Zeit')
plt.ylabel('Frequenz')

plt.subplot(212)
plt.title('Frequenzkomposition')
plt.plot(x, sin + cos_h + cos)
plt.legend(['a(t) + b(t) + c(t)'])
plt.xlabel('Zeit')
plt.ylabel('Frequenz')
# plt.ylim(-3, 3)
plt.show()
