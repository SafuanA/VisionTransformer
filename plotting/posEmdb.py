import numpy as np
import matplotlib.pyplot as plt


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P


P = getPositionEncoding(seq_len=100, d=512, n=10000)
print(P)


def plotSinusoid(k, d=512, n=10000):
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2 * x / d)
    y = np.sin(k / denominator)
    plt.plot(x, y)


def plotCos(k, d=512, n=10000):
    x = np.arange(0, 100, 1)
    denominator = np.power(n, 2 * x / d)
    y = np.cos(k / denominator)
    plt.plot(x, y)


fig = plt.figure()
val = []
plt.subplot(211)
plt.title('Positional Embeddings Sin')
for i in range(4):
    plotSinusoid(i * 4)
    val.append('pos = ' + str(i * 4))
plt.legend(val, loc='lower right')

plt.subplot(212)
val = []
plt.title('Positional Embeddings Cos')
for i in range(4):
    plotCos((i * 4) + 1)
    val.append('pos = ' + str((i * 4) + 1))
plt.legend(val, loc='lower right')
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.99,
                    top=0.9,
                    wspace=0.5,
                    hspace=0.5)

P = getPositionEncoding(seq_len=100, d=512, n=10000)
cax = plt.matshow(P)
plt.title("Visualisierung Positional Embedding")
plt.xlabel("d_model")
plt.ylabel("Sequenz")
plt.gcf().colorbar(cax)
plt.show()
