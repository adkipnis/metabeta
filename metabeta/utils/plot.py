from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

cmap = plt.get_cmap('tab20')
PALETTE = [mcolors.to_hex(cmap(i)) for i in range(0, cmap.N, 2)]
PALETTE += [mcolors.to_hex(cmap(i)) for i in range(1, cmap.N, 2)]

