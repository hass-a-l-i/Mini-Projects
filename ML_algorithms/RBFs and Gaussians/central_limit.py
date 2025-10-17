import matplotlib.pyplot as plt
from common_funcs import gaussian_overlay_uniform, gaussian_overlay_log

N = int(1e5)  # number of runs - need high to get decent gaussian shape
K = 100  # number of samples per run
bin = 100

#gaussian_overlay_uniform(N, K, bin)
gaussian_overlay_log(N, K, bin)  # as K increases, better fit to gaussian - otherwise skewed because of log
plt.show()


