import matplotlib.pyplot as plt
import numpy as np

max_flops_unboosted = 12288  # GFLOP/s
max_flops_boosted = 13444.5  # Empirical roofline toolkit

max_g_bandwidth_warburton = 540  # GB/s
max_g_bandwidth_ert = 561.4
max_l1_bandwidth = 2610.5

flops_per_byte_accessed = np.arange(0, 101)
max_flops_unboosted_array = max_flops_unboosted * \
    np.ones_like(flops_per_byte_accessed)

max_flops_g_unboosted_data = np.minimum(flops_per_byte_accessed
    * max_g_bandwidth_warburton, max_flops_unboosted_array)
max_flops_l1_unboosted_data = np.minimum(flops_per_byte_accessed
    * max_l1_bandwidth, max_flops_unboosted_array)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.loglog(flops_per_byte_accessed, max_flops_g_unboosted_data,
    label='Device memory roofline')
ax.loglog(flops_per_byte_accessed, max_flops_l1_unboosted_data,
    label='L1 cache/Local memory roofline')

theoretical_x_1 = 3*2*np.array([10, 20, 35, 56, 85, 120]) \
    / (4 + 12)  # Assumes one read and three stores
theoretical_x_4 = 3*2*np.array([10, 20, 35, 56, 84, 120]) \
    / (4 + 12 + 12)  # Assumes four reads and three stores
theoretical_x_7 = 3*2*np.array([10, 20, 35, 56, 84, 120]) \
    / (4 + 2*(12+12))  # Assumes seven reads and three stores
#theoretical_x = 2*np.arange(1,33) / (4 + 4) # Assumes one read and one stores
theoretical_y_1 = np.minimum(theoretical_x_1
    * max_g_bandwidth_warburton, max_flops_unboosted)
theoretical_y_4 = np.minimum(theoretical_x_4
    * max_g_bandwidth_warburton, max_flops_unboosted)
theoretical_y_7 = np.minimum(theoretical_x_7
    * max_g_bandwidth_warburton, max_flops_unboosted)
empirical_x = theoretical_x_4.copy()
#empirical_x[0:3] = theoretical_x_1[0:3]
empirical_y = [2026.9636053441898, 4049.8734098551745, 7085.0042493541905,
    8143.440577930807, 9010.054141132498, 10126.59788574097]
print(theoretical_x_1)
print(theoretical_y_1)
print(theoretical_x_4)
print(theoretical_y_4)

pn_labels = ['2', '3', '4', '5', '6', '7']

plt.title("Grudge elementwise differentiation kernel: FP32")
ax.loglog(theoretical_x_1, theoretical_y_1, 'sy',
    label='4 device memory accesses model (3 writes, 1 read)', markersize=8)
ax.loglog(theoretical_x_4, theoretical_y_4, 'ob',
    label='7 device memory accesses model, (3 writes, 4 reads)')
#plt.loglog(theoretical_x_7, theoretical_y_7,'oy', label='13 accesses model')
ax.loglog(theoretical_x_1, empirical_y, '.r',
    label='Experimental results assuming 4 accesses')
for i in range(6):
    ax.annotate(pn_labels[i], x=(theoretical_x_1[i], empirical_y[i]))
ax.loglog(theoretical_x_4, empirical_y, '.g',
    label='Experimental results assuming 7 accesses')
for i in range(6):
    ax.annotate(pn_labels[i], xy=(theoretical_x_4[i], empirical_y[i]))
plt.ylabel("GFLOP/s")
plt.xlabel("Bytes per flop")
plt.legend()
#plt.yticks(theoretical_y)
plt.show()
