import numpy as np
import matplotlib.pyplot as plt
from uncertainty_quantification.visualization_utils import DEFAULT_FIG_SIZE, DEFAULT_FONT_SIZE, DEFAULT_LINE_WIDTH
import numpy as np

# ---------------------------------------------------------------------------
# DEFAULT_FONT_SIZE, DEFAULT_FIG_SIZE = 12, (5, 3.5)
k          = np.array([1, 3, 8, 16])
maj_0      = np.array([0.6477647058823529, 0.6669411764705883, 0.6941176470588235, 0.7078823529411765]) * 100
maj_25     = np.array([0.5981196581196582, 0.6267521367521367, 0.66, 0.6687179487179488]) * 100
maj_200    = np.array([0.3531764705882353, 0.37411764705882355, 0.42458823529411766, 0.44435294117647056]) * 100
alpha = 0.6
markersize = 60
FONT_SIZE = DEFAULT_FONT_SIZE + 30
plt.rc('font', size=FONT_SIZE)

# === 1) Funnel ==============================================================
# positions  = np.linspace(1, 200, 200)
# p          = np.log(1.76 / 1.23) / np.log(25)
# half_width = 1 / (positions ** p)
#
# fig, ax = plt.subplots(figsize=(6, 3))
# ax.plot(positions,  half_width,  c="black")
# ax.plot(positions, -half_width,  c="black")
# ax.set(xlabel="position", xlim=(0, 210),
#        ylim=(-1.1, 1.3), xticks=[1, 25, 200])
# ax.yaxis.set_visible(False)
# for side in ("left", "right", "top"):
#     ax.spines[side].set_visible(False)
# ax.spines["bottom"].set_position("zero")
# fig.savefig("funnel.pdf", bbox_inches="tight")
# plt.close(fig)

# === 2) Maj@K – two curves, arrow marking the drop =========================
fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
ax.plot(k, maj_0 , "o-",  lw=DEFAULT_LINE_WIDTH, label="Normal", color="black", markersize=markersize)
ax.plot(k, maj_25, "d-", lw=DEFAULT_LINE_WIDTH, label="After 25th Output Token", color="blue", markersize=markersize)
# ax.annotate("", xy=(k[-1], maj_25[-1]), xytext=(k[-2], maj_25[-2]),
#             arrowprops=dict(arrowstyle="->", lw=1.8))
for xi, y_orig, y_25 in zip(k, maj_0, maj_25):
    ax.annotate("",
                xy=(xi, y_25 + 0.3),      # arrow head (lower point)
                xytext=(xi, y_orig - 0.3),# arrow tail (upper point)
                arrowprops=dict(arrowstyle="->", lw=DEFAULT_LINE_WIDTH * 2, color="blue", alpha=alpha + 0.2))
ax.set_xticks(k)
ax.set_xlabel("K")
ax.set_ylabel("Maj@K")
# ax.legend(frameon=False)
# plt.subplots_adjust(bottom=0.005)
fig.legend(bbox_to_anchor=(0.5, 0.1), loc='upper center', ncol=1)
# fig.legend(loc='outside lower center', ncol=1)
fig.tight_layout()
fig.savefig("maj_two_curves.pdf", bbox_inches="tight")
# fig.savefig("maj_two_curves.pdf", bbox_inches="tight")
plt.close(fig)

# === 3) Maj@K – three curves ===============================================
fig, ax = plt.subplots(figsize=DEFAULT_FIG_SIZE)
ax.plot(k, maj_0  ,"o-", lw=DEFAULT_LINE_WIDTH, label="Normal", color="black", markersize=markersize)
ax.plot(k, maj_25 ,"d-", lw=DEFAULT_LINE_WIDTH, label="After 25th Output Token", color="blue", markersize=markersize)
ax.plot(k, maj_200,"*-", lw=DEFAULT_LINE_WIDTH, label="After 200th Output Token", color='green', markersize=markersize)
# for xi, y_orig, y_25 in zip(k, maj_0, maj_25):
#     ax.annotate("",
#                 xy=(xi, y_25),      # arrow head (lower point)
#                 xytext=(xi, y_orig),# arrow tail (upper point)
#                 arrowprops=dict(arrowstyle="->", lw=3, color='blue', alpha=0.3))
for xi, y_25, y_200 in zip(k, maj_25, maj_200):
    ax.annotate("",
                xy=(xi, y_200 + 1),      # arrow head (lower point)
                xytext=(xi, y_25 - 1),# arrow tail (upper point)
                arrowprops=dict(arrowstyle="->", lw=DEFAULT_LINE_WIDTH * 2, color='green', alpha=alpha + 0.2))
ax.set_xticks(k)
ax.set_xlabel("K")
ax.set_ylabel("Maj@K")
# ax.legend(frameon=False)
# plt.subplots_adjust(bottom=0.005)
fig.legend(bbox_to_anchor=(0.5, 0.1), loc='upper center', ncol=1)
# fig.legend(loc='outside lower center', ncol=1)
# fig.legend()
fig.tight_layout()
fig.savefig("maj_three_curves.pdf", bbox_inches="tight")
# fig.savefig("maj_three_curves.pdf", bbox_inches="tight")
plt.close(fig)

print("Done.  You’ll find funnel.pdf, maj_two_curves.pdf and maj_three_curves.pdf in the working directory.")
