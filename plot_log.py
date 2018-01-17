import matplotlib.pyplot as plt
import csv, os

MODEL_NAME = 'noweight'
LOG_FILE = './models/noweight/log.txt'
acc = []
val_acc = []
loss = []
val_loss = []
with open(LOG_FILE, 'r') as f:
    re = csv.reader(f)
    for r in re:
        loss.append(r[1])
        acc.append(r[2])
        val_loss.append(r[3])
        val_acc.append(r[4])

# epochs = len(acc)
# plt.plot(range(epochs), acc, label="acc")
# plt.plot(range(epochs), val_acc, label="val_acc")
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join("img", MODEL_NAME+ "_log.png"), bbox_inches='tight', dpi=160)

epochs = len(acc)
plt.plot(range(epochs), loss, label="loss")
plt.plot(range(epochs), val_loss, label="val_loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig(os.path.join("img", MODEL_NAME+ "_log_loss.png"), bbox_inches='tight', dpi=160)
