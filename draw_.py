import pickle
import matplotlib.pyplot as plt
import numpy as np


epochs_l = np.arange(1, 101)

with open("res_exp3.pt", "rb") as f:
    data = pickle.load(f)
    loss_list = data["loss"]
    train_acc_list = data["train_acc"]
    val_acc_list = data["val_acc"]
    test_acc_list = data["test_acc"]

best_val_idx = np.argmax(val_acc_list)
plt.figure(figsize=(5, 5))
plt.plot(epochs_l, train_acc_list, label='Train Acc', color='#B2912F')
plt.plot(epochs_l, val_acc_list, label='Val Accu', color='#FAA43A')
plt.plot(epochs_l, test_acc_list, label='Test Accu', color='#60BD68')
plt.text(epochs_l[best_val_idx], val_acc_list[best_val_idx] + 0.005,
        f'val:{val_acc_list[best_val_idx]:.4f}, test:{test_acc_list[best_val_idx]:.4f}',
        ha='center', va='bottom', fontsize=9, color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Acc vs Epochs for MultiTableBRIDGE')
plt.legend()
# plt.show()
plt.savefig('./mbridge_acc_plot.png')


#############################################################
with open("res_o.pt", "rb") as f:
    data = pickle.load(f)
    loss_list2 = data["loss"]
    train_acc_list2 = data["train_acc"]
    val_acc_list2 = data["val_acc"]
    test_acc_list2 = data["test_acc"]

best_val_idx2 = np.argmax(val_acc_list2)
plt.figure(figsize=(5, 5))
plt.plot(epochs_l, train_acc_list2, label='Train Acc', color='#B2912F')
plt.plot(epochs_l, val_acc_list2, label='Val Accu', color='#FAA43A')
plt.plot(epochs_l, test_acc_list2, label='Test Accu', color='#60BD68')
plt.text(epochs_l[best_val_idx2], val_acc_list2[best_val_idx2] + 0.005,
        f'val:{val_acc_list2[best_val_idx2]:.4f}, test:{test_acc_list2[best_val_idx2]:.4f}',
        ha='center', va='bottom', fontsize=9, color='red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Acc vs Epochs for BRIDGE')
plt.legend()
# plt.show()
plt.savefig('./o_acc_plot.png')


#############################################################
plt.figure(figsize=(5, 5))
plt.plot(epochs_l, loss_list, label='MultiTableBRIDGE', color='#8c564b')
plt.plot(epochs_l, loss_list2, label='BRIDGE Loss', color='#17becf')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.savefig('./loss_plot.png')
# plt.show()

val_acc_list = np.array(val_acc_list, dtype=np.float32)