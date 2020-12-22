import matplotlib.pyplot as plt
import numpy as np

def plot_loss(loss1, loss2=None, loss1_label="Loss1", loss2_label="Loss2", save_path=None):
    plt.figure(figsize=(15,7))
    
    plt.plot(loss1, label=loss1_label)
    if loss2 is not None:
        plt.plot(loss2, label=loss2_label)
    
    plt.grid()
    plt.show()
    if save_path is not None:
        plt.savefig(save_path)