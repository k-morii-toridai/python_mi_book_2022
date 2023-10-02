import matplotlib.pyplot as plt
import os


def plot_GPR(X, y, Xtrain, ytrain, yp_mean, yp_std, acq, it, ia=None,
             comment: str = None, metadata=None,
             tickfontsize=20, labelfontsize=20, legendfontsize=20, titlefontsize=25):
    """plot y.mean += y.std and aquisition functions

    Args:
        X (np.array): descriptor
        y (np.array): target values
        Xtrain (np.array): training descriptor data
        ytrain (np.array): training target values
        yp_mean (np.array): the mean values of predictions
        yp_std (np.array): the stddev vlaues of predictions
        acq (np.array): aquisition function values
        it (int): # of iterations
        ia (np.array, optional): a list of actions. Defaults to None.
        comment (str, optional): コメント. Defauls to None.
        metadata (str, optional): 表示用データ. Defaults to None.
        tickfontsize (int. optional): ticks font size. Defaults to 20.
        labelfontsize (int, optional): label font size. Defaults to 20.
        legendfontsize (int, optional): legend font size. Defauls to 20.
        titlefontsize (int, optional): title font size. Defauls to 25.
    """
    yminus = yp_mean - yp_std
    yplus = yp_mean + yp_std
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(X[:, 0], y, "--", color="blue", label="expriment")
    ax.plot(X[:, 0], acq, lw=5 , color="green", label="aquisition function")   
    ax.plot(Xtrain[:, 0], ytrain, "o", markersize=10, color="blue", label="train")
    # alphaで線を半透明にする
    ax.fill_between(X[:, 0], yminus, yplus, color="red", alpha=0.1)
    ax.plot(X[:, 0], yp_mean.reshape(-1), 
            color="red", label="predict$\pm\sigma$")

    ax.set_title(f"iteration {it}", fontsize=titlefontsize)
    if ia is not None:
        ax.axvline(X[ia, 0], color="green", linestyle="--")
        # plt.plot(X[ia,0],yp_mean.reshape(-1)[ia],"o",color="green",label="selected action")
    # ax.legend(fontsize=legendfontsize)
    ax.tick_params(axis = 'x', labelsize =tickfontsize)
    ax.tick_params(axis = 'y', labelsize =tickfontsize)    
    fig.tight_layout()
    if metadata is not None:
        filename = "_".join([metadata["prefix"], metadata["dataname"],
                             "rand", str(metadata["random_state"]),
                             str(comment), str(it)]) + ".pdf"
        print(filename)
        fig.savefig(os.path.join(metadata["outputdir"], filename))
