from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_index_r2(df_result, y, yerr, xlabel, ylabel, labelfontsize=15,
                tickfontsize=15, legendfontsize=15):
    """plot index vs r2.

    Args:
        df_result (pd.DataFrame): data.
        y (str): y of data.
        yerr (str): yerr of data.
        xlabel (str): string to pass ax.xlabel.
        ylabel (str): string to pass ax.ylbael.
        labelfontsize (int, optional): label font size. Defaults to 15.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        legendfontsize (int, optional): legend font size. Defaults to 15.
    """
    fig, ax = plt.subplots()
    df_result.plot(y=y, yerr=yerr, ax=ax)
    ax.set_xlabel(xlabel, fontsize=labelfontsize)
    ax.set_ylabel(ylabel, fontsize=labelfontsize)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    ax.legend(fontsize=legendfontsize)
    fig.tight_layout()

def plot_importance(df, x, y, sortkey=None, yscale="log",
                    tickfontsize=15, labelfontsize=15, legendfontsize=15):
    """plot importance.

    Args:
        df (pd.DataFrame): データ.
        x (str): x to plot bar.
        y (str): y to plot bar.
        sortkey (str, optional): sortkey. Defaults to None.
        yscale (str, optional): yscale string. Defaults to "log".
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        labelfontsize (int, optional): label font size. Defaults to 15.
        legendfontsize (int, optional): label font size. Defaults to 15.
    """
    fig, ax = plt.subplots()
    if sortkey is None:
        sortkey = y
    _df = df.sort_values(by=sortkey, ascending=False)
    _df.plot.bar(x=x, y=y, ax=ax)
    ax.set_yscale(yscale)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    ax.set_xlabel(ax.get_xlabel(), fontsize=labelfontsize)
    ax.legend(fontsize=legendfontsize)
    fig.tight_layout()
    

def plot_r2_hist(df, xlim=None, bins=100, tickfontsize=15, labelfontsize=15):
    """R2のDOSを図示する．

    Args:
        df (pd.DataFrame): データ
        xlim (tuple(float, float), optional): 図のx range. Defaults to None.
        bins (int, optional): histogramのbin数. Defaults to 100.
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        labelfontsize (int, optional): label font size. Defaults to 15.
    """
    fig, ax = plt.subplots()
    x = df["score_mean"]
    occurrence, edges = np.histogram(x, bins=bins, range=xlim)
    left = (edges[:-1]+edges[1:])*0.5
    ax.bar(left, occurrence, width=left[1]-left[0])
    # df.hist("score_mean", bins=100, xlim=xlim, ax=ax)
    ax.set_xlabel("$R^2$", fontsize=labelfontsize)
    ax.set_ylabel("occurrence", fontsize=labelfontsize)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)
    fig.tight_layout()


def calculate_coeffix(descriptor, combilist, coeflist):
    """表示のために 係数０の部分を加えて係数を作りなおす．

    Args:
        descriptor (list[str]): all the descriptor names
        combilist (list): a list of descriptor combinations of the models
        coeflist (np.array): a list of coefficients of the models

    Returns:
        list: a list of coefficnets whose length is the same as the length of all the descriptors
    """
    n = len(descriptor)
    coeffixlist = []
    for combi, coef in zip(combilist, coeflist):

        coeffix = np.zeros((n))
        # if combi=[1,2], and coef=[val1,val2], then coeffix=[0,val1,val2,0,0]
        for i, id in enumerate(combi):
            coeffix[id] = coef[i]

        # 都合でlistに直す．
        coeffixlist.append(list(coeffix))
    return coeffixlist


def plot_weight_diagram(df_result, descriptor_names, ax=None, nmax=200):
    """weight diagramの表示

    Args:
        df_result (pd.DataFrame): dataimport pandas as pd
        descriptor_names (List[str]): 説明変数カラムリスト
        ax (matplotlib.axes): axes. Defaults to None.
        nmax (int, optional): the maximum number of the data to show. Defaults to 200.

    """
    ax_orig = ax
    x = df_result.loc[:nmax, descriptor_names].values
    x = np.log10(np.abs(x))
    df_x = pd.DataFrame(x, columns=descriptor_names).replace(
        [-np.inf, np.inf], np.nan)
    df_weight_diagram = df_x.fillna(-3)
    if ax_orig is None:
        fig, ax = plt.subplots()
    # ax.set_title("log10(abs(coef))")
    sns.heatmap(df_weight_diagram.T, ax=ax)
    ax.set_ylim((-0.5, df_weight_diagram.shape[1]+0.5))
    if ax_orig is None:
        fig.tight_layout()


def make_counts(df_result, descriptor_names, sentense, ratio=False):
    """
    説明変数が用いられた回数を計算する．

    Args:
        df_result (pd.DataFrame): データ
        descriptor_names (list[str]): 説明変数名リスト．
        sentense (str): query文
        ratio (bool, optional): 回数(False), 割合(True)を返す． Defaults to False.

    Returns:
        pd.DataFrame: 回数もしくは割合データ．
    """
    x = df_result[descriptor_names].values != 0  # 係数が０でない．＝その説明変数が含まれるモデル．
    df_indicator_diagram = df_result.copy()
    df_indicator_diagram.loc[:, descriptor_names] = x

    dfq = df_indicator_diagram.query(sentense)
    print("all=", dfq.shape[0])
    if ratio:
        return np.sum(dfq[descriptor_names], axis=0)/dfq.shape[0]
    else:
        return np.sum(dfq[descriptor_names], axis=0)


def make_and_plot_block_weight_list(df_result, descriptor_names, querylist):
    """
    querylistのblock weight diagramを計算する．

    Args:
        df_result (pd.DataFrame): データ．
        descriptor_names (list[str]): 説明名リスト．
        querylist (list[str]): query文リスト．
    Returns:
        pd.DataFrame: block weight diagram.
    """
    result = []
    for sentense in querylist:
        # 前の図に合わせるためにdescriptor_namesの順序を逆にする．
        t = make_counts(
            df_result, descriptor_names[::-1], sentense, ratio=True)
        result.append(t)
    dfq = pd.DataFrame(result, index=querylist)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(dfq.T, ax=ax)  # 前の図に合わせるためにtransposeする．
    ax.set_xticklabels(ax.get_xticklabels(), ha='right', rotation=45)
    fig.tight_layout()

    return dfq


def make_indicator_diagram(df_result, descriptor_names, nmax=50):
    """indicator diagramを作成する．

    Args:
        df_result (pd.DataFrame): data
        nmax (int, optional): the maximum number of the data to show. Defaults to 50.
    """
    x = df_result[descriptor_names].values != 0
    df_indicator_diagram = pd.DataFrame(x, columns=descriptor_names)
    return df_indicator_diagram


def make_df_by_index(df_indicator_diagram, descriptor_names, index):
    """部分データを得るためのデータを作る．

    return valueはindexで指定された範囲で，
    descriptor_names（０でない数）+"N"（データインスタンス総数）をカラムに持つデータになる．

    Args:
        df_indicator_diagram (pd.DataFrame): indicator diagram データ
        descriptor_names (list[str]): 説明変数名リスト
        index (list[int]): データインスタンスの部分indexリスト

    Returns:
        pd.DataFrame: indicator diagram部分データ
    """
    dfq = df_indicator_diagram.iloc[index, :]
    print("all=", dfq[descriptor_names].shape[0])
    df_all = pd.DataFrame({"N": [dfq[descriptor_names].shape[0]]},)
    dfq_sum = dfq[descriptor_names].astype(int).sum(axis=0)
    df1 = pd.DataFrame(dfq_sum).T

    return pd.concat([df1, df_all], axis=1)
    # print(np.sum(dfq[descriptor_names], axis=0))


def make_all_ind_by_index(df_indicator_diagram, descriptor_names, regionindex, regionsize):
    """各領域の非ゼロの説明変数の割合を得る．

    regionindex=[0,1,..,N]
    for i in regionindex:
        region = [ i*regionsize, (i+1)*regionsize ]
    と各data instance index領域を定義する．

    Args:
        df_indicator_diagram (pd.DataFrame): データ
        descriptor_names (list[str]): 説明変数名リスト
        regionindex (list[int])): 領域インデックスリスト
        regionsize (int)): 領域サイズ

    Returns:
        pd.DataFrame: 分割領域ごとのデータ
    """
    df_ind_list = []
    for i in regionindex:
        region = list(range(i*regionsize, (i+1)*regionsize))
        df_ind = make_df_by_index(
            df_indicator_diagram, descriptor_names, region)
        df_ind_list.append(df_ind)
    _df = pd.concat(df_ind_list, axis=0).reset_index(drop=True)

    names = list(_df.columns)
    names.remove("N")
    v0 = _df["N"]
    for name in names:
        _df[name] = _df[name]/v0

    if False:
        fig, ax = plt.subplots()
        _df[names].T.plot(ax=ax)
        ax.set_ylabel("frequency")
        ax.set_xticks(list(range(len(names))))
        ax.set_xticklabels(names, rotation=90)
        ax.set_ylim((0, 1))
    return _df

from matplotlib.ticker import MaxNLocator

def plot_df_imp_by_index(df_imp_by_index, descriptor_names, regions, regionsize,
                        tickfontsize=15, legendfontsize=12, labelfontsize=15):
    """各領域の説明変数の頻度を図示する．

    Args:
        df_imp_by_index (list[pd.DataFrame]): _description_
        descriptor_names (list[str]): 説明変数名リスト
        regions (list[int])): 領域リスト
        regionsize (int): 領域サイズ
        comment (str): 表示用コメント. Defaults to "importancebyindex".
        metadata (dict): 表示用データ. Defaults to G_METADATA. 
        tickfontsize (int, optional): ticks font size. Defaults to 15.
        legendfontsize (int, optional): ticks font size. Defaults to 15.
        labelfontsize (int, optional): ticks font size. Defaults to 15.
    """
    xticks_str = []
    for i in regions:
        xticks_str.append("[{}:{})".format(i*regionsize, (i+1)*regionsize))
    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if False:
        df_imp_by_index[descriptor_names].plot(marker="o", ax=ax)
    else:
        marker_list = [".", "o", "v", "^", "<", ">"]
        marker_list += ["8", "s", "p", "*", "h", "H", "+", "x", "D", "d"]
        for exp_name, marker in zip(descriptor_names, marker_list):
            df_imp_by_index[exp_name].plot(marker=marker, ax=ax)
    ax.set_xticks(list(range(len(regions))))
    ax.set_xticklabels(xticks_str, rotation=90)
    ax.tick_params(axis='x', labelsize=tickfontsize)
    ax.tick_params(axis='y', labelsize=tickfontsize)    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=legendfontsize)
    ax.set_ylabel("occurrence", fontsize=labelfontsize)
    fig.tight_layout()
