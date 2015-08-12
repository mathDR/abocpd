import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from getMedianRunLength import getMedianRunLength


def convert2AlertSingle(Rs, last_alarm, tshold):
    if last_alarm > Rs.shape[0]:
        last_alarm = Rs.shape[0]

    changePointProb = Rs[:last_alarm + 1].sum()
    if changePointProb >= tshold:
        alert = True
        last_alarm = 0
    else:
        alert = False
        last_alarm += 1
    return alert, last_alarm


def convert2Alert(Rs, alertTsh):
    max_run, T = Rs.shape
    last_alarm = np.inf
    alert = [False] * T
    for i in range(T):
        alert[i], last_alarm = convert2AlertSingle(
            Rs[:, i], last_alarm, alertTsh)
    return alert


def plotS(S, X, timeindex, changePoints=None):

    alertThreshold = 0.95
    alert = convert2Alert(S, alertThreshold)
    alertInd = [i for i, j in enumerate(alert) if j]

    fig = plt.figure()

    _ = plt.subplot(2, 1, 1)
    _ = plt.plot(timeindex, X)
    _ = plt.plot(timeindex[alertInd], np.mean(X) * np.ones_like(
        timeindex[alertInd]), 'rx', markersize=12, mew=3)
    if changePoints:
        _ = plt.plot(timeindex[changePoints], np.mean(X) * np.ones_like(
            timeindex[changePoints]), 'kx', markersize=12, mew=3)
    _ = plt.xlim([timeindex[0], timeindex[-1]])
    _ = plt.ylim([X.min(), X.max()])

    _ = plt.grid()

    _ = plt.subplot(2, 1, 2)
    _ = plt.imshow(
        np.cumsum(S, axis=0), extent=[timeindex[0], timeindex[-1], 0,
                                      np.asarray(range(S.shape[0]))[-1]], aspect='auto', cmap=cm.Greys_r, origin='lower')
    Mrun, tmp = getMedianRunLength(S)
    _ = plt.plot(timeindex, Mrun, 'r')
    _ = plt.xlim([timeindex[0], timeindex[-1]])
    _ = plt.ylim([Mrun.min(), Mrun.max()])
    _ = plt.ylabel('Median run length')
    return fig
