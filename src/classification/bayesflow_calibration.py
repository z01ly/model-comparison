# lib: bayesflow
# https://github.com/stefanradev93/BayesFlow

from sklearn.calibration import calibration_curve
import numpy as np
import matplotlib.pyplot as plt


def expected_calibration_error(m_true, m_pred, num_bins=10):
    """Estimates the expected calibration error (ECE) of a model comparison network according to [1].

    [1] Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).
        Obtaining well calibrated probabilities using bayesian binning.
        In Proceedings of the AAAI conference on artificial intelligence (Vol. 29, No. 1).

    Notes
    -----
    Make sure that ``m_true`` are **one-hot encoded** classes!

    Parameters
    ----------
    m_true      : np.ndarray of shape (num_sim, num_models)
        The one-hot-encoded true model indices.
    m_pred      : tf.tensor of shape (num_sim, num_models)
        The predicted posterior model probabilities.
    num_bins    : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).

    Returns
    -------
    cal_errs    : list of length (num_models)
        The ECEs for each model.
    probs       : list of length (num_models)
        The bin information for constructing the calibration curves.
        Each list contains two arrays of length (num_bins) with the predicted and true probabilities for each bin.
    """

    # Convert tf.Tensors to numpy, if passed
    if type(m_true) is not np.ndarray:
        m_true = m_true.numpy()
    if type(m_pred) is not np.ndarray:
        m_pred = m_pred.numpy()

    # Extract number of models and prepare containers
    n_models = m_true.shape[1]
    cal_errs = []
    probs_true = []
    probs_pred = []

    # Loop for each model and compute calibration errs per bin
    for k in range(n_models):
        y_true = (m_true.argmax(axis=1) == k).astype(np.float32)
        y_prob = m_pred[:, k]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=num_bins)

        # Compute ECE by weighting bin errors by bin size
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob)
        bin_total = np.bincount(binids, minlength=len(bins))
        nonzero = bin_total != 0
        cal_err = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

        cal_errs.append(cal_err)
        probs_true.append(prob_true)
        probs_pred.append(prob_pred)
    return cal_errs, probs_true, probs_pred


def plot_calibration_curves(
    true_models,
    pred_models,
    model_names=None,
    num_bins=10,
    label_fontsize=16,
    legend_fontsize=14,
    title_fontsize=18,
    tick_fontsize=12,
    epsilon=0.02,
    fig_size=None,
    color="#8f2727",
):
    """Plots the calibration curves, the ECEs and the marginal histograms of predicted posterior model probabilities
    for a model comparison problem. The marginal histograms inform about the fraction of predictions in each bin.
    Depends on the ``expected_calibration_error`` function for computing the ECE.

    Parameters
    ----------
    true_models       : np.ndarray of shape (num_data_sets, num_models)
        The one-hot-encoded true model indices per data set.
    pred_models       : np.ndarray of shape (num_data_sets, num_models)
        The predicted posterior model probabilities (PMPs) per data set.
    model_names       : list or None, optional, default: None
        The model names for nice plot titles. Inferred if None.
    num_bins          : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text (ECE value)
    title_fontsize    : int, optional, default: 16
        The font size of the title text. Only relevant if `stacked=False`
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    epsilon           : float, optional, default: 0.02
        A small amount to pad the [0, 1]-bounded axes from both side.
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor. Inferred if ``None``
    color             : str, optional, default: '#8f2727'
        The color of the calibration curves

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving
    """

    num_models = true_models.shape[-1]
    if model_names is None:
        model_names = [rf"$M_{{{m}}}$" for m in range(1, num_models + 1)]

    # Determine n_subplots dynamically
    n_row = int(np.ceil(num_models / 6))
    n_col = int(np.ceil(num_models / n_row))
    cal_errs, probs_true, probs_pred = expected_calibration_error(true_models, pred_models, num_bins)

    # Initialize figure
    if fig_size is None:
        fig_size = (int(5 * n_col), int(5 * n_row))
    fig, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    if n_row > 1:
        ax = axarr.flat

    # Plot marginal calibration curves in a loop
    if n_row > 1:
        ax = axarr.flat
    else:
        ax = axarr
    for j in range(num_models):
        # Plot calibration curve
        ax[j].plot(probs_pred[j], probs_true[j], "o-", color=color)

        # Plot PMP distribution over bins
        uniform_bins = np.linspace(0.0, 1.0, num_bins + 1)
        norm_weights = np.ones_like(pred_models) / len(pred_models)
        ax[j].hist(pred_models[:, j], bins=uniform_bins, weights=norm_weights[:, j], color="grey", alpha=0.3)

        # Plot AB line
        ax[j].plot((0, 1), (0, 1), "--", color="black", alpha=0.9)

        # Tweak plot
        ax[j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax[j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)
        ax[j].set_title(model_names[j], fontsize=title_fontsize)
        ax[j].spines["right"].set_visible(False)
        ax[j].spines["top"].set_visible(False)
        ax[j].set_xlim([0 - epsilon, 1 + epsilon])
        ax[j].set_ylim([0 - epsilon, 1 + epsilon])
        ax[j].set_xlabel("Predicted probability", fontsize=label_fontsize)
        ax[j].set_ylabel("True probability", fontsize=label_fontsize)
        ax[j].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        # Plot without grid 
        # ax[j].grid(alpha=0.5)

        # Add ECE label
        ax[j].text(
            0.1,
            0.9,
            r"$\widehat{{\mathrm{{ECE}}}}$ = {0:.3f}".format(cal_errs[j]),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax[j].transAxes,
            size=legend_fontsize,
        )

    fig.tight_layout()
    return fig