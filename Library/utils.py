# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os
import pandas as pd
import config
import numpy as np
from scipy import stats
from scipy.special import softmax
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib as mpl


def plot_training(H, output_path):
    def chart_lines(ax, line1, line2,
                    line1_label, line2_label,
                    x_label, y_label):
        ax.plot(line1, label=line1_label)
        ax.plot(line2, label=line2_label)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(loc="lower left")

    # build a dataframe
    history_data = pd.DataFrame({
        "Train loss": H.history["loss"],
        "Train acc": H.history["acc"],
        "Validation loss": H.history["val_loss"],
        "Validation acc": H.history["val_acc"],  # not accuracy in the conventional sense
    })
    data_path = os.path.sep.join([
        output_path,
        f"Train history, {config.FEATURE_DISTANCE}.csv"
    ])
    history_data.to_csv(data_path, index=False)

    # plot a figure (smaller, safe size)
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Training metrics")

    # plot loss
    chart_lines(ax=ax1,
                line1=history_data["Train loss"],
                line2=history_data["Validation loss"],
                line1_label="Train",
                line2_label="Validation",
                x_label="Epoch", y_label="Loss")

    # plot acc
    chart_lines(ax=ax2,
                line1=history_data["Train acc"],
                line2=history_data["Validation acc"],
                line1_label="Train",
                line2_label="Validation",
                x_label="Epoch", y_label="Accuracy (not conventional sense)") 

    # save safely (lower dpi, png instead of jpg)
    fig.savefig(
        os.path.sep.join([output_path, f"Plot train, {config.FEATURE_DISTANCE}.png"]),
        dpi=100,
        bbox_inches="tight"
    )
    plt.close(fig)  # free memory

	

def embedding_PDFs_1D(embedding_X, source_list=["noise", "qb", "le"]):
    """
    embedding_X: {"noise": data,
                  "qb": data,
                  "le": data}
    return embeddings_X_PDFs = {
                                "Norm": {"noise": obj, "qb": obj, "le": obj}
                                "Kernel": {"noise": obj, "qb": obj, "le": obj}
                                }
    """

    PDF_Normals = {}
    PDF_KDEs = {}

    for source_label in source_list:
        values = np.array(embedding_X[source_label])

        if len(values)>0:
            # normal distribution
            PDF_Normals[source_label]  = stats.norm(np.mean(values), np.std(values)) 

            # Kernel density estimation
            PDF_KDEs[source_label] = stats.gaussian_kde(values.T)                    

    embeddings_X_PDFs = {"Norm": PDF_Normals,
                        "Kernel": PDF_KDEs}

    return embeddings_X_PDFs



def embedding_PDFs_3D(embedding_Z, embedding_N, embedding_E,
                      source_list=["noise", "qb", "le"]):
    """
    embedding_Z/N/E: {"noise": data,
                    "qb": data,
                    "le": data}
    return embeddings_3C_PDFss = {
                                "Norm": {"noise": obj, "qb": obj, "le": obj}
                                "Kernel": {"noise": obj, "qb": obj, "le": obj}
                                }
    """   
    PDF_Normals = {}
    PDF_KDEs = {}

    # build embedding samples
    for source_label in source_list:
        values_Z = np.array(embedding_Z[source_label])
        values_N = np.array(embedding_N[source_label])
        values_E = np.array(embedding_E[source_label])

        if len(values_Z)>0:

            samples_3D = np.column_stack((values_E, values_N, values_Z))

            # normal distribution
            means_estimated = np.mean(samples_3D, axis=0)
            cov_matrix_estimated = np.cov(samples_3D.T)
            PDF_Normals[source_label] = stats.multivariate_normal(mean=means_estimated, cov=cov_matrix_estimated)

            # Kernel density estimation
            PDF_KDEs[source_label] = stats.gaussian_kde(samples_3D.T)     
    
    embeddings_3C_PDFs = {"Norm": PDF_Normals,
                        "Kernel": PDF_KDEs}

    return embeddings_3C_PDFs



def infer_3C_PDFs(input_embeddings_3C, embeddings_3C_PDFs,
                    choose_pdf="Kernel"):
    
    # evet likelihood
    likelihood_dict = \
        embedding_likelihood(input_embeddings_3C, embeddings_3C_PDFs[choose_pdf])

    # softmax values
    likelihood_softmax = softmax([likelihood_dict["noise"],
                                  likelihood_dict["qb"],
                                  likelihood_dict["le"]])
    
    # pred labels
    infer_type = np.argmax(likelihood_softmax)

    return int(infer_type), likelihood_dict, list(np.round(likelihood_softmax, 4))



def two_class_convert(true_list, pred_list):
    """
    0: noise
    1: seismic
    """
    _temp_data = np.array(true_list)
    _temp_data[_temp_data>1] = 1
    two_class_true = list(_temp_data)

    _temp_data = np.array(pred_list)
    _temp_data[_temp_data>1] = 1
    two_class_pred = list(_temp_data)    

    return two_class_true, two_class_pred




def calc_confusion_metrics(true_labels, predicted_labels):
    # create confusion matrix
    cnf_matrix = confusion_matrix(true_labels, predicted_labels)   # , normalize='all'

    # calc metrics for all classes at once
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    # Overall accuracy: there are several accuracies, including but not limited to macro-average acc., micro-average acc..
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)

    # F1-score
    F1 = 2*(PPV*TPR)/(PPV+TPR)

    # calculate weighted average metrics
    ACC_avg = np.sum((cnf_matrix.sum(axis=1) * ACC))/cnf_matrix.sum()
    TPR_avg = np.sum((cnf_matrix.sum(axis=1) * TPR))/cnf_matrix.sum()
    TNR_avg = np.sum((cnf_matrix.sum(axis=1) * TNR))/cnf_matrix.sum()
    PPV_avg = np.sum((cnf_matrix.sum(axis=1) * PPV))/cnf_matrix.sum()
    NPV_avg = np.sum((cnf_matrix.sum(axis=1) * NPV))/cnf_matrix.sum()
    FPR_avg = np.sum((cnf_matrix.sum(axis=1) * FPR))/cnf_matrix.sum()
    FNR_avg = np.sum((cnf_matrix.sum(axis=1) * FNR))/cnf_matrix.sum()
    FDR_avg = np.sum((cnf_matrix.sum(axis=1) * FDR))/cnf_matrix.sum()

    F1_avg = np.sum((cnf_matrix.sum(axis=1) * F1))/cnf_matrix.sum()

    metrics = {"Accuracy": list(ACC), "Accuracy (avg.)": ACC_avg,
               "True positive rate": list(TPR), "True positive rate (avg.)": TPR_avg,
               "True negative rate": list(TNR), "True negative rate (avg.)": TNR_avg,
               "Positive predictive value": list(PPV), "Positive predictive value (avg.)": PPV_avg,
               "Negative predictive value": list(NPV), "Negative predictive value (avg.)": NPV_avg,
               "False positive rate": list(FPR), "False positive rate (avg.)": FPR_avg,
               "False negative rate": list(FNR), "False negative rate (avg.)": FNR_avg,
               "False discovery rate": list(FDR), "False discovery rate (avg.)": FDR_avg,
               "F1-score": list(F1), "F1-score (avg.)": F1_avg,
               }

    for key, value in metrics.items():
        print(f"{key}: {value}")

    return cnf_matrix, metrics



def latent_codes_1D(data, model):
    _input = np.array(data).reshape(1, -1, 1)

    _embedding = model(_input).numpy()[0].reshape(1, -1)

    # keras v3
    # _embedding = list(model(_input).values())[0].numpy()[0].reshape(1, -1)

    return _embedding[0]



def infer_1C_PDFs(input_embedding, embeddings_PDFs, choose_pdf="Kernel"):
    """
    embeddings_PDFs: Probability density function object
                    {
                    "Kernel": obj,
                    "Norm": obj,
                    }

    0: noise
    1: qb
    2: le

    return: infer_type, likelihood_dict, likelihood_softmax
    """
    # distribution likelihood
    likelihood_dict = \
        embedding_likelihood(input_embedding, embeddings_PDFs[choose_pdf])

    # softmax values
    likelihood_softmax = softmax([likelihood_dict["noise"],
                                  likelihood_dict["qb"],
                                  likelihood_dict["le"]])
    # pred labels
    infer_type = np.argmax(likelihood_softmax)

    return int(infer_type), likelihood_dict, list(np.round(likelihood_softmax, 4))



def embedding_likelihood(input_embedding, embeddings_PDF):

    likelihood_noise = embeddings_PDF["noise"].pdf(input_embedding)

    if "qb" in embeddings_PDF.keys():
        likelihood_qb = embeddings_PDF["qb"].pdf(input_embedding)
    else:
        likelihood_qb = np.zeros(len(input_embedding))

    likelihood_le = embeddings_PDF["le"].pdf(input_embedding)

    return {"noise": float(likelihood_noise),
            "qb": float(likelihood_qb), 
            "le": float(likelihood_le)}



def plot_confusion(title, true_labels, matrix,
                   metrics = None,
                   fig_title="Confusion matrix", axLabelSize=8, axTickSize=8,
                   fig_size=[5, 3.9],
                   subAdjust=(0.18, 0.82, 0.2, 0.8)):


    mpl.rcParams['axes.titlesize'] = axLabelSize    
    mpl.rcParams['xtick.labelsize'] = axTickSize  
    mpl.rcParams['ytick.labelsize'] = axTickSize
    mpl.rcParams['axes.labelsize'] = axTickSize    
    mpl.rcParams['axes.labelpad'] = 2               
    mpl.rcParams['legend.fontsize'] = axTickSize   
    mpl.rcParams['savefig.dpi'] = 300 

    cm = 1/2.54
    fig = plt.figure(
                    # layout='constrained',
                    figsize=(fig_size[0]*cm, fig_size[1]*cm),
                    dpi=165)
    
    fig.subplots_adjust(left=subAdjust[0], right=subAdjust[1],
                        bottom=subAdjust[2], top=subAdjust[3],
                        )  

    fig.canvas.manager.set_window_title(fig_title)
    ax = fig.add_subplot(111)

    ax.set_title(title, size=axLabelSize)
    ax.set_ylabel('Ground truth')

    # plot confusion matrix
    ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=true_labels).plot(
                            include_values=True, ax=ax, colorbar=False,
                            values_format="", text_kw={"fontsize":axTickSize},
                            cmap="Blues")
    
    if metrics == None:
        # Remove x-axis labels and ticks
        x_ticklabels = ['' for i in range(len(matrix))]
    else:
        # add precision, or positive predictive value
        x_ticklabels = np.round(np.array(metrics["Positive predictive value"])*100, 2)
        ax.annotate('PPV:', xy=(0, 0), xycoords=ax.get_xaxis_transform(),
                   xytext=(-33, -13), textcoords="offset points",
                   fontsize=axTickSize)
        
        # add recall, or true positive rate
        y_ticklabels = np.round(np.array(metrics["True positive rate"])*100, 2)

        num_tick = len(matrix) * 2 + 1
        vals = np.linspace(-0.5, len(matrix)-0.5, num_tick)
        tick_value = vals[1:-1:2] + axTickSize*0.005

        for val, lab in zip(tick_value, y_ticklabels):
            ax.text(1.1, val, lab, transform=ax.get_yaxis_transform(),
                    fontsize=axTickSize)
            
        ax.text(1.1, -0.55, "TPR:", transform=ax.get_yaxis_transform(),
                fontsize=axTickSize)


    # add metrics
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    ax.xaxis.set_ticklabels(x_ticklabels)
    ax.xaxis.set_label_position('top')
    ax.set_xlabel('Predicted')
    ax.set_aspect(aspect='equal')

    return fig