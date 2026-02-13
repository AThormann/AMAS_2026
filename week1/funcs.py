import numpy as np
import matplotlib.pyplot as plt
from scipy import stats 
from scipy import special                      
from iminuit import Minuit
from iminuit.cost import LeastSquares
import pandas as pd
import seaborn as sns
from IPython.core.display import Latex
import sympy as sp
from numpy.linalg import inv




# ------------------------------------------------------------------------------------- #
#colorpalette and plot style settings 

sns.set()
sns.set_context("paper", font_scale=1.4)
sns.set_style("ticks")

colors = sns.color_palette('deep', 10, desat = 0.8)
colors = colors[::-1]
colors.pop(2)

sns.set_palette(colors)

sns.palplot(colors)
plt.rcParams['axes.grid'] = True
#plt.rcParams['axes.grid.axis'] = 'y'
#plt.rcParams['axes.grid.which'] = 'major'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 1.5
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 150
plt.style.use('seaborn-v0_8')

plt.rcParams['legend.facecolor'] = 'white'

rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]




# --------------------------------------------------------------------------------------------- #
#plotting


def histogram_plot(data, Nbins, xmin, xmax,label, xlabel="value", ylabel="counts", title="", join=False, ax=None):
    if join == False:
        fig, ax = plt.subplots(figsize=(16, 6))
    else:
        ax=ax
    hist = ax.hist(data, bins=Nbins, range=(xmin, xmax), histtype='step', alpha=0.8, linewidth=2, label=label)
    counts, bin_edges = np.histogram(data, bins=Nbins, range=(xmin, xmax))
    #mask out the empty bins:
    x = (bin_edges[1:][counts>0] + bin_edges[:-1][counts>0])/2 
    y = counts[counts>0]
    sy = np.sqrt(counts[counts>0]) # poisson error


    ax.errorbar(x, y, yerr=sy, xerr=0.0, label='Bin centers (poisson error)', fmt='.k',  ecolor='k', elinewidth=1, capsize=1, capthick=1)

    # Set the figure texts; xlabel, ylabel and title.
    hist_info = [f'Nbins = {Nbins},\nxmin = {xmin:.2f}, xmax = {xmax:.2f}']
    ax.set(xlabel=xlabel,           # the label of the y axis
       ylabel=ylabel,           # the label of the y axis
       title=title)    # the title of the plot
    ax.text(0.7, 0.95, "\n".join(hist_info), transform=ax.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    #ax.legend(title="\n".join(hist_info), fontsize=18, title_fontsize = 18, alignment = 'center');       # could also be # loc = 'upper right' e.g.   
    return counts, bin_edges, sy, ax


def plot_fit(expr,mfit,  xmin, xmax, label, x, y, ax=None):
    params = mfit.values[:]
    chi2 = mfit.fval
    Ndof = len(x[y > 0]) - mfit.nfit
    Prob = stats.chi2.sf(chi2, Ndof) 
    x_axis = np.linspace(xmin, xmax,1000)
    fit_vals = expr(x_axis, *params)
    ax.plot(x_axis, fit_vals, label = label)
    fit_info = [f"$\\chi^2$ / $N_\\mathrm{{dof}}$ = {chi2:.1f} / {Ndof}", f"P($\\chi^2$, $N_\\mathrm{{dof}}$) = {Prob:.3f}",]
    for p, v, e in zip(mfit.parameters, mfit.values[:], mfit.errors[:]) :
        Ndecimals = max(0,-np.int32(np.log10(e)-1-np.log10(2)))                                # Number of significant digits
        fit_info.append(f"{p} = ${v:{10}.{Ndecimals}{"f"}} \\pm {e:{10}.{Ndecimals}{"f"}}$")
    ax.legend(title="\n".join(fit_info), fontsize=18, title_fontsize = 18, alignment = 'center')
    return chi2, Ndof, Prob, ax






# ---------------------------------------------------------------------------------------------------- #

#print function for LaTeX in Jupyter notebooks
def lprint(*args,**kwargs):
    """Pretty print arguments as LaTeX using IPython display system 
    
    Parameters
    ----------
    args : tuple 
        What to print (in LaTeX math mode)
    kwargs : dict 
        optional keywords to pass to `display` 
    """
    display(Latex('$$'+' '.join(args)+'$$'),**kwargs)



# Error propagation function

def propagate_error(expr, vars, vals, sigmas, corr=None, show=True):

    n = len(vars)

    if corr is None:
        corr = np.eye(n)

    corr = np.array(corr, dtype=float)

  
    sigma_syms = sp.symbols([f"sigma_{v}" for v in vars])
    Sigma_sym = sp.zeros(n,n)

    for i in range(n):
        for j in range(n):
            Sigma_sym[i,j] = corr[i,j] * sigma_syms[i] * sigma_syms[j] #covariance element

    
    grad = sp.Matrix([sp.diff(expr, v) for v in vars])
    sigma_expr = sp.sqrt((grad.T * Sigma_sym * grad)[0])

    subs = {vars[i]: vals[i] for i in range(n)}
    subs.update({sigma_syms[i]: sigmas[i] for i in range(n)})

    f_val = float(expr.subs(subs))
    sigma_val = float(sigma_expr.subs(subs))

    if show:
        lprint(r"f = " + sp.latex(expr))

        lprint(
            r"\nabla f = \begin{pmatrix}" +
            r"\\".join([sp.latex(g) for g in grad]) +
            r"\end{pmatrix}"
        )

        lprint(
            r"\Sigma = " + sp.latex(Sigma_sym)
        )

        lprint(
            r"\sigma_f = \sqrt{ \nabla f^T \Sigma \nabla f } =" + sp.latex(sigma_expr)
        )

        lprint(
            rf"f = ({f_val:.4g} \pm {sigma_val:.4g})"
        )

    return f_val, sigma_val, sigma_expr





# ---------------------------------------------------------------------------------------------------- #
#fisher and ROC curve separation

#calc_seperation is taken from fisher_discriminant_ExampleSolution.ipynb
def calc_separation(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    d = np.abs((mean_x - mean_y)) / np.sqrt(std_x**2 + std_y**2)
    return d


def scatter_cov_matrix(vars_1, vars_2, labels,
                                   group1_name="Group 1",
                                   group2_name="Group 2",
                                   figsize=(10, 10),
                                   point_size=15,
                                   alpha=0.5,
                                   annotate = True):

    n = len(vars_1)

    if len(vars_2) != n or len(labels) != n:
        raise ValueError("vars_1, vars_2, and labels must have the same length")

   
    X1 = np.vstack(vars_1).T  
    X2 = np.vstack(vars_2).T  
    cov_1 = np.cov(X1, rowvar=False, ddof=1)
    cov_2 = np.cov(X2, rowvar=False, ddof=1)

    
    fig, axes = plt.subplots(n, n, figsize=figsize, sharex="col", sharey="row")

    for i in range(n):      
        for j in range(n):  
            ax = axes[i, j]

            
            if i == j:
                continue

            
            ax.scatter(X1[:, j], X1[:, i], s=point_size, alpha=alpha, label=group1_name)
            ax.scatter(X2[:, j], X2[:, i], s=point_size, alpha=alpha, label=group2_name)

            
            if annotate:
                ax.text(
                    0.05, 0.95,
                    f"{group1_name}: {cov_1[i,j]:.2f}\n{group2_name}: {cov_2[i,j]:.2f}",
                    transform=ax.transAxes,
                    ha="left", va="top",
                    fontsize=8)

        
            if i == n - 1:
                ax.set_xlabel(labels[j])
            if j == 0:
                ax.set_ylabel(labels[i])

    
    axes[0,0].legend()
    plt.tight_layout()
    plt.show()

    return X1, X2, cov_1, cov_2



def fisher_coeff(cov_1, cov_2, X1, X2):
    mu_1 = np.mean(X1, axis=0)
    mu_2 = np.mean(X2, axis=0)
    joint_cov = cov_1.copy() + cov_2.copy()
    joint_cov_inv = inv(joint_cov.copy())
    wf = np.dot(joint_cov_inv, (mu_2 - mu_1))
    return wf
    
def plot_fisher(wf, X1, X2, Nbins, label1, label2):
    fisher_1 = X1 @ wf
    fisher_2 = X2 @ wf
    fisher = np.vstack([fisher_1, fisher_2])
    xmin= np.min(fisher)
    xmax = np.max(fisher)
    
    counts_1, bin_edges_1,sy_1,ax = histogram_plot(fisher_1, Nbins, xmin=xmin, xmax=xmax,label=label1, xlabel="", ylabel="counts", title="Fisher seperation", join=False)
    counts_2,bin_edges_2,sy_2,ax =histogram_plot(fisher_2, Nbins, xmin=xmin, xmax=xmax,label=label2, xlabel="", ylabel="counts", title="Fisher seperation", join=True, ax=ax)
    d= calc_separation(fisher_1,fisher_2)
    plt.text(xmax-1.5, np.max(counts_2), f"Seperation: {d:.2}")
    plt.legend()
    return counts_1, bin_edges_1, counts_2, bin_edges_2, d


#this function is heavily influenced by the file MakeROCfigure.ipynb from the lecture repository

def calc_ROC(counts_1, edges_1, counts_2, edges_2):


    y_sig, x_sig_edges = counts_1, edges_1
    y_bkg, x_bkg_edges = counts_2, edges_2

    
    if not np.array_equal(x_sig_edges, x_bkg_edges):
        raise ValueError("Signal and background histograms have different bins")

    # integration
    S = y_sig.sum()
    B = y_bkg.sum()

    n_bins = len(y_sig)

    TPR = np.zeros(n_bins)  
    FPR = np.zeros(n_bins)   

    
    for i in range(n_bins):
        #true positives 
        TP = np.sum(y_sig[i:])

        #false negatives 
        FN = np.sum(y_sig[:i])

        #false positives 
        FP = np.sum(y_bkg[i:])

        #true negatives 
        TN = np.sum(y_bkg[:i])

        # Rates
        TPR[i] = TP / (TP + FN)  
        FPR[i] = FP / (FP + TN)

    return FPR, TPR






#-------------------------------------------------------------------------------------#



# Cleaning data:

def ChauvenetsCriterion( inlist, pmin = 0.05, CCverbose = True) :

    if (type(inlist) is not list) :
        print(f"Error: Input to function ChauvenentsCriterion is NOT a list!")
        return [-9999999999.9, -99999999999.9]

    outlist = [entry for entry in inlist]
    mean, std = np.mean(inlist), np.std(inlist)   # Calculation of initial mean and rms.
    
    # Loop over the following iterations, until the furthest outlier is probably enough (p_any_outliers > pmin):
    while True :
        # Find the furthers outlier, i.e. most distant measurement from mean (least probable) and its index:
        ifurthest  = 0
        dLfurthest = 0.0
        for number, entry in enumerate( outlist ) :
            if (abs(entry - mean) > dLfurthest) :
                ifurthest = number                       # Note the index, so that this entry can later be removed!
                dLfurthest = abs(entry - mean)

        # Calculate the probability of any such outliers (taking into account that there are many measurements!):
        Nsigma_outlier = dLfurthest / std
        p_this_outlier = special.erfc(Nsigma_outlier / np.sqrt(2)) / 2.0
        p_any_outliers = 1.0 - (1.0 - p_this_outlier)**int(len(outlist))

        if (CCverbose) :
            print(f" {ifurthest:3d}: L={outlist[ifurthest]:5.3f}  dL={dLfurthest:5.3f}  Nsig={Nsigma_outlier:5.2f}" +
                  f" p_loc={p_this_outlier:10.8f}  p_glob={p_any_outliers:10.8f} >? pmin={pmin:5.3f}   N={len(outlist):3d}" +
                  f" mean={mean:6.4f}  std={std:6.4f}", end="")
            
        # Key line: If the furthest outlier is probably enough, then stop rejecting points:
        if (p_any_outliers > pmin) :
            if (CCverbose) : print(f"  -> Accepted")
            break

        # Remove the furthest point from the list of accepted measurements (if any are left!),
        # Recalculate mean and RMS, and finally reiterate:
        if (len(outlist) > 1) :
            if (CCverbose) : print(f"  -> Rejected")
            outlist.pop(ifurthest)
            mean, std = np.mean(outlist), np.std(outlist)
        else :
            print(f"\n  ERROR: All measurements have been rejected!")
            break

    print(f"  The number of accepted / rejected points is {len(outlist)} / {len(inlist)-len(outlist)}")
    return outlist





     

