import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from torch import device

mpl.rcParams['xtick.minor.size'] = 0
mpl.rcParams['xtick.minor.width'] = 0
mpl.rc('font', size=18)
xlabels = [16, 32, 64, 128, 256, 480]
xlabels_txt = [str(i) for i in xlabels]
print(xlabels, xlabels_txt)

log_file = open('../log', 'r')
log = log_file.readlines()

print(len(log))

for t in ['forward', 'infer']:

    param_to_time = {}
    for item in log:
        training_arg = eval(item)
        training_params = training_arg['training_params']
        param_str = training_params[0]
        max_len = training_arg['input_length']
        if param_str not in param_to_time:
            param_to_time[param_str] = {}
        param_to_time[param_str][max_len] = training_arg[f"sst-2_train_{t}_time"]

    label_to_name = ['adapter', 'bias', 'prompt', 'all', 'prompt,bias', 'prompt,adapter', 'bias,adapter', 'prompt,bias,adapter']
    label2n = ['AP', 'BF / FT', 'PT', 'FT', 'PT+BF', 'PT+AP', 'BF+AP', 'PT+BF+AP']
    label_to_color = ['#fe5803', '#c34f97', '#0044fb', '#f2a196', '#700a97', '#fd9404', '#d36c62']

    for x in xlabels:
        param_to_time['bias'][x] += param_to_time['all'][x]
        param_to_time['bias'][x] *= 0.5
    fig, ax = plt.subplots()
    for param in param_to_time:
        i = label_to_name.index(param)
        if i > 2:
            continue
        x = xlabels
        y = []
        for item in x:
            y.append(param_to_time[param][item])
        
        ax.plot(x, y, '-d', color=label_to_color[i], label=label2n[i])
    legend=ax.legend()
    # for txt in legend.get_texts():
    #     txt.set_ha('center') # ha is alias for horizontalalignment
    #     txt.set_position((20,0))

    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.title.set_text(f'forward time'.title())
    ax.set_ylabel('Time(ms)')
    ax.set_xlabel('Length of Inputs')
    ax.set_xscale('log')
    ax.set_xticks(xlabels)
    fontdict = {'fontsize': 12, 
    'fontweight': mpl.rcParams['axes.titleweight'] }
    ax.set_xticklabels(xlabels_txt, fontdict=fontdict)
    ymin, ymax = ax.get_ylim()
    # ylabel = [0.92, 0.93, 0.94, 0.95, 0.96, 0.97]
    ylabel = np.linspace(ymin, ymax, 5)
    if ylabel[0] < 0:
        print(ylabel[0])
        ylabel[0] = 0
    plt.yticks(ylabel)
    ylabel_txt = [round(i*1000, 1) for i in ylabel]
    ax.set_yticklabels(ylabel_txt)
    for y0 in ylabel:
        ax.axhline(y=y0, color='gray', linestyle='--', linewidth=0.5)
                    # path = os.path.join('../', 'result', 'SST-2-full-mul-'+a+'-'+b+'-'+c+'-'+n+'13')
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    tp = ax.figure.subplotpars.top
    bt = ax.figure.subplotpars.bottom
    figw = 6.0/(r-l)
    figh = 4.0/(tp-bt)
    ax.figure.set_size_inches(figw, figh)
    path = os.path.join('../', 'figure-time')
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.savefig(os.path.join(path, f'{t}.pdf'), bbox_inches='tight', pad_inches=0.1)
                    
