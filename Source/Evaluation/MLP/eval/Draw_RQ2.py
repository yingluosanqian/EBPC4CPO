import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from Source.Util.util import get_root_path

plt.rcParams['font.sans-serif'] = ['Times New Roman']


def load_result(coreset_name, feature_name, model_type):
    ls = []
    prefix = f'{get_root_path()}/Evaluation/MLP/eval'
    filepath = f'{prefix}/evaluation_result_{coreset_name}_{feature_name}_{model_type}.txt'
    with open(filepath, 'r') as f:
        for line in f:
            if 'geo_mean' in line and 'total' not in line:
                line = line.split(' ')[0]
                value = float(line.split('=')[1])
                ls.append(value)
    return ls


def work(data_bs):
    data_bs = [(np.array(data), label) for data, label in data_bs]

    # Set up the x-axis values (iterations 1 to 50)
    iterations = [i for i in range(1, 11)] + [i for i in range(15, 51, 5)]

    # Create the plot
    plt.figure(figsize=(13, 10))
    markers = ['*', '*', '.', '.', '^', '^', 'v', 'v']
    linestyles = [':', '-', ':', '-', ':', '-', ':', '-']
    markersizes = ['18', '18', '20', '20', '15', '15', '15', '15']
    for (data, label), marker, markersize, linestyle in zip(data_bs, markers, markersizes, linestyles):
        data = [data[idx - 1] for idx in iterations]
        plt.plot(iterations, data,
                 label=label,
                 marker=marker,
                 markerfacecolor='none',
                 markersize=markersize,
                 linestyle=linestyle)

    # Adding labels and title
    # plt.title('Comparison of Optimization Methods Over Iterations')
    plt.xlabel('Top K', fontsize=32)
    plt.ylabel('GMeanOverOz', fontsize=32)

    plt.legend(loc='lower right', fontsize='23', frameon=True, shadow=True, borderpad=1)
    plt.tick_params(axis='x', labelsize=30)
    plt.tick_params(axis='y', labelsize=30)

    # Adding grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.savefig('RQ2.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def work_table(data_bs):
    for i in [3, 5, 10]:
        sentence = '\\midrule'
        print(sentence)
        sentence = '\\multirow{2}*{k=' + str(i) + '} & Autophase'
        for data, _ in data_bs[::2]:
            sentence += f' & {data[i - 1]: .3f}'
        sentence += '\\\\'
        print(sentence)
        sentence = '& InstCount'
        for data, _ in data_bs[1::2]:
            sentence += f' & {data[i - 1]: .3f}'
        sentence += '\\\\'
        print(sentence)


if __name__ == '__main__':
    data_bs = []
    data_bs.append((load_result(f'bpc_coreset', f'Autophase', 'mlp'), 'BPC + Autophase'))
    data_bs.append((load_result(f'bpc_coreset', f'InstCountNorm', 'mlp'), 'BPC + InstCount'))
    data_bs.append((load_result(f'gens_coreset', f'Autophase', 'mlp'), 'GNES + Autophase'))
    data_bs.append((load_result(f'gens_coreset', f'InstCountNorm', 'mlp'), 'GENS + Instcount'))
    data_bs.append((load_result(f'nvp_coreset_2', f'Autophase', 'mlp'), 'NVP-2 + Autophase'))
    data_bs.append((load_result(f'nvp_coreset_2', f'InstCountNorm', 'mlp'), 'NVP-2 + InstCount'))
    data_bs.append((load_result(f'nvp_coreset_1', f'Autophase', 'mlp'), 'NVP-1 + Autophase'))
    data_bs.append((load_result(f'nvp_coreset_1', f'InstCountNorm', 'mlp'), 'NVP-1 + InstCount'))
    work(data_bs)

    data_bs = []
    data_bs.append((load_result(f'bpc_coreset', f'Autophase', 'mlp'), 'BPC + Autophase'))
    data_bs.append((load_result(f'bpc_coreset', f'InstCountNorm', 'mlp'), 'BPC + InstCount'))
    data_bs.append((load_result(f'nvp_coreset_1', f'Autophase', 'mlp'), 'NVP-1 + Autophase'))
    data_bs.append((load_result(f'nvp_coreset_1', f'InstCountNorm', 'mlp'), 'NVP-1 + InstCount'))
    data_bs.append((load_result(f'nvp_coreset_2', f'Autophase', 'mlp'), 'NVP-2 + Autophase'))
    data_bs.append((load_result(f'nvp_coreset_2', f'InstCountNorm', 'mlp'), 'NVP-2 + InstCount'))
    data_bs.append((load_result(f'gens_coreset', f'Autophase', 'mlp'), 'GNES + Autophase'))
    data_bs.append((load_result(f'gens_coreset', f'InstCountNorm', 'mlp'), 'GENS + Instcount'))
    work_table(data_bs)
    pass
