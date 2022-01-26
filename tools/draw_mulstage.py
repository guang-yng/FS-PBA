import os
import torch
import matplotlib.pyplot as plt
import seaborn

label_to_name = ['adapter', 'bias', 'prompt']
label_to_color = ['r', 'c', 'b']

for i in range(3):
    for j in range(3):
        for k in range(3):
            if i == j or i == k or j == k: 
                continue
            a = label_to_name[i]
            b = label_to_name[j]
            c = label_to_name[k]
            for n in ['', 'N-']:
                best_loss = 1
                best_lr = None
                for lr in ['1e-2', '1e-3', '1e-4', '1e-5']:
                    path = os.path.join('../', 'result', 'SST-2-full-mul-'+a+'-'+b+'-'+c+'-'+n+lr+'-13')
                    log = torch.load(os.path.join(path, 'log_history.bin'))
                    if log[-1]['eval_loss'] < best_loss:
                        best_loss = log[-1]['eval_loss']
                        best_lr = lr
                path = os.path.join('../', 'result', 'SST-2-full-mul-'+a+'-'+b+'-'+c+'-'+n+best_lr+'-13')

                log = torch.load(os.path.join(path, 'log_history.bin'))
                print(log)
                train_loss = []
                for it, item in enumerate(log):
                    if 'eval_acc' in item and 'eval_acc' not in log[it-1]:
                        train_loss.append(item['eval_acc'])
                print(train_loss)
                exit(0)
                
                assert(len(train_loss) == 90)
                fig, ax = plt.subplots()
                ax.plot(range(200, 6200, 200), train_loss[:30], color=label_to_color[i], linewidth=2.0, label=label_to_name[i])
                ax.plot(range(6000, 12200, 200), train_loss[29:60], color=label_to_color[j], linewidth=2.0, label=label_to_name[j])
                ax.plot(range(12000, 18200, 200), train_loss[59:], color=label_to_color[k], linewidth=2.0, label=label_to_name[k])
                ax.set_ylim(0.8, 1.0)
                ax.set_xlim(0)
                ax.legend()
                if n == '':
                    plt.title("Evaluating Acc for "+a+'-'+b+'-'+c+" method with template")
                else:
                    plt.title("Evaluating Acc for "+a+'-'+b+'-'+c+" method without template")
                # path = os.path.join('../', 'result', 'SST-2-full-mul-'+a+'-'+b+'-'+c+'-'+n+'13')
                path = os.path.join('../', 'figure')
                if not os.path.isdir(path):
                    os.mkdir(path)
                plt.savefig(os.path.join(path, a+'-'+b+'-'+c+'.png'))
                

