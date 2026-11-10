"""Standalone publication-style plots of actual paired results and training history."""
from common import ROOT,read


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=read(ROOT/'Output/metrics.json')['results']
    output=ROOT/'Output/figures';output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(16,4.3),layout='constrained')
    conditions=['Original','JPEG Q15','Low-res 64','Dark x0.5','Bright x1.5']
    for ax,kind,groups,labels,field,title in [(axes[0],'llm',['target','general'],['JSON target (100)','General (100)'],'accuracy_percent','LLM exact answers'),
        (axes[1],'vlm',list(metrics['vlm']['before']),conditions,'semantic_accuracy_percent','VLM count content'),
        (axes[2],'vlm',list(metrics['vlm']['before']),conditions,'format_valid_percent','VLM integer format')]:
        x=list(range(len(groups)))
        ax.bar([i-0.18 for i in x],[metrics[kind]['before'][g][field] for g in groups],width=.36,label='Before',color='#9aaabd')
        ax.bar([i+0.18 for i in x],[metrics[kind]['after'][g][field] for g in groups],width=.36,label='DPO + LoRA',color='#246bad')
        ax.set_xticks(x,labels,rotation=20 if kind=='vlm' else 0);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_title(title);ax.legend(frameon=False)
    fig.savefig(output/'before_after.png',dpi=180);fig.savefig(output/'before_after.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,kind in zip(axes,['llm','vlm']):
        rows=read(ROOT/f'Output/{kind}/training.json')['log_history']
        train=[r for r in rows if 'loss' in r and 'step' in r]
        dev=[r for r in rows if 'eval_loss' in r and 'step' in r]
        ax.plot([r['step'] for r in train],[r['loss'] for r in train],alpha=.7,label='Train DPO loss')
        ax.plot([r['step'] for r in dev],[r['eval_loss'] for r in dev],'o-',label='Dev DPO loss')
        ax.set_xlabel('Optimizer step');ax.set_ylabel('DPO loss');ax.set_title(kind.upper());ax.legend(frameon=False)
    fig.savefig(output/'training_loss.png',dpi=180);fig.savefig(output/'training_loss.pdf');plt.close(fig)
    print('Saved figures: '+str(output))


if __name__=='__main__':main()
