"""Compact spectrum/runtime figures with the original publication geometry."""
from pathlib import Path
import numpy as np
from .paths import DATA


def plot_comparison(case, grid, reference, curves, timings, destination):
    """Save a 5.9 x 2.65 inch vector PDF and PNG; return the PDF path.

    `timings` contains seconds. IR uses a common reference-peak normalization;
    emission uses absolute intensity. Insets never rescale individual curves.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator
    for font in (DATA/'fonts').glob('Tinos-*.ttf'):
        font_manager.fontManager.addfont(str(font))
    methods=['tangent1','cocheb','tdvp1']
    labels={'tangent1':'TaChe 1-site','cocheb':'coCheb','tdvp1':'TDVP 1-site'}
    colors={'tangent1':'#2171B5','cocheb':'#238B45','tdvp1':'#B34D88'}
    styles={'tangent1':'-','cocheb':(0,(5,1.4,1,1.4)),'tdvp1':(0,(1,1.3))}
    widths={'tangent1':1.15,'cocheb':.9,'tdvp1':.9}
    marks={'cocheb':dict(marker='s',markersize=2.1,markeredgewidth=.2),
           'tdvp1':dict(marker='o',markersize=2.2,markeredgewidth=.2)}
    scale=max(reference) if case=='IR' else 1.
    style={'font.family':'serif','font.serif':['Tinos'],'mathtext.fontset':'stix',
        'font.size':10,'axes.labelsize':10.5,'xtick.labelsize':9.5,'ytick.labelsize':9.5,
        'legend.fontsize':9,'axes.linewidth':.8,'lines.linewidth':1.1,
        'xtick.direction':'in','ytick.direction':'in','xtick.major.size':3,'ytick.major.size':3,
        'xtick.major.width':.7,'ytick.major.width':.7,'xtick.minor.size':1.6,'ytick.minor.size':1.6,
        'xtick.top':True,'ytick.right':True,'pdf.fonttype':42,'pdf.compression':9,
        'savefig.facecolor':'white','figure.facecolor':'white','axes.unicode_minus':True,'path.simplify':False}
    def layers(ax,inset=False):
        ax.fill_between(grid,0,reference/scale,color='#E1E1E1',edgecolor='none',zorder=.5)
        ax.plot(grid,reference/scale,color='#B0B0B0',lw=1.9 if inset else 2.1,zorder=1)
        for z,m in enumerate(['cocheb','tdvp1','tangent1'],2):
            y=curves[m]/scale
            ax.plot(grid,y,color=colors[m],ls=styles[m],lw=widths[m]-(.1 if inset else 0),zorder=z)
        # Place markers by distance along the visible curve in typographic points.
        for m,marker in marks.items():
            y=curves[m]/scale
            xmin,xmax=ax.get_xlim();ymin,ymax=ax.get_ylim()
            visible=np.flatnonzero((grid>=xmin)&(grid<=xmax)&(y>=max(0,ymin)+.012*(ymax-ymin))&(y<=ymax))
            box=ax.get_position();fw,fh=ax.figure.get_size_inches()
            sx=box.width*fw*72/(xmax-xmin);sy=box.height*fh*72/(ymax-ymin)
            selected=[];spacing=16. if inset else 24.;offset=.25 if m=='cocheb' else .75
            for segment in np.split(visible,np.flatnonzero(np.diff(visible)>1)+1):
                if not len(segment):continue
                distance=np.r_[0,np.cumsum(np.hypot(np.diff(grid[segment])*sx,np.diff(y[segment])*sy))]
                targets=np.arange(offset*spacing,distance[-1],spacing)
                if not len(targets):targets=np.array([offset*distance[-1]])
                right=np.searchsorted(distance,targets).clip(0,len(segment)-1);left=np.maximum(right-1,0)
                nearest=np.where(abs(distance[left]-targets)<abs(distance[right]-targets),left,right)
                selected.extend(segment[np.unique(nearest)])
            ids=np.asarray(selected,dtype=int)
            ax.plot(grid[ids],y[ids],ls='none',color=colors[m],zorder=12,**marker)
    with plt.rc_context(style):
        fig=plt.figure(figsize=(5.9,2.65))
        ax=fig.add_axes([.095,.58/2.65,.455,1.46/2.65])
        at=fig.add_axes([.765,.58/2.65,.215,1.46/2.65])
        if case=='IR':
            ax.set(xlim=(0,4000),ylim=(-.025,1.08),xlabel=r'Wavenumber (cm$^{-1}$)',ylabel='IR intensity (a.u.)')
            ax.set_xticks([0,1000,2000,3000,4000]);ax.set_yticks([0,.5,1.])
            center=10*round(float(grid[np.argmax(abs(curves['cocheb']-reference))])/10)
            xlim=(center-45,center+45);local=(grid>=xlim[0])&(grid<=xlim[1])
            peak=max(max(reference[local]),*(max(curves[m][local]) for m in methods))/scale
            ylim=(-.025*peak,1.12*peak);xticks=[center-20,center+20]
            ticks=MaxNLocator(nbins=2,steps=[1,2,5,10]).tick_values(0,ylim[1])
            yticks=ticks[(ticks>=0)&(ticks<=ylim[1])]
        else:
            ax.set(xlim=(-2,8),ylim=(-.004,.24),xlabel=r'Frequency / (1400 cm$^{-1}$)',ylabel='Emission intensity')
            ax.set_xticks([-2,0,2,4,6,8]);ax.set_yticks([0,.1,.2])
            center=round(float(grid[np.argmax(reference)]),1)
            xlim=(center-.35,center+.35);ylim=(0,1.08*max(reference))
            xticks=[round(center-.2,1),round(center+.2,1)];yticks=[.1,.2]
        layers(ax)
        inset=ax.inset_axes([.63,.49,.35,.43]);inset.set(xlim=xlim,ylim=ylim)
        layers(inset,True);inset.set_xticks(xticks);inset.set_yticks(yticks)
        inset.tick_params(labelsize=9,length=2,pad=1)
        for spine in inset.spines.values():spine.set_linewidth(.6)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2));ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        values=[timings[m]/60 for m in methods];limit=max(values)*1.5
        at.barh(np.arange(3),values,height=.58,color=[colors[m] for m in methods],edgecolor='none')
        at.set_yticks(np.arange(3),[labels[m] for m in methods]);at.tick_params(axis='y',length=0,pad=5,labelsize=9.4)
        at.set(ylim=(2.6,-.6),xlim=(0,limit),xlabel='Time (min)')
        ticks=MaxNLocator(nbins=3).tick_values(0,limit);at.set_xticks(ticks[(ticks>=0)&(ticks<=limit)])
        at.xaxis.set_minor_locator(AutoMinorLocator(2))
        for i,value in enumerate(values):at.text(value+.025*limit,i,f'{value:.2f}',va='center',fontsize=9)
        ax.text(.025,.96,'(a) '+rf'$D={16 if case=="IR" else 32}$',transform=ax.transAxes,ha='left',va='top',fontsize=10.5)
        at.text(.96,.98,'(b)',transform=at.transAxes,ha='right',va='top',fontsize=10.5)
        handles=[Line2D([],[],color=colors[m],ls=styles[m],lw=widths[m],**marks.get(m,{})) for m in methods]
        handles.append(Patch(facecolor='#E1E1E1',edgecolor='#B0B0B0',linewidth=.8))
        legend=fig.legend(handles,[labels[m] for m in methods]+['reference'],loc='upper center',
            bbox_to_anchor=(.53,.995),ncol=2,frameon=True,fancybox=False,edgecolor='black',framealpha=1,
            borderpad=.35,columnspacing=1.3,handlelength=2.4,handletextpad=.5,labelspacing=.2)
        legend.get_frame().set_linewidth(.6)
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer();fw,fh=fig.canvas.get_width_height()
        for text in fig.findobj(match=matplotlib.text.Text):
            if text.get_visible() and text.get_text():
                box=text.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > fw+1 or box.y1 > fh+1:
                    raise RuntimeError(f'Figure text extends outside the canvas: {text.get_text()}')
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination,metadata={'Title':'','Author':'','CreationDate':None,'ModDate':None})
        fig.savefig(destination.with_suffix('.png'),dpi=240)
        plt.close(fig)
    return destination
