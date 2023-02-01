
import numpy as np
import re 
from isotope import ion as I
from starshot import Shot
from matplotlib import pylab as plt
import math

from starshot.ioncolor import ioncolor
from starshot.simplenet import SimpleNet
from starshot.kepnet import KepNet
from starshot.plotutil import fig_setup
from starshot.base import Base

from ionmap import decay
from abuset import AbuSet, AbuDump

import time # always good to have
import resource
from human import time2human
import color
import ionmap
from abusets import SolAbu

from isoplot import IsoPlot, IsoPlotMult
from nucplot import NucPlot
import matplotlib.patches as mpatches

from bsummer import onezonerflow, makearray, scale_list, rflowposneg


reactiondict = {0: '(n,$\gamma$) net', 1: '(n,$\gamma$) backwards', 2: '(p,n) net', 3: '(p,n) backwards', 4: '(p,$\gamma$) net', 5: '(p,$\gamma$) backwards', 6: '($\\alpha$,p) net', 7: '($\\alpha$,p) backwards', 8: '($\\alpha$,n) net', 9: '($\\alpha$,n) backwards', 10: '($\\alpha$,$\gamma$) net', 11: 'ag backwards', 12: 'weak net', 13: 'weak backwards', 14: 'b net', 'ng net': 0, 'ng backwards': 1, 'pn net': 2, 'pn backwards': 3, 'pg net': 4, 'pg backwards': 5, 'ap net': 6, 'ap backwards': 7, 'an net': 8, 'an backwards': 9, 'ag net': 10, 'ag backwards': 11, 'weak net': 12, 'weak backwards': 13, 'b net': 14}

save_results_to = 'Results/'

def get_super(x): #only for numbers
    x = float(x)
    x = int(x)
    x = str(x)
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def makelist(name, zone):
    useful = []
    for i,line in enumerate(name.flowb[zone].splitlines()):
            if line.startswith(' ========= FLOWS ========= '):
                for line in name.flowb[zone].splitlines()[i+13:]:
                    if 'total' not in line:
                            useful.append(line)
                    else:
                        return useful

def valuearray(listname):
    finallist = []
    ionlist = []
    for string in listname:
        newlist = [string[i*9:(i+1)*9] for i in range(16)]
        ion = I(newlist[0])
        c = [float(re.sub(r"(\d)([+-])", r"\1E\2", y)) for y in newlist[1:]]
        #print((templist))
        finallist.append(c)
        ionlist.append(ion)
    finallist = np.array(finallist, dtype = np.float64)
    ionlist = np.array(ionlist, dtype=object)
    return finallist, ionlist
    
def makearray(name, zone):
    return valuearray(makelist(name, zone))
    
def ionlist(name, zone):
        finallist, ionlist = makearray(name, zone)
        return ionlist

def findelement(name, element, zone):
        listname = ionlist(name, zone)
        for i in range(len(listname)):
            if str(listname[i]==element) == 'True':
                return i
        else:
            #print(f"{element} is not in list, set -1")
            i = -1
            return i
    
def allzones(name, ion):
    flows = []
    for i in range(1,int(name.flowb.shape[0])-1):
        ionnumber = findelement(name, ion, i)
        if ionnumber == -1:
            flows.append(np.array([ 0.0e+000,  0.0e+000,  0.0e+000,  0.0e+000,  0.0e+000,  0.0e+000,
         0.0e+000,  0.0e+000,  0.0e+000,  0.0e+000,  0.0e+000,  0.0e+000,
         0.0e+000,  0.0e+000, 0.0e+000]))
            #print("neg 1")
        else:
            #print(ionnumber)
            #print(i)
            flows.append(name.makearray(i)[0][ionnumber])
    return flows
    
def onereaction(name, ion, reactionnumber, steps):
    ionflows = []
    full = allzones(name,ion)
    #print((name.flowb.shape[0]))
    for i in range(0,int(name.flowb.shape[0])-2,steps):
        #print(i)
        ionflows.append(full[i][reactionnumber])
    return ionflows

def allreactions(name, ion, steps):
    ionflows = []
    full = allzones(name,ion)
    for j in range(15):
        templist = []
        for i in range(0,int(name.flowb.shape[0])-2,steps):
            #print(i)
            templist.append(full[i][j])
        ionflows.append(templist)
    return ionflows

def singularplot(name, ion, reactionnumber, steps):
    fig,ax=plt.subplots()
    y = name.ionflows(ion,reactionnumber,steps) 
    x = name.y_m[1:len(name.y_m)-1:steps]
    ax.plot(np.log10(x),(y),marker = '.', markersize = 10)
    ax.set_title(f'Flow of {ion} in the {reactiondict[reactionnumber]} reaction vs column depth')
    ax.set_xlabel('log10 of column depth, (g/cm^2)')
    ax.set_ylabel('Flow (mol/g/s)')
    fig.savefig(save_results_to + f'{ion}_single_{reactionnumber}_{steps}.pdf', dpi = 300)
    
def findmax(self, ion, steps):
    all_y = allreactions(self, ion, steps)
    all_x = self.y_m[1:len(self.y_m)-1:steps]
    #print(all_x)
    maxval = []
    coldepth = []
    for i in range(6):
        rnum = 2*i
        r_y = all_y[rnum]
        max_y = np.amax(r_y)
        locations = np.where(r_y == max_y)
        firstposition = locations[0][0]
        x_val = all_x[firstposition]
        coldepth.append(x_val)
        maxval.append(max_y)
    return maxval, coldepth

def reactioneffect(reactionnumber):
    if reactionnumber == 0:
        return 1, 0
    if reactionnumber == 2:
        return -1, 1
    if reactionnumber == 4:
        return 0, 1
    if reactionnumber == 6:
        return 2, 1
    if reactionnumber == 8:
        return 1, 2
    if reactionnumber == 10:
        return 2, 2 

class SummerShot(Shot):
    def makearray(self, zone):
        return makearray(self, zone)
    def printions(self, zone):
        finallist, ionlist = self.makearray(zone)
        return ionlist
    def ionflows(self, ion, reactionnumber, steps):
        return onereaction(self, ion, reactionnumber, steps)
    def ionflows2(self, ion, r1, r2, steps):
        ionflows1 = []
        ionflows2 = []
        print((self.flowb.shape[0]))
        for i in range(0,int(self.flowb.shape[0])-2,steps):
            print(i)
            fulllist = (allzones(self,ion)[i])
            ionflows1.append(fulllist[r1])
            ionflows2.append(fulllist[r2])
        return ionflows1, ionflows2
    def all_y(name, ion):
        return allzones(name, ion)
    def singleplot(self, ion, reactionnumber, steps):
        return singularplot(self, ion, reactionnumber, steps)
    def roptions():
        for i in range(15):
            print(f'{i}:', reactiondict[i])
    def rname(reaction):
        return reactiondict[reaction]
    def findelement(self, element, zone):
        listname = ionlist(self,zone)
        for i in range(len(listname)):
            if str(listname[i]==element) == 'True':
                return i
        else:
            #print(f"{element} is not in list, set -1")
            i = -1
            return i
    def all_r(self, ion, steps):
        return allreactions(self,ion,steps)
    def allplots(self, ion, steps):
        fig,ax=plt.subplots(2,3, figsize=(9,6))
        rnum = -2
        all_y = allreactions(self, ion, steps)
        for i in range(2):
            for j in range(3):
                rnum = rnum + 2
                y = all_y[rnum]
                x = self.y_m[1:len(self.y_m)-1:steps]
                ax[i,j].plot(np.log10(x),(y),marker = '.', markersize = 10, label = f'{reactiondict}[rnum]')
                ax[i,j].set_title(f'{reactiondict[rnum]}')
                #ax[i,j].legend(loc ="lower left");
        fig.suptitle(f'Graphs of {ion} flows vs column depth')
        fig.supxlabel('log10 of column depth, (g/cm^2)')
        fig.supylabel('Flow (mol/g/s)')
        fig.savefig(save_results_to + f'{ion}_run_{steps}.pdf', dpi = 300)
        fig.show()
    def maximumval(self, ion, steps):
        return findmax(self, ion, steps)
    def max_plots_q(ion, steps, start, change, runs):
        fig,ax=plt.subplots(2,3, figsize=(13,12))
        x_change = []
        all_reactions = [[],[],[],[],[],[]]
        all_depths = [[],[],[],[],[],[]]
        for i in range(runs):
            x_change.append(start+change*i)
            run = SummerShot(Q=start+change*i, mdot = 1, abu=dict(h1=0.9, he4=0.1), ymax=12, flowb=True, kaptab=4, M=1.4)
            maxval, coldepth = findmax(run, ion, steps)
            for j in range(6):
                all_reactions[j].append(maxval[j])
                all_depths[j].append(coldepth[j])
        rnum = -2
        rnumy = -1
        #print(np.log10(all_depths))
        #vmin = np.min([np.min(data), np.min(data1)])
        #vmax = np.max([np.max(data), np.max(data1)])
        for i in range(2):
            for j in range(3):
                rnum = rnum + 2
                rnumy = rnumy+1
                y = all_reactions[rnumy]
                x = x_change
                cont = np.log10(all_depths[rnumy])
                ax[i,j].plot(x,y,marker=".",zorder=1, color = "black")
                a0 = ax[i,j].scatter((x),(y),c=cont, cmap = 'magma', vmin=3, vmax=12)
                ax[i,j].set_xlim(0.8*min(x),1.2*max(x))
                ax[i,j].set_ylim(0,1.05*max(y))
                #cbar = fig.colorbar(a0)
                ax[i,j].set_aspect('auto')
                ax[i,j].set_title(f'{reactiondict[rnum]}')
        fig.suptitle(f'Graphs of {ion} flow maximum values vs change in Q and column depth')
        fig.supxlabel('Q value')
        plt.colorbar(a0, ax=ax.ravel().tolist(), shrink=0.9)
        fig.supylabel('Flow (mol/g/s), Colourbars-> log10 of column depth, (g/cm^2)')
        fig.savefig(save_results_to + f'{ion}_qchange_{start}_{runs}.pdf', dpi = 300)
        fig.show()
        
                   
                
    def plot_one_flow(self, zone, reaction, j=None, y=None, ax=None, fig=None, name=None, loc='best', **kwargs):
            #reaction = input('Reaction name:')
            reactionnumber = reactiondict[str(reaction)]
            nchange, pchange = reactioneffect(reactionnumber)
            ax, fig = fig_setup(self, ax, fig, name=name)
            ax.set_aspect('equal')

            if y is not None:
                assert j is None
                j = len(self.y) - 1 - np.searchsorted(self.y[::-1], y)
                if j < 1:
                    j = 1

            if j is None:
                j = 1
            if j == 0:
                abu = self.abub_acc
            elif isinstance(self.abub, AbuSet):
                assert j == 1, f'invalid zone number {j}'
                abu = self.abub
            else:
                abu = self.abub.zone_abu(j)

            if 'lim' in kwargs:
                kwargs.setdefault('log_abu_min', np.log10(kwargs.pop('lim')))

            if not 'cm' in kwargs:
                kwargs['cm'] = color.ColorBlendBspline(
                    ('white',)+tuple(
                        [color.ColorScale(
                            color.colormap('plasma_r'),
                            lambda x: (x-0.2)/0.8)]*2)
                    + ('black',), frac=(0,.2,1),k=1)
            self.nuc =  NucPlot(abu, ax=ax, fig=fig, **kwargs)


            y = 2 / (3 * self.xkn[self.jm+1])
            if j > 0:
                y += 0.5 * (self.y[j] + self.y[j-1])

            fig.text(0.01, 0.99, fr'log $y$ = {np.log10(self.y_m[zone]):7.3f}', ha='left', va='top')

            rflowpos, rflowneg, ionpos, ionneg, ionlist, rflow = rflowposneg(self, zone, reactionnumber)
            widths = scale_list(rflow,0.5,1.5)
            opac = scale_list(rflow,0.15,0.999)
            maxposflow = np.max(rflowpos)
            maxposion = ionpos[np.argmax(rflowpos)]
            maxnegflow = np.min(rflowneg)
            maxnegion = ionneg[np.argmin(rflowneg)]
            #print(rflow,widths)
            ax = self.nuc.ax
            #ax.text(0.85, 0.04, f'pos: {maxposflow}, {maxposion}',
     #horizontalalignment='center',
     #verticalalignment='center',
     #transform = ax.transAxes, color = 'green', backgroundcolor='white', fontsize='small')
            #ax.text(0.85, 0.10, f'neg: {maxnegflow}, {maxnegion}',
     #horizontalalignment='center',
     #verticalalignment='center',
     #transform = ax.transAxes, color = 'red', backgroundcolor='white', fontsize='small')
            #ax.text(0.85, 0.16, f'Max flows (mol/g/s):',
     #horizontalalignment='center',
     #verticalalignment='center',
     #transform = ax.transAxes, color = 'black', backgroundcolor='white', fontsize='small')
            ax.text(0.10, 0.95, f'{reactiondict[reactionnumber]}',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax.transAxes, backgroundcolor='white')
            num=-1
            for ion in ionlist:
                num = num+1
                p = I(ion).Z
                n = I(ion).A-p
                width = widths[num]
                opacity = opac[num]
                if ion in ionpos:
                    ax.annotate("", xy=(n+nchange, p+pchange), xycoords='data', xytext=(n, p), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3",lw=width, alpha=opacity, color='green'),)
                elif ion in ionneg:
                    ax.annotate("", xy=(n, p), xycoords='data', xytext=(n+nchange, p+pchange), textcoords='data', arrowprops=dict(arrowstyle="->", connectionstyle="arc3",lw=width, alpha=opacity, color='red'),) 
            if self.tn[0] == self.tn[-1]:
                ax.plot([None], color='#ffffff00', label=f'${temperature2human(self.tn[0], latex=True)}$')
                ax.plot([None], color='#ffffff00', label=f'${density2human(self.dn[0], latex=True)}$')
                leg = ax.legend(loc='lower right', handlelength=0)
                leg.set_draggable(True)
            red_patch = mpatches.Patch(color='red', label='Negative flows')
            green_patch = mpatches.Patch(color='green', label='Positive flows')
            ax.legend(handles=[red_patch, green_patch],loc='lower right',
            borderaxespad=0, frameon=True, fontsize = 'small', ncol=2, framealpha =1, edgecolor='white', borderpad=0.2, bbox_to_anchor=(0.97, 0.03)) 
            fig.show()    
            #savestatus = input("save?(y/n): ")
            #if savestatus=='y':
                #title=input('title?: ')
                #fig.savefig(f'Results/ionflows/{title}.pdf')
    def allplotsionlist(self, ionlist, steps):
        fig,ax=plt.subplots(2,3, figsize=(9,6))
        powerlist = np.zeros(6)
        y_list = []
        for ion in ionlist:
            templist = []
            all_y = allreactions(self, ion, steps)
            rnum = -2
            for count in range(6):
                rnum = rnum + 2
                y = all_y[rnum]
                templist.append(y)
                absolute_val = np.abs(y).max()
                if absolute_val != 0:
                    power = np.log10(absolute_val)
                    if abs(power) > powerlist[count]:
                        powerlist[count] = (math.trunc(power))
            y_list.append(templist)   
        index = -1
        print(powerlist[0],powerlist)
        for ion in ionlist:
            index = index + 1
            reactioncount = -1
            rnum = -2
            for i in range(2):
                for j in range(3):
                    reactioncount = reactioncount + 1
                    rnum = rnum + 2
                    y = y_list[index][reactioncount]
                    y = np.array(y)
                    y = y/(10**powerlist[reactioncount])
                    x = self.y_m[1:len(self.y_m)-1:steps]
                    ax[i,j].plot((x),(y), label=f'{ion}', alpha=0.85)
                    #ax[i,j].set_title(f'{reactiondict[rnum]}')
                    ax[i,j].set_xscale('log')
                    ax[i,j].text(0.20, 0.5, f'{reactiondict[rnum]}',
     horizontalalignment='center',
     verticalalignment='center',
     transform = ax[i,j].transAxes, color = 'black', backgroundcolor='white', fontsize='small')
                    ax[i,j].set_ylabel(f'Flow  ($10{get_super(str(powerlist[reactioncount]))}$ $mol$ $g{get_super(str(-1))}$ $s{get_super(str(-1))}$)', fontsize='small')
                    #ax[i,j].legend(loc ="lower left");
        #fig.suptitle(f'Graphs of {ion} flows vs column depth')
        ax[0,0].legend(loc='upper left',
            borderaxespad=0, frameon=False, fontsize = 'x-small', ncol=1)
        fig.supxlabel(f'Column depth, $(g/cm^2)$')
        #fig.supylabel(f'Flow $(mol/g/s)$')
        fig.tight_layout()
        fig.show()
        #savestatus = input("save?(y/n): ")
        #if savestatus=='y':
            #title=input('title?: ')
            #fig.savefig(f'Results/all_reactions/{title}.pdf')
        #fig.show()
    def printmaxvals(self, zone, reaction):
        reactionnumber = reactiondict[str(reaction)]
        rflowpos, rflowneg, ionpos, ionneg, ionlist, rflow = rflowposneg(self, zone, reactionnumber)
        maxposflow = np.max(rflowpos)
        maxposion = ionpos[np.argmax(rflowpos)]
        maxnegflow = np.min(rflowneg)
        maxnegion = ionneg[np.argmin(rflowneg)]
        print(f'Maximum negative flow: {maxnegflow} mol/g/s, {maxnegion}')
        print(f'Maximum positive flow: {maxposflow} mol/g/s, {maxposion}')


