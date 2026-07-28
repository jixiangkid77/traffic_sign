# -*- coding: utf-8 -*-
# Graphical Abstract v3 (Access-2026-18185): 660x295, JPG < 45 KB
# All 5 LeNet-5 layers (Table IV), 3-model results, aligned arrows/boxes.
# On Windows change FONT to 'Arial' and rerun.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from PIL import Image
import os

FONT = 'Liberation Sans'
plt.rcParams['font.family'] = FONT
plt.rcParams['mathtext.fontset'] = 'dejavusans'

BLUE='#1F77B4'; RED='#D62728'; GRAY='#909090'
FILLGRAY='#D9D9D9'; LGRAY='#CCCCCC'; DARK='#333333'; LBLUE='#EAF2FA'

W,H = 660,295
fig = plt.figure(figsize=(W/100,H/100), dpi=200)
ax = fig.add_axes([0,0,1,1]); ax.set_xlim(0,W); ax.set_ylim(0,H); ax.axis('off')
def T(y): return H-y
def txt(x,y,s,size=10,color=DARK,bold=False,italic=False,ha='center',va='center'):
    ax.text(x,T(y),s,fontsize=size,color=color,fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',ha=ha,va=va)

# frame + title
ax.add_patch(Rectangle((1.5,1.5),W-3,H-3,fill=False,edgecolor=LGRAY,lw=1))
txt(W/2,16,'Layer-Aware Selective Weight Protection for PCM-Based NN Inference',size=12.5,bold=True)
ax.plot([12,W-12],[T(30),T(30)],color=LGRAY,lw=1)
for xd in (150,455):
    ax.plot([xd,xd],[T(38),T(285)],color=LGRAY,lw=1)

# headers + colored underline (aligned identical y)
def header(cx,label,color):
    txt(cx,49,label,size=11.5,color=color,bold=True)
    ax.plot([cx-20,cx+20],[T(59),T(59)],color=color,lw=2)
header(80,'Problem',RED)
header(302,'Layer-Aware Selective Protection',BLUE)
header(553,'Results & Benefits',BLUE)

# ---------- LEFT (cx=80): problem chain, arrows on exact centerline ----------
CX=80
box=FancyBboxPatch((38,T(101)),84,34,boxstyle='round,pad=1.5',facecolor='#F7F7F7',edgecolor=GRAY,lw=1.1)
ax.add_patch(box)
for r in range(3):                      # mini cell grid = memory array
    for c in range(4):
        ax.add_patch(Rectangle((45+c*9.5,T(75+r*9)),7.5,7,facecolor=FILLGRAY,edgecolor=GRAY,lw=0.4))
txt(103,84,'PCM\nweights',size=8.8,color=DARK)
ax.annotate('',xy=(CX,T(126)),xytext=(CX,T(105)),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.8))
txt(CX,137,'Conductance drift',size=10.2,color=RED,bold=True)
txt(CX,151,r'$G(t)=G_0\,(t/t_0)^{-\nu}$',size=8.5,color=GRAY)
ax.annotate('',xy=(CX,T(181)),xytext=(CX,T(160)),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.8))
txt(CX,192,'Accuracy loss',size=10.2,color=RED,bold=True)
sx=[50,70,90,106]; sy=[208,213,223,234]
ax.plot(sx[:3],[T(v) for v in sy[:3]],color=RED,lw=1.8,solid_capstyle='round')
ax.annotate('',xy=(sx[3],T(sy[3])),xytext=(sx[2],T(sy[2])),arrowprops=dict(arrowstyle='-|>',color=RED,lw=1.8))
txt(116,214,'over\ntime',size=8,color=GRAY,ha='left')

# ---------- MIDDLE: all 5 LeNet-5 layers (Table IV, network order) ----------
bx,bw,bh = 196,114,16
rows=[('conv1',1.000,'100%'),('conv2',0.617,'61.7%'),('fc1',0.007,'0.7%'),
      ('fc2',0.031,'3.1%'),('fc3',0.945,'94.5%')]
ytops=[66,90,114,138,162]
for (nm,fr,lab),yt in zip(rows,ytops):
    txt(192,yt+bh/2,nm,size=9.5,ha='right')
    ax.add_patch(Rectangle((bx,T(yt+bh)),bw,bh,facecolor=FILLGRAY,edgecolor=GRAY,lw=0.8))
    ax.add_patch(Rectangle((bx,T(yt+bh)),max(bw*fr,2.5),bh,facecolor=BLUE,edgecolor='none'))
    txt(bx+bw+5,yt+bh/2,lab,size=9.5,color=BLUE,bold=True,ha='left')
txt(302,192,'Sensitivity-weighted allocation (\u03b2 = 5%, Table IV)',size=8.5,color=GRAY,italic=True)
# compact color key
ax.add_patch(Rectangle((225,T(213)),10,10,facecolor=BLUE)); txt(240,208,'protected',size=8.8,ha='left')
ax.add_patch(Rectangle((318,T(213)),10,10,facecolor=FILLGRAY,edgecolor=GRAY,lw=0.7)); txt(333,208,'unprotected',size=8.8,ha='left')

# destination boxes, arrows horizontally centered on each box
sram=FancyBboxPatch((352,T(110)),96,42,boxstyle='round,pad=1.5',facecolor=LBLUE,edgecolor=BLUE,lw=1.3)
ax.add_patch(sram)
txt(400,81,'Drift-Immune',size=8.8,color=BLUE,bold=True)
txt(400,95,'Storage (SRAM)',size=8.8,color=BLUE,bold=True)
pcm=FancyBboxPatch((352,T(172)),96,42,boxstyle='round,pad=1.5',facecolor='#F7F7F7',edgecolor=GRAY,lw=1.2)
ax.add_patch(pcm)
txt(400,143,'PCM array',size=8.8,color=DARK,bold=True)
txt(400,157,'(majority)',size=8,color=GRAY)
ax.annotate('',xy=(351,T(89)),xytext=(340,T(89)),arrowprops=dict(arrowstyle='-|>',color=BLUE,lw=1.9))
ax.annotate('',xy=(351,T(151)),xytext=(340,T(151)),arrowprops=dict(arrowstyle='-|>',color=GRAY,lw=1.9))

# ---------- RIGHT: three-model results + benefits ----------
def bullet(y): ax.scatter([464],[T(y)],marker='>',s=24,color=BLUE)
bullet(70); txt(474,70,'Near-baseline accuracy',size=9.8,bold=True,ha='left')
txt(474,85,'LeNet-5: 98.56% (\u03b2 = 5%)',size=9.2,ha='left')
txt(474,99,'ResNet-18: 93.38% (\u03b2 = 5%)',size=9.2,ha='left')
txt(474,113,'ResNet-50 (ImageNet):',size=9.2,ha='left')
txt(474,127,'90.2% recovery at \u03b2 = 50%',size=9.2,ha='left')
bullet(159); txt(474,159,'Robust to drift mismatch',size=9.8,bold=True,ha='left')
txt(474,174,'and PVT stress',size=9.8,ha='left')
bullet(206); txt(474,206,'Explicitly quantified',size=9.8,bold=True,ha='left')
txt(474,221,'hardware overhead',size=9.8,ha='left')
txt(474,235,'(Section VI)',size=8.5,color=GRAY,ha='left')

fig.savefig('/home/claude/ga_2x.png',dpi=200); plt.close(fig)
img=Image.open('/home/claude/ga_2x.png').convert('RGB').resize((660,295),Image.LANCZOS)
q=92
while q>=70:
    img.save('/home/claude/Graphical_Abstract.jpg','JPEG',quality=q,optimize=True)
    kb=os.path.getsize('/home/claude/Graphical_Abstract.jpg')/1024
    if kb<45: break
    q-=5
print(f'660x295 JPG, q={q}, {kb:.1f} KB')
