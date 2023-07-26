import os
import inquirer
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider, Button

from funcs import *

folder = '/ICRA_EXPORT'
w_path = '../../Users/FILLIUNG Martin/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
path = os.path.abspath(w_path)
directory = sorted(os.listdir(path))
directory = [file for file in directory if '.csv' in file]
N_files = len(directory)

questions = [
  inquirer.List('file',
                message="What file veux-tu play ?",
                choices=directory,
            ),
]
answers = inquirer.prompt(questions)
file = answers['file']
full_path = path + "/" + file
data = pd.read_csv(full_path)
N = data['Time'].shape[0]
n_points = 16 if not 'dynamique7' in file else 13


f = plt.figure()
plt.subplots_adjust(hspace=0.)
a = plt.subplot2grid((4, 2), (0, 0), 4, 1, f, projection='3d')
b = plt.subplot2grid((4, 2), (0, 1), 1, 1, f)
c = plt.subplot2grid((4, 2), (1, 1), 1, 1, f)
d = plt.subplot2grid((4, 2), (2, 1), 1, 1, f)
e = plt.subplot2grid((4, 2), (3, 1), 1, 1, f)

axfreq = f.add_axes([0.25, 0.025, 0.65, 0.04])
time_slider = Slider(
    ax=axfreq,
    label='',
    valmin=0,
    valinit=0,
    valstep=1,
    valmax=N-1
)


def update(i):
    a.cla()
    b.cla()
    c.cla()
    d.cla()
    e.cla()
    Xm, Ym, Zm = get_vcable_points_at_index(data, i, n_points)
    Xvc, Yvc, Zvc = get_vcat_points_at_index(data, i, n_points)
    Xtc, Ytc, Ztc = get_tcat_points_at_index(data, i, n_points)
    Dxvc, Dyvc, Dzvc, Dvc = get_dist_to_vcat_at_index(data, i, n_points)
    Dxtc, Dytc, Dztc, Dtc = get_dist_to_tcat_at_index(data, i, n_points)
    a.scatter(Xm, Ym, Zm, c='k', marker='d')
    a.plot(Xvc, Yvc, Zvc, '-o', color='b')
    a.plot(Xtc, Ytc, Ztc, '-o', color='r')
    b.bar([i - .25 for i in range(len(Dxvc))], Dxvc, width=.5, color='b')
    c.bar([i - .25 for i in range(len(Dyvc))], Dyvc, width=.5, color='b')
    d.bar([i - .25 for i in range(len(Dzvc))], Dzvc, width=.5, color='b')
    e.bar([i - .25 for i in range(len(Dvc))], Dzvc, width=.5, color='b')
    b.bar([i + .25 for i in range(len(Dxtc))], Dxtc, width=.5, color='r')
    c.bar([i + .25 for i in range(len(Dytc))], Dytc, width=.5, color='r')
    d.bar([i + .25 for i in range(len(Dztc))], Dztc, width=.5, color='r')
    e.bar([i + .25 for i in range(len(Dtc))], Dztc, width=.5, color='r')
    a.set_title(f'{file}\nTheta = {data["Theta"][i]:.2f}; Gamma= {data["Gamma"][i]:.2f}')
    a.legend(('marqueurs', 'chainette verticale', 'chainette inclinée'))
    a.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")
    a.set_zlabel("z (mm)")
    b.set_ylabel('écart sur X (mm)')
    c.set_ylabel('écart sur Y (mm)')
    d.set_ylabel('écart sur Z (mm)')
    e.set_ylabel('norme de l`écart (mm)')
    d.set_xlabel('marqueur')
    a.axis('equal')
    plt.pause(.0001)


time_slider.on_changed(update)

playAx = f.add_axes([0.1, 0.025, 0.1, 0.04])
playBtn = Button(playAx, 'Play', hovercolor='0.975')


def play(event):
    for i in range(0, N):
        time_slider.set_val(i)


playBtn.on_clicked(play)

plt.show()


