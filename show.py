import os

import inquirer
from matplotlib.widgets import Button, Slider
from numpy import nanmean

from funcs import *

folder = '/ICRA_EXPORT'
w_path = (
		'../../Users/FILLIUNG Martin/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder)
# w_path = '../../Users/marti/OneDrive - Université de Toulon/Thèse/CEPHISMER-11-2022_POST_TRAITEMENT' + folder
path = os.path.abspath( w_path )
directory = sorted( os.listdir( path ) )
directory = [ file for file in directory if '.csv' in file ]
N_files = len( directory )

questions = [ inquirer.List(
		'file', message = "What file veux-tu play ?", choices = directory, ), ]
file = directory[ 0 ]
answers = inquirer.prompt( questions )
file = answers[ 'file' ]
full_path = path + "/" + file
data = pd.read_csv( full_path )
N = data[ 'Time' ].shape[ 0 ]
n_points = 16 if not 'dynamique7' in file else 13

f = plt.figure()
plt.subplots_adjust( hspace = 0. )
a = plt.subplot2grid( (4, 2), (0, 0), 4, 1, f, projection = '3d' )
b = plt.subplot2grid( (4, 2), (0, 1), 1, 1, f )
c = plt.subplot2grid( (4, 2), (1, 1), 1, 1, f )
d = plt.subplot2grid( (4, 2), (2, 1), 1, 1, f )
e = plt.subplot2grid( (4, 2), (3, 1), 1, 1, f )

axfreq = f.add_axes( [ 0.25, 0.025, 0.65, 0.04 ] )
time_slider = Slider(
		ax = axfreq, label = '', valmin = 0, valinit = 0, valstep = 1, valmax = N - 1
		)


def update( i ):
	a.cla()
	b.cla()
	c.cla()
	d.cla()
	e.cla()
	Xm, Ym, Zm = get_named_points_at_index( 'cable_cor', data, i, n_points )
	Xvc, Yvc, Zvc = get_named_points_at_index( 'vcat', data, i, n_points )
	Xtc, Ytc, Ztc = get_named_points_at_index( 't_acat', data, i, n_points )
	Xgc, Ygc, Zgc = get_named_points_at_index( 'g_acat', data, i, n_points )
	Xtgc, Ytgc, Ztgc = get_named_points_at_index( 'tg_acat', data, i, n_points )
	Dxvc, Dyvc, Dzvc, Dvc = get_named_dists_at_index( 'vcat', data, i, n_points )
	Dxtc, Dytc, Dztc, Dtc = get_named_dists_at_index( 't_acat', data, i, n_points )
	Dxgc, Dygc, Dzgc, Dgc = get_named_dists_at_index( 'g_acat', data, i, n_points )
	Dxtgc, Dytgc, Dztgc, Dtgc = get_named_dists_at_index( 'tg_acat', data, i, n_points )
	a.scatter( Xm, Ym, Zm, c = 'k', marker = 'd' )
	a.plot( Xvc, Yvc, Zvc, '-o', color = 'b' )
	a.plot( Xtc, Ytc, Ztc, '-o', color = 'g' )
	a.plot( Xgc, Ygc, Zgc, '-o', color = 'r' )
	a.plot( Xtgc, Ytgc, Ztgc, '-o', color = 'm' )
	b.bar( [ i - .375 for i in range( len( Dxvc ) ) ], Dxvc, width = .25, color = 'b' )
	c.bar( [ i - .375 for i in range( len( Dyvc ) ) ], Dyvc, width = .25, color = 'b' )
	d.bar( [ i - .375 for i in range( len( Dzvc ) ) ], Dzvc, width = .25, color = 'b' )
	e.bar( [ i - .375 for i in range( len( Dvc ) ) ], Dvc, width = .25, color = 'b' )
	b.bar( [ i - .125 for i in range( len( Dxtc ) ) ], Dxtc, width = .25, color = 'g' )
	c.bar( [ i - .125 for i in range( len( Dytc ) ) ], Dytc, width = .25, color = 'g' )
	d.bar( [ i - .125 for i in range( len( Dztc ) ) ], Dztc, width = .25, color = 'g' )
	e.bar( [ i - .125 for i in range( len( Dtc ) ) ], Dtc, width = .25, color = 'g' )
	b.bar( [ i + .125 for i in range( len( Dxgc ) ) ], Dxgc, width = .25, color = 'r' )
	c.bar( [ i + .125 for i in range( len( Dygc ) ) ], Dygc, width = .25, color = 'r' )
	d.bar( [ i + .125 for i in range( len( Dzgc ) ) ], Dzgc, width = .25, color = 'r' )
	e.bar( [ i + .125 for i in range( len( Dgc ) ) ], Dgc, width = .25, color = 'r' )
	b.bar( [ i + .375 for i in range( len( Dxtgc ) ) ], Dxtgc, width = .25, color = 'm' )
	c.bar( [ i + .375 for i in range( len( Dytgc ) ) ], Dytgc, width = .25, color = 'm' )
	d.bar( [ i + .375 for i in range( len( Dztgc ) ) ], Dztgc, width = .25, color = 'm' )
	e.bar( [ i + .375 for i in range( len( Dtgc ) ) ], Dtgc, width = .25, color = 'm' )
	for j in range( 1, n_points ):
		b.axvline( j - .5, color = 'k', linestyle = ':' )
		c.axvline( j - .5, color = 'k', linestyle = ':' )
		d.axvline( j - .5, color = 'k', linestyle = ':' )
		e.axvline( j - .5, color = 'k', linestyle = ':' )
	e.axhline( nanmean( Dvc ), color = 'b' )
	e.axhline( nanmean( Dgc ), color = 'r' )
	e.axhline( nanmean( Dtc ), color = 'g' )
	e.axhline( nanmean( Dtgc ), color = 'm' )
	a.set_title(
			f'{file} '
			f'(vcat: {data[ "vcat_0 X" ].count() / data.shape[ 0 ]:.2f}; '
			f't_acat: {data[ "t_acat_0 X" ].count() / data.shape[ 0 ]:.2f}; '
			f'g_acat: {data[ "g_acat_0 X" ].count() / data.shape[ 0 ]:.2f}; '
			f'tg_acat: {data[ "tg_acat_0 X" ].count() / data.shape[ 0 ]:.2f})\n'
			f'Theta = {data[ "Theta" ][ i ]:.2f}; '
			f'Gamma= {data[ "Gamma" ][ i ]:.2f}'
			)
	a.legend(
			('marqueurs', 'chainette standard', 'chainette theta-augmentée', 'chainette gamma-augmentée',
			 'chainette theta-gamma-augmentée')
			)
	a.set_xlabel( "x (mm)" )
	a.set_ylabel( "y (mm)" )
	a.set_zlabel( "z (mm)" )
	b.set_ylabel( 'écart sur X (mm)' )
	c.set_ylabel( 'écart sur Y (mm)' )
	d.set_ylabel( 'écart sur Z (mm)' )
	e.set_ylabel( 'norme de l`écart (mm)' )
	d.set_xlabel( 'marqueur' )
	a.axis( 'equal' )
	e.set_xticks( [ i for i in range( n_points ) ] )
	plt.pause( .0001 )


time_slider.on_changed( update )

playAx = f.add_axes( [ 0.1, 0.025, 0.1, 0.04 ] )
playBtn = Button( playAx, 'Play', hovercolor = '0.975' )


def play( event ):
	for i in range( 0, N ):
		time_slider.set_val( i )


playBtn.on_clicked( play )

plt.show()
