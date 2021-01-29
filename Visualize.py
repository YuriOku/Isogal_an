# %%
import yt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

time = 250

models = [
    ["ss_model/ver01193","Sphericalsuperbubble"],
    ["Osaka19", "Osaka2019"],
    ["NoFB2", "Nofeedback"],
          ]
unit_base = {'UnitLength_in_cm'         : 3.08568e+21,
             'UnitMass_in_g'            :   1.989e+43,
             'UnitVelocity_in_cm_per_s' :        1e+5,
             'metallicity'              :   (1/0.0134, 'Zsun')
             }

bbox_lim = 1e4 #kpc

bbox = [[-bbox_lim,bbox_lim],
        [-bbox_lim,bbox_lim],
        [-bbox_lim,bbox_lim]]
# %%
ds = [0]*len(models)
center = [0]*len(models)
for i in range(len(models)):
    ds[i] = yt.load('{0}/snapshot_{1:03}/snapshot_{1:03}.hdf5'.format(models[i][0], time),unit_base=unit_base,bounding_box=bbox)
    ad = ds[0].all_data()
    density = ad[("PartType0","density")]
    wdens = np.where(density == np.max(density))
    coordinates = ad[("PartType0","Coordinates")]
    center[i] = coordinates[wdens][0]
# print ('center = ',center)

# %% density
# field = "density"
# for i in range(len(models)):
#   px = yt.OffAxisProjectionPlot(ds[i], [1,0,0], ('gas', '{}'.format(field)), center[i], width=45, north_vector=[0,0,1])
#   px.set_xlabel("x")
#   px.set_ylabel("y")
#   # px = yt.ProjectionPlot(ds[i], 'z', ('gas', '{}'.format(field)), center = center[i], width=45)
#   px.set_font({'size':32})
#   # px.set_zlim(field="{}".format(field), zmin=5e-6, zmax=1e-1)
#   # px.hide_colorbar()
#   # px.hide_axes(field='x')
#   px.show()
#   # px.save("results/density-{}.pdf".format(models[i][1]))
# # %% metallicity
# field = "density"
# for i in range(len(models)):
#   px = yt.SlicePlot(ds[i], 'x', ('gas', '{}'.format(field)), center = center[i], width=400)
#   px.set_cmap(field="{}".format(field), cmap='hot')
#   # px.set_zlim(field="{}".format(field), zmin=1e-5, zmax=1e-1)
#   px.set_font({'size':32})
#   px.show()

# %%
sorted(ds[0].field_list)
# %% density
# ----------
field = "density"
fig = plt.figure()
grid = AxesGrid(fig, (0.075,0.075,0.85,0.85),
                nrows_ncols = (2, 3),
                axes_pad = 0.05,
                label_mode = "L",
                share_all = True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad="1%")

for i in range(len(models)):
  px = yt.OffAxisProjectionPlot(ds[i], [0,0,1], ('gas', '{}'.format(field)), center = center[i], width=45, north_vector=[0,1,0])
  px.set_xlabel("x [kpc]")
  px.set_ylabel("y [kpc]")
  px.set_font({'size':20})
  px.set_zlim(field="{}".format(field), zmin=5e-6, zmax=1e-1)
  plot = px.plots['{}'.format(field)]
  plot.figure = fig
  plot.axes = grid[i].axes
  plot.cax = grid.cbar_axes[i]
  px.set_background_color(field)
  px._setup_plots()
for i in range(len(models)):
  px = yt.OffAxisProjectionPlot(ds[i], [0,1,0], ('gas', '{}'.format(field)), center = center[i], width=45, north_vector=[0,0,1])
  px.set_xlabel("x [kpc]")
  px.set_ylabel("z [kpc]")
  px.set_font({'size':20})
  px.set_zlim(field="{}".format(field), zmin=5e-6, zmax=1e-1)
  plot = px.plots['{}'.format(field)]
  plot.figure = fig
  plot.axes = grid[i+len(models)].axes
  plot.cax = grid.cbar_axes[i+len(models)]
  px.set_background_color(field)
  px._setup_plots()
plt.savefig("results/plot{}.pdf".format(field), bbox_inches="tight")

# %% temperature 20kpc
# ---------------------
field = "temperature"
fig = plt.figure()
grid = AxesGrid(fig, (0.075,0.075,0.85,0.85),
                nrows_ncols = (2, 3),
                axes_pad = 0.05,
                label_mode = "L",
                share_all = True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad="1%")

for i in range(len(models)):
  px = yt.OffAxisProjectionPlot(ds[i], [0,0,1], ('gas', '{}'.format(field)), center = center[i], width=45, north_vector=[0,1,0], weight_field='density')
  px.set_xlabel("x [kpc]")
  px.set_ylabel("y [kpc]")
  px.set_font({'size':20})
  px.set_zlim(field="{}".format(field), zmin=1e3, zmax=1e7)
  px.set_cmap(field="{}".format(field), cmap='hot')
  plot = px.plots['{}'.format(field)]
  plot.figure = fig
  plot.axes = grid[i].axes
  plot.cax = grid.cbar_axes[i]
  px.set_background_color(field)
  px._setup_plots()
for i in range(len(models)):
  px = yt.OffAxisProjectionPlot(ds[i], [0,1,0], ('gas', '{}'.format(field)), center = center[i], width=45, north_vector=[0,0,1], weight_field='density')
  px.set_xlabel("x [kpc]")
  px.set_ylabel("z [kpc]")
  px.set_font({'size':20})
  px.set_zlim(field="{}".format(field), zmin=1e3, zmax=1e7)
  px.set_cmap(field="{}".format(field), cmap='hot')
  plot = px.plots['{}'.format(field)]
  plot.figure = fig
  plot.axes = grid[i+len(models)].axes
  plot.cax = grid.cbar_axes[i+len(models)]
  px.set_background_color(field)
  px._setup_plots()
plt.savefig("results/plot{}20kpc.pdf".format(field), bbox_inches="tight")



# %% temperature 200kpc
# ---------------------
field = "temperature"
fig = plt.figure()
grid = AxesGrid(fig, (0.075,0.075,0.85,0.85),
                nrows_ncols = (1, 3),
                axes_pad = 0.05,
                label_mode = "L",
                share_all = True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad="1%")

for i in range(len(models)):
  px = yt.OffAxisProjectionPlot(ds[i], [0,1,0], ('gas', '{}'.format(field)), center = center[i], width=500, north_vector=[0,0,1], weight_field='density')
  px.set_xlabel("x [kpc]")
  px.set_ylabel("z [kpc]")
  px.set_font({'size':20})
  px.set_zlim(field="{}".format(field), zmin=1e3, zmax=1e7)
  px.set_cmap(field="{}".format(field), cmap='hot')
  plot = px.plots['{}'.format(field)]
  plot.figure = fig
  plot.axes = grid[i].axes
  plot.cax = grid.cbar_axes[i]
  px.set_background_color(field)
  px._setup_plots()
plt.savefig("results/plot{}200kpc.pdf".format(field), bbox_inches="tight")



# %% metallicity 200kpc
# ---------------------
field = "metallicity"
fig = plt.figure()
grid = AxesGrid(fig, (0.075,0.075,0.85,0.85),
                nrows_ncols = (1, 3),
                axes_pad = 0.05,
                label_mode = "L",
                share_all = True,
                cbar_location="right",
                cbar_mode="single",
                cbar_size="5%",
                cbar_pad="1%")

for i in range(len(models)):
  px = yt.OffAxisProjectionPlot(ds[i], [0,1,0], ('gas', '{}'.format(field)), center = center[i], width=500, north_vector=[0,0,1], weight_field='density')
  px.set_xlabel("x [kpc]")
  px.set_ylabel("z [kpc]")
  px.set_font({'size':20})
  px.set_zlim(field , zmin=2e-4, zmax=5e-2)
  px.set_cmap(field , cmap='Haze')
  px.set_unit(field, '')
  px.set_background_color('{}'.format(field))
  plot = px.plots['{}'.format(field)]
  plot.figure = fig
  plot.axes = grid[i].axes
  plot.cax = grid.cbar_axes[i]
  px.set_background_color(field)
  px._setup_plots()
plt.savefig("results/plot{}200kpc.pdf".format(field), bbox_inches="tight")
# %%
