
'''
Read in the effects from the fit and plot the important effects.
'''

import matplotlib as mpl
import matplotlib.pyplot as p
import matplotlib.colors as cols
import matplotlib.cm as cm
import numpy as np
from pandas import read_csv
import warnings


def effect_plots(distance_file, effects_file, min_zscore=2.0,
                 params=["fc", "pb", "m", "k", "sf", "vp"]):
    '''
    Creates a series of plots for the important effects in the model.
    '''

    if isinstance(distance_file, str):
        distances = read_csv(distance_file)
    else:
        distances = distance_file

    if isinstance(effects_file, str):
        effects = read_csv(effects_file)
    else:
        effects = effects_file

    # Extract the design matrix
    design = distances.T.ix[params].T

    # Now drop the design matrix and the Fiducial number
    distances = distances.T.drop(params+["Cube"]).T

    # Get the model effects from the index
    model_effects = effects.index

    # Replace the shorthand names
    rep_name = {'fc': "Face", "pb": "Plasma Beta", 'm': "Mach",
                'k': "Driving Scale", 'sf': "Solenoidal Fraction",
                'vp': "Virial Parameter"}

    # Now loop through the statistics in each file
    for stat in distances.columns[8:10]:
        try:
            response = effects[stat].abs()
        except KeyError:
            warnings.warn("%s effects not found. Skipping" % (stat))
            continue

        # Find the important effects
        # Values are ~ z-scores. Significant effects have an absolute value
        # greater than 2.
        imp_effect = np.where(response.abs() > min_zscore)

        # Create colours based on the absolute values of the responses
        cool = p.get_cmap('cool')
        cNorm = cols.Normalize(vmin=np.log(response.min()),
                               vmax=np.log(response.max()))
        scalMap = cm.ScalarMappable(norm=cNorm, cmap=cool)

        # Create plots for the main effects, regardless of importance
        fig, axes = p.subplots(2, 3, sharex=True)
        for i, (param, ax) in enumerate(zip(params, axes.flatten())):
            low_data = distances[stat][design[param] == -1]
            high_data = distances[stat][design[param] == 1]
            ax.plot([-1] * len(low_data), low_data, "kD", alpha=0.05)
            ax.plot([1] * len(high_data), high_data, "kD", alpha=0.05)

            # Plot the slope
            ax.plot([-1, 1], [low_data.mean(), high_data.mean()],
                    color=scalMap.to_rgba(response[param]),
                    label=rep_name[param])
            ax.set_xlim([-2, 2])

            ax.set_title(rep_name[param])
            # Y-ticks at row beginnings
            if i == 0 or i == 3:
                pass
            else:
                ax.set_yticklabels([])
            # X-ticks on bottom row
            if i < 3:
                ax.set_xticks([])
            else:
                ax.set_xticks([-1, 1])

        # Loop through the important effects
        for posn in enumerate(imp_effect):
            index = model_effects[posn]

            # Check if its an interaction term
            if not ":" in index and posn < len(params):
                continue
            int_params = [s for s in rep_name.keys()
                          if s in index.split(":")]
            index = ":".join(int_params)

            for param in int_params:
                # Get the right axis to plot on
                ax = axes.flatten()[params.index(param)]

        fig.subplots_adjust(right=0.85)
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cool, norm=cNorm)
        cb.set_ticks(np.log(response))
        cb.set_ticklabels(model_effects)
        fig.show()

