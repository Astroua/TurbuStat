
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
                 params=["fc", "pb", "m", "k", "sf", "vp"], save=False):
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
    distances = distances.T.drop(params + ["Cube"]).T

    # Get the model effects from the index
    model_effects = effects.index

    # Replace the shorthand names
    rep_name = {'fc': "Face", "pb": "Plasma Beta", 'm': "Mach",
                'k': "Driving Scale", 'sf': "Solenoidal Fraction",
                'vp': "Virial Parameter"}

    # Now loop through the statistics in each file
    for stat in distances.columns:
        try:
            response = effects[stat].abs()
        except KeyError:
            warnings.warn("%s effects not found. Skipping" % (stat))
            continue

        # Ignore higher than 2nd order effects
        response = response[:21]

        # Find the most important effects 2nd order effects
        # Values are ~ z-scores. Significant effects have an absolute value
        # greater than 2.
        imp_inters = response[6:].order(ascending=False)[:3]

        # Create colours based on the absolute values of the responses
        # cool = p.get_cmap('cool')
        milagro = \
            colormap_milagro(np.log10(response.min()),
                             np.log10(response.max()),
                             np.log10(min_zscore))
        cNorm = cols.Normalize(vmin=np.log10(response.min()),
                               vmax=np.log10(response.max()))
        scalMap = cm.ScalarMappable(norm=cNorm, cmap=milagro)

        # Create plots for the main effects, regardless of importance
        fig, axes = p.subplots(3, 3, sharex=True)
        for i, (param, ax) in enumerate(zip(params+list(imp_inters.index), axes.flatten())):
            if i < 6:
                low_data = distances[stat][design[param] == -1]
                high_data = distances[stat][design[param] == 1]
                ax.plot([-1, 1], [low_data.mean(), high_data.mean()],
                        marker="D",
                        color=scalMap.to_rgba(np.log10(response[param])),
                        lw=0)

                # Plot the slope
                ax.plot([-1, 1], [low_data.mean(), high_data.mean()],
                        color=scalMap.to_rgba(np.log10(response[param])),
                        label=rep_name[param])

                ax.set_title(param)

            else:
                param1 = param.split(":")[0]
                param2 = param.split(":")[-1]

                low_low = distances[stat][np.logical_and(design[param1] == -1, design[param2] == -1)]
                low_high = distances[stat][np.logical_and(design[param1] == 1, design[param2] == -1)]
                high_low = distances[stat][np.logical_and(design[param1] == -1, design[param2] == 1)]
                high_high = distances[stat][np.logical_and(design[param1] == 1, design[param2] == 1)]

                ax.plot([-1, 1], [low_low.mean(), low_high.mean()], marker="D",
                        color=scalMap.to_rgba(np.log10(imp_inters[param])))
                ax.plot([-1, 1],
                        [high_low.mean(), high_high.mean()], marker="s",
                        color=scalMap.to_rgba(np.log10(imp_inters[param])))

                ax.plot([-1, 1], [low_low.mean(), low_high.mean()],
                        color=scalMap.to_rgba(np.log10(imp_inters[param])))
                ax.plot([-1, 1], [high_low.mean(), high_high.mean()],
                        color=scalMap.to_rgba(np.log10(imp_inters[param])))

                ax.set_title(param)

            ax.set_xlim([-2, 2])
            ax.set_ylim([distances[stat].min(), distances[stat].max()])

            # Y-ticks at row beginnings
            if i == 0 or i == 3 or i == 6:
                pass
            else:
                ax.set_yticklabels([])
            # X-ticks on bottom row
            if i < 3:
                ax.set_xticks([])
            else:
                ax.set_xticks([-1, 1])

        # Loop through the important effects
        # for posn in enumerate(imp_effect):
        #     index = model_effects[posn]

        #     # Check if its an interaction term
        #     if not ":" in index and posn < len(params):
        #         continue
        #     int_params = [s for s in rep_name.keys()
        #                   if s in index.split(":")]
        #     index = ":".join(int_params)

        #     for param in int_params:
        #         # Get the right axis to plot on
        #         ax = axes.flatten()[params.index(param)]

        fig.subplots_adjust(right=0.85)
        cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=milagro, norm=cNorm)
        cb.set_ticks(np.log10(response))
        cb.set_ticklabels(model_effects)
        cb.ax.tick_params(labelsize=10)

        if save:
            fig.savefig("full_factorial_"+stat+"_modeleffects.pdf")
        else:
            fig.show()
            p.show()


def colormap_milagro(vmin, vmax, vtransition, width=0.0001, huestart=0.6):

    """
    Gammapy License:

    Copyright (c) 2014, Gammapy Developers All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

        Redistributions of source code must retain the above copyright notice,
        this list of conditions and the following disclaimer.

        Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

        Neither the name of the Astropy Team nor the names of its contributors
        may be used to endorse or promote products derived from this software
        without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
    TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
    PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
    LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
    NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    --------------------------------------------------------------------------

    Colormap often used in Milagro collaboration publications.

    This colormap is gray below ``vtransition`` and similar to the jet
    colormap above.

    A sharp gray -> color transition is often used for significance images
    with a transition value of ``vtransition ~ 5`` or ``vtransition ~ 7``,
    so that the following effect is achieved:

    - gray: non-significant features are not well visible
    - color: significant features at the detection threshold ``vmid``

    Note that this colormap is often critizised for over-exaggerating small
    differences in significance below and above the gray - color transition
    threshold.

    Parameters
    ----------
    vmin : float
        Minimum value (color: black)
    vmax : float
        Maximum value
    vtransition : float
        Transition value (below: gray, above: color).
    width : float
        Width of the transition
    huestart : float
        Hue of the color at ``vtransition``

    Returns
    -------
    colormap : `matplotlib.colors.LinearSegmentedColormap`
        Colormap

    Examples
    --------
    >>> from gammapy.image import colormap_milagro
    >>> vmin, vmax, vtransition = -5, 15, 5
    >>> cmap = colormap_milagro(vmin=vmin, vmax=vmax, vtransition=vtransition)

    """

    from colorsys import hls_to_rgb

    from matplotlib.colors import LinearSegmentedColormap

    # Compute normalised red, blue, yellow values
    transition = float(vtransition - vmin) / (vmax - vmin)

    # Create custom colormap
    # List entries: (value, (H, L, S))
    colors = [(0, (0.5, 0, 0)),
              (transition - width, (0.5, 0, 0)),
              (transition, (huestart, 0.4, 0.5)),
              (transition + width, (huestart, 0.4, 1)),
              # (0.99, (0, 0.6, 1)),
              (1, (0, 0.5, 0.5)),
              ]
    # Convert HLS values to RGB values
    rgb_colors = [(val, hls_to_rgb(*hls)) for (val, hls) in colors]
    cmap = LinearSegmentedColormap.from_list(name='milagro', colors=rgb_colors)

    return cmap
