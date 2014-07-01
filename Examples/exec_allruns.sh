
# Script for running all comparisons in a simulation set
# Note that cross-comparisons between faces are not included.


# For full factorial
for j in {0..4} # Loop through Fiducials
do
    for face in {0..2} # Loop through faces
    do
        python output_single_timestep.py Fiducial${j}.${face}.0 $face fiducial${j}.${face} T 7
        cd /srv/astro/erickoch/Dropbox/code_development/TurbuStat/Examples
    done
done

## Fiducial Comparisons
for face in {0..2}
do
    python output_single_timestep.py fid_comp $face fiducial_comparisons_face${face} T 7 F
    cd /srv/astro/erickoch/Dropbox/code_development/TurbuStat/Examples
done

# For fractional factorial
# for j in {1..6}
# do
#     for face in {0,2}
#     do
#         python output.py Fiducial128_${j}.${face}.0 $face fiducial${j}.${face} T 10
#         cd /srv/astro/erickoch/Dropbox/code_development/TurbuStat/Examples
#     done
# done

# ## Fiducial Comparisons
# for face in {0,2}
# do
#     python output.py fid_comp $face fiducial_comparisons_face${face} T 10 F
#     cd /srv/astro/erickoch/Dropbox/code_development/TurbuStat/Examples
# done