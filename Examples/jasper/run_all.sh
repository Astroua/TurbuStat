
# Submit jobs to run all comparisons

# Noiseless
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial0_all.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial1_all.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial2_all.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial3_all.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial4_all.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial_comp.pbs


# Noisy
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial0_all_noise.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial1_all_noise.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial2_all_noise.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial3_all_noise.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial4_all_noise.pbs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/fiducial_comp_noise.pbs

# Obs to Obs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/complete_to_complete.pbs

# Des to Obs
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/des_to_complete.pbs

# Obs to Fid
qsub /home/ekoch/code_repos/TurbuStat/Examples/jasper/complete_to_fid.pbs

