
import os
from turbustat.analysis import comparison_plot
import matplotlib.pyplot as p

# Disable interactive mode
p.ioff()

# Creates the distance plots for all

path = "/Users/eric/Dropbox/AstroStatistics/Full Factorial/"
comparisons = ["0_0", "0_2", "2_0", "2_2"]
paper_comparisons = ["0_0", "2_2"]
design_matrix = os.path.join(path, "Design7Matrix.csv")

# Clean results

comparison_plot(
    os.path.join(path, "Full Results"), comparisons=comparisons,
    out_path=os.path.join(path, "Full Results", "Distance Plots"),
    num_fids=5, design_matrix=design_matrix)

comparison_plot(
    os.path.join(path, "Full Results"), comparisons=paper_comparisons,
    out_path=os.path.join(path, "Full Results", "Distance Plots Paper"),
    num_fids=5, design_matrix=design_matrix)

# Noisy results

comparison_plot(
    os.path.join(path, "Noisy Full Results/"), comparisons=comparisons,
    out_path=os.path.join(path, "Noisy Full Results", "Distance Plots"),
    num_fids=5, design_matrix=design_matrix)

comparison_plot(
    os.path.join(path, "Noisy Full Results/"), comparisons=paper_comparisons,
    out_path=os.path.join(path, "Noisy Full Results", "Distance Plots Paper"),
    num_fids=5, design_matrix=design_matrix)

# Obs to Fid

obs_to_fid_path = os.path.join(path, "Observational Results", "Obs_to_Fid")
obs_to_fid_comparisons = ["0_0", "2_2"]

comparison_plot(
    obs_to_fid_path, comparisons=obs_to_fid_comparisons,
    out_path=os.path.join(obs_to_fid_path, "Distance Plots"),
    num_fids=5, design_matrix=design_matrix, obs_to_fid=True, legend=False,
    statistics=["Cramer", "DeltaVariance", "Dendrogram_Hist",
                "Dendrogram_Num", "PCA", "PDF_Hellinger", "SCF", "VCA", "VCS",
                "VCS_Density", "VCS_Velocity"])

# Des to Obs

des_to_obs_path = os.path.join(path, "Observational Results", "Des_to_Obs")
des_to_obs_comparisons = ["0_obs", "2_obs"]

comparison_plot(
    des_to_obs_path, comparisons=des_to_obs_comparisons,
    out_path=os.path.join(des_to_obs_path, "Distance Plots"),
    num_fids=3, design_matrix=design_matrix,
    legend_labels=["Oph A", "IC 348", "NGC 1333"],
    statistics=["Cramer", "DeltaVariance", "Dendrogram_Hist",
                "Dendrogram_Num", "PCA", "PDF_Hellinger", "SCF", "VCA", "VCS",
                "VCS_Density", "VCS_Velocity"])
