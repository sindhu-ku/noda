import ROOT

def plot_scaled_histograms(file_name, hist_names, legend_names=None, output_file=None):
    """
    Plots multiple histograms from a ROOT file on the same canvas after scaling.

    Parameters:
        file_name (str): Path to the ROOT file.
        hist_names (list of str): List of histogram names to be plotted.
        legend_names (list of str, optional): List of legend names corresponding to histograms.
        output_file (str, optional): File path to save the canvas. Defaults to None.
    """
    # Open the ROOT file
    root_file = ROOT.TFile.Open(file_name, "READ")
    if not root_file or root_file.IsZombie():
        raise FileNotFoundError(f"Cannot open the ROOT file: {file_name}")

    # Check that legend names match hist_names
    if legend_names and len(legend_names) != len(hist_names):
        raise ValueError("The length of 'legend_names' must match 'hist_names'.")

    # Prepare a canvas
    canvas = ROOT.TCanvas("c1", "Scaled Histograms", 1800, 1500)
    canvas.SetGrid()
    legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)

    # Loop over histograms
    hist_list = []
    for i, hist_name in enumerate(hist_names):
        hist = root_file.Get(hist_name)
        if not hist:
            print(f"Warning: Histogram {hist_name} not found in {file_name}")
            continue

        # Normalize histogram
        hist.Scale(1.0 / hist.Integral() if hist.Integral() > 0 else 1.0)

        # Set style and add to canvas
        hist.SetLineColor(i + 1)
        hist.SetLineWidth(2)
        hist_list.append(hist)

        # Draw the histogram
        draw_option = "HIST SAME" if i > 0 else "HIST"
        hist.Draw(draw_option)
        hist.GetXaxis().SetTitle("Neutrino energy [MeV]")
        hist.GetYaxis().SetTitle("a.u.")
        hist.GetXaxis().SetLabelSize(0.05)  # Set X-axis label font size
        hist.GetYaxis().SetLabelSize(0.05)  # Set Y-axis label font size
        hist.GetXaxis().SetTitleSize(0.06)  # Set X-axis title font size
        hist.GetYaxis().SetTitleSize(0.06)  # Set Y-axis title font size
        # Add to legend
        legend_label = legend_names[i] if legend_names else hist_name
        legend.AddEntry(hist, legend_label, "l")
        legend.SetTextSize(0.05)

    # Draw legend
    legend.Draw()

    # Save the canvas if an output file is specified
    if output_file:
        canvas.SaveAs(output_file)

    # Keep the canvas open
    canvas.Update()
    input("Press Enter to exit...")

    # Close the ROOT file
    root_file.Close()

# Example usage
if __name__ == "__main__":
    root_file_name = "Sindhu_geo.root"
    histogram_names = ["raw_geoTh", "raw_geoU"]
    legend_labels = ["^{232}Th", "^{238}U"]
    plot_scaled_histograms(root_file_name, histogram_names, legend_names=legend_labels)
