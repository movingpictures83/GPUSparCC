# SparCCGPU
# Language: CUDA
# Input: CSV (abundances)
# Output: CSV (correlations)
# Tested with: PluMA 1.0, CUDA 8.0

SparCC on the GPU (Alonso, Escobar, Panoff and Suarez, 2017).  Runs the SparCC algorithm
(Friedman and Alm, 2012) taking advantage of GPU parallelism.

The plugin accepts an input file of abundances in CSV format, with rows
representing samples and columns representing entities.  Entry (i, j) is then the abundance
of entity j in sample i.

The plugin produces an output file of correlations in CSV format computed using
SparCC where both rows and columns represent entities and
entry (i, j) is the correlation between entity i and entity j.
