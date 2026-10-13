# VAE MPS/MLX re-verification archive (partial)

This branch archives existing local benchmark scripts and five JSON reports
for three scenes: single_subject (MPS + MLX), spatial (MPS + MLX),
and indoor (MPS only).

- Source files came from an uncommitted local working tree and are preserved verbatim.
- The three MPS frame arrays (*.npy, about 60 MB each) and ignored logs
  remain local. JSON paths may refer to local files unavailable in CI.
- Outdoor, camera_motion and indoor MLX records are NOT RUN / NOT AVAILABLE in this archive.
- Python source syntax and JSON parsing were checked; no new model/VR/GPU
  inference or numerical regression test was run for this archival commit.
- A process RSS figure does not represent full unified GPU memory pressure.
- Do not interpret these partial outputs as a final five-scene VAE PASS gate.
