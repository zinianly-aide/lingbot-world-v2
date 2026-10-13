# Quest Q1/Q2 local streaming evidence archive

The seven JSON files in this branch are snapshots of prior local
Q1 latent tap, Q1.5 progressive VAE, and Q2 live Quest streaming runs.
They are archived verbatim from the local working directory.

- A recorded `"pass": true` describes that particular historical run,
  **not** a fresh test of this branch, GPU, or a current Quest session.
- Paths and `127.0.0.1` bridge URLs refer to the originating Mac.
- Binary latents, JPEG frames, caches, logs and other large generated
  artifacts are not committed.
- Format check: all seven source JSON files parse.
- Live model/GPU/Quest playback and end-to-end regression: NOT RUN
  while preparing this archival branch.
- Based on the reviewed Q2.6 experiment branch; no change to signaling,
  generation weights, main branch or existing G0.x baselines.
