# FlyGym CPU installation and stepping smoke

This is **installation and physics-step feasibility**, not a language-model bridge,
learned motor policy, stable locomotion benchmark, or biological alignment result.
The original project environment was not modified.

- Official package: FlyGym 2.1.0, Python 3.12.11, MuJoCo 3.9.0, macOS arm64.
- Official hybrid controller on MixedTerrainWorld, 42 position-actuated DoFs,
  adhesion enabled, seed 0. No renderer, Warp, GPU or language model.
- 1,000 steps at 0.0001 s = 0.1 simulated seconds, after default warmup.
- Setup/import/warmup: 29.92 s; stepping: 0.5743 s (~1,741 steps/s,
  0.174× real time). This single short warm run is not a performance benchmark.
- Thorax displacement: [0.5268, 0.3176, -0.2516] mm. Finite positions throughout;
  the short duration does not establish upright/stable walking.
- 79 packaged asset files hashed. No external lazy asset downloads occurred.
  The hybrid controller loads the upstream bundled `single_steps_untethered.pkl`;
  it is the pinned package's preprogrammed walking data, not a newly fitted policy.

## Reproduce in a separate environment

```sh
uv venv --python 3.12.11 /tmp/novigrad-flygym-reproduction
uv pip install --python /tmp/novigrad-flygym-reproduction/bin/python \
  -r results/robot-simulator-smoke/requirements-freeze.txt
/tmp/novigrad-flygym-reproduction/bin/python examples/bio_bridge/flygym_smoke.py \
  --out /tmp/novigrad-flygym-reproduction-result --seconds .1
```

`results.json` contains exact versions, distribution METADATA/RECORD hashes,
packaged asset hashes, sample positions and timings. RECORD hashes bind installed
package inventories; they are not a wheel-download lock usable with pip
`--require-hashes`. No wheels, model meshes, or checkpoints are redistributed.
`manifest.json` hashes the result, freeze and reproducer.

## Upstream provenance and licenses

- [Official installation](https://neuromechfly.org/installation/)
- [Official hybrid-controller tutorial](https://neuromechfly.org/tutorials/4c_hybrid_controller/)
- [Pinned FlyGym source v2.1.0](https://github.com/NeLy-EPFL/flygym/tree/v2.1.0)
- [FlyGym Apache-2.0 license](https://github.com/NeLy-EPFL/flygym/blob/v2.1.0/LICENSE)
- [MuJoCo license](https://github.com/google-deepmind/mujoco/blob/main/LICENSE)

Installed FlyGym and MuJoCo distributions declare Apache-2.0, independently of
Novigrad's MIT license. MuJoCo also ships `LICENSES_THIRD_PARTY.md`; individual
other dependencies retain their own licenses. This directory contains our
reproducer and provenance only, not relicensed upstream assets. The simplified
NeuroMechFly meshes and walking data used here came from the FlyGym wheel.
Optional full-size meshes would use the institutional endpoint
`https://datasets.epfl.ch/nely-public-share/flygym_assets/`; none were requested.

## Same-environment replay

A second process repeated the same 0.1-second setup. The step count, timestep, actuated DoFs, initial thorax position, all saved trajectory samples and final displacement matched exactly. Wall-clock timings are excluded. `replay-verification.json` records this limited check; it does not establish cross-machine reproducibility or long-term stability.
