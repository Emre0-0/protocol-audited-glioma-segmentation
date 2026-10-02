# Third-party attribution

`research/vendor/BraTS-2024-Metrics/` contains selected, unmodified files from
[rachitsaluja/BraTS-2024-Metrics](https://github.com/rachitsaluja/BraTS-2024-Metrics),
commit `f1131e340e155908553c226675644a6363b998b7`, licensed under Apache-2.0.
The upstream LICENSE, README, original requirements and source headers are retained.
The bundled `surface_distance` package retains its original source notices.
`UPSTREAM.json` records SHA-256 hashes; the release verifier checks those files.

nnU-Net v2.7.0 and dynamic-network-architectures are installed dependencies,
not redistributed framework forks. Their authors retain their respective rights.
The custom trainers derive from nnU-Net's public interfaces. Cite nnU-Net and
the BraTS post-treatment challenge when using this pipeline.

The project wrapper temporarily aliases `np.Inf` to `np.inf` for NumPy 2
compatibility. The vendor metric algorithm is unchanged. Its published
requirements are retained for provenance; install the release's root
requirements files for the tested environment.

The repository's MIT license covers project-authored code and documentation.
It does not replace upstream licenses, authorize dataset redistribution, or
grant rights to checkpoint weights that are not included in the repository.
