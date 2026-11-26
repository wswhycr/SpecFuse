# Data provenance and licensing

## SLT tables (`data/*.xlsx`)

The three tables contain haemagglutination-inhibition (HI) titres for ordered
influenza strain pairs, together with the HA sequences of the strains involved.
They are redistributed here from the public dataset released with

    Liu et al., Expert Systems with Applications 298 (2026) 129925 (DiffAD)

which points to a Figshare collection (https://figshare.com/s/a3cc87207a2b49cbbb3d)
and to the GitHub repository `lywey/FluAntigen`
(https://github.com/lywey/FluAntigen, "Hemagglutination Inhibition Titer Dataset
of Multiple Influenza Subtypes").

**No license is declared for the FluAntigen repository** (GitHub reports
`license: null`) and the DiffAD article does not state one for the data. The
underlying HI measurements originate from WHO/CDC surveillance tables that have
been digitised by earlier antigenic-cartography studies.

Consequences for redistribution:

- Use of these tables for research, with citation, follows normal academic
  practice.
- **Redistribution requires permission from the data authors.** Release of this
  package must therefore be confirmed with the corresponding author before it is
  made public. If permission is not obtained, publish `scripts/fetch_data.py`
  instead of the tables themselves.

## ProtVec (`data/protVec_100d_3grams.csv`)

100-dimensional ProtVec trigram embeddings, as distributed with the DiffAD
release; originally described in Asgari & Mofrad, PLoS ONE 10(3): e0118432
(2015). Same redistribution caveat as above.

## Derived artefacts (`splits/`, `frozen/`)

`splits/` contains index sets and observed distances for the three splits of
each dataset and seed. `frozen/` contains the weights and outputs of the
upstream encoder (gated local-Fourier adapter, graph embeddings).

These were produced by us from the tables above and are released with the code.
Because they are derived from the same third-party measurements, the same
redistribution caveat applies.

## PLM embeddings (`embeddings/`)

Two embedding families are provided, because they were computed on two different
variants of the benchmark tables:

- `plm_sequences/` - sequence-level embeddings (1024-1536 d) for BioGPT (1.6B),
  ESM2_t33 (650M), ProGen2-base (355M) and ProtBERT (420M), stored as CSV with a
  metadata header. These were computed on the **published DiffAD tables**
  (`nature566H1N1`, `nature566H3N2`, `nature585BVictoria`, without the
  "不对称" suffix) and are the embeddings behind the PLM comparison.
- `plm_strains_protT5/` - per-strain ProtT5-XL embeddings (1024 d), stored as
  `.npy` with a `.names.txt` row index and a `.json` header. These were computed
  on the **asymmetric tables** used elsewhere in this paper
  (`nature566H1N1不对称`, `nature566H3N2不对称`,
  `nature585BVictoria不对称`).

TODO before release: the two families must be aligned to a single table variant
so that the PLM comparison and the main results are computed on identical data.

The embeddings were computed by us from the sequences in the tables; the
pretrained weights remain under the licences of their respective releases
(BioGPT: MIT; ESM2: MIT; ProGen2: BSD-3; ProtBERT: MIT; ProtT5: Apache-2.0).
