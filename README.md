# Pedal Encoding Demo

A working implementation of the three-layer sustain pedal encoding framework from:

> Zhang et al., "A Context-Aware Piano Performance Encoding Framework for Sustain Pedaling," *Music Encoding Conference (MEC) 2026*.

Applied to the **Batik-plays-Mozart** corpus (Hu & Widmer, 2023): 36 Mozart piano sonata movements performed by Roland Batik on a Bösendorfer SE290 with continuous CC64 pedal data.

---

## Structure

```
pedal-encoding-demo/
├── src/
│   └── layers.py          # Three-layer framework implementation
├── notebooks/
│   ├── 01_data_loading.ipynb      # Dataset tour: match files, CC64 signal, annotations
│   ├── 02_layer_framework.ipynb   # Full three-layer encoding on KV279 mvt. 1
│   └── 03_cadence_analysis.ipynb  # Pedal depth profiles at cadences, all 36 movements
└── figures/
    ├── cadence_depth_profiles.pdf  # Main corpus result (Fig. 4 analogue)
    └── PAC_by_sonata.pdf           # Per-sonata PAC profiles
```

## The Three Layers

| Layer | Name | What it captures |
|-------|------|-----------------|
| I | Physical | Raw CC64 event segmentation — onset, offset, depth trajectory |
| II | Contextual | Score-aligned descriptors: δ_onset, δ_offset (in IBI units), articulation ratio AR |
| III | Semantic | Interpretable intent labels calibrated to the Mozart corpus |

### Layer III labels (Mozart-calibrated)

Thresholds derived from the full Batik-plays-Mozart corpus (n = 10,245 pedal events):

| Label | Criterion | Interpretation |
|-------|-----------|---------------|
| `half-pedal` | max depth < 0.50 | Shallow depression for coloristic resonance |
| `touch` | dur < 0.30 IBI, full depth | Very brief — accent or staccato articulation aid |
| `anticipatory` | δ_onset < −0.10 IBI | Pressed before beat — legato across barline |
| `sustaining` | dur > 1.07 IBI | Held beyond one beat — harmonic blending |
| `rhythmic` | δ_onset ≥ −0.10, δ_offset < 0.05, 0.30 ≤ dur ≤ 1.07 | Classical Wechselpedal — at/after beat, clean release |
| `over-legato` | AR median > 2.68 | Heavy note overlap beyond rhythmic norm |
| `other` | none of the above | — |

## Key Finding

Pedal depth profiles sampled in a ±2-beat window around cadence onsets reveal systematic differences by cadence type across all 36 movements:

- **PAC** (n = 547): pedal lifted *at* the cadence (depth drops to ~0.11), then restored — confirms the performer marks harmonic resolution with a clean pedal change
- **HC** (n = 410): gentler lift (~0.16 at cadence), consistent with the harmonic tension remaining unresolved
- **IAC / DC**: more varied profiles reflecting their intermediate cadential weight

This pedal-depth analogue of the local tempo analysis in Hu & Widmer (2023, Fig. 4) shows that pedaling, like tempo, is a musically structured expressive parameter that responds systematically to harmonic syntax.

## Setup

```bash
conda create -n pedal-enc python=3.11
conda activate pedal-enc
pip install partitura numpy scipy pandas matplotlib jupyter
```

Requires the Batik-plays-Mozart dataset at `/Volumes/Extreme SSD/batik_plays_mozart/` (match files + annotated score parts). Update the `DATASET` path at the top of each notebook if your copy is elsewhere.

```bash
jupyter notebook notebooks/
```

Run the notebooks in order: 01 → 02 → 03.

## References

- Zhang et al. (2026). *A Context-Aware Piano Performance Encoding Framework for Sustain Pedaling.* MEC 2026.
- Hu, Q. & Widmer, G. (2023). *A Continuous Representation of Tempo in the Batik-plays-Mozart Dataset.* ISMIR 2023.
