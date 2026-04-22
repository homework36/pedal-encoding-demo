# Pedal Encoding Demo

A working preliminary implementation of the three-layer sustain pedal encoding framework applied to the **Batik-plays-Mozart** corpus (Hu & Widmer, 2023).

---

## Notebooks

### 01: Data loading and exploration
Tours the corpus structure: loading a match file with `partitura`, inspecting the raw CC64 pedal signal, visualizing the score-to-performance note alignment, and displaying harmony/cadence annotations along the performance timeline. Confirms the dataset is well-formed (2,803 score notes, 7,620 CC64 messages for KV279 mvt. 1; complete match alignment with zero deletions) and introduces the data structures used throughout.

### 02: Three-layer encoding framework
Full implementation of all three layers on KV279 mvt. 1.

- **Layer I:** segments 240 discrete pedal events from the CC64 stream by threshold crossing
- **Layer II:** computes score-aligned descriptors for each event: δ_onset and δ_offset (timing of pedal press/release relative to nearest beat, normalized by IBI) and articulation ratio AR (effective sounding duration / IOI, per voice)
- **Layer III:** assigns Mozart-calibrated semantic labels; each event can carry multiple labels reflecting its compound function. In KV279 mvt. 1, the most commonly assigned label is *rhythmic* (53% of events), followed by *extended* (22%) and *note-blend* (18%). Percentages sum beyond 100% because labels are not mutually exclusive.
- **Figure 3:** four-panel aligned view showing piano roll (score context), raw CC64 depth (Layer I), timing deltas (Layer II), and semantic label color bands (Layer III)

### 03: Pedal depth profiles at cadences (full corpus)
Loads all 36 movements, finds annotated cadence onsets, and samples CC64 depth in a ±2-beat window normalized by local IBI. Averages profiles by cadence type across PAC (n=547), HC (n=410), IAC (n=75), and DC (n=47). The main result is that pedal depth at the cadential beat follows harmonic strength: PAC produces the sharpest lift (mean depth 0.11, near-complete pedal release), HC a shallower lift (0.16), and IAC the least pronounced (0.22). This hierarchy mirrors the musicological strength ranking and holds consistently per-sonata. The final subplot grid shows per-sonata PAC profiles, confirming the behavior is not driven by a single outlier.

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
| I | Physical | Raw CC64 event segmentation: onset, offset, depth trajectory |
| II | Contextual | Score-aligned descriptors: δ_onset, δ_offset (in IBI units), articulation ratio AR |
| III | Semantic | Interpretable intent labels calibrated to the Mozart corpus |

### Layer III labels (Mozart-calibrated)

The original thresholds were designed generically; applying them naively to Mozart produced degenerate results (83% of events labeled as heavy note overlap) because even a brief pedal press extends notes past a short IOI in classical texture. To fix this, we ran all three layers across the full 36-movement corpus (n = 10,245 events) and set thresholds from the resulting distributions: δ_onset p25 = −0.10, dur_ibi p10/p75 = 0.30/1.07, AR p75 = 2.68, and depth median ~1.0 (over 75% of events reach full pedal depth, so 0.50 is the right half-pedal cutoff). Across the full corpus, the most commonly assigned labels are *rhythmic* (40% of events), *extended* (23%), and *note-blend* (22%), with *anticipatory* at 19%. These are co-occurrence rates across all 36 movements; a single event may carry more than one label.

Thresholds derived from the full Batik-plays-Mozart corpus (n = 10,245 pedal events):

| Label | Criterion | Interpretation |
|-------|-----------|---------------|
| `half-pedal` | max depth < 0.50 | Shallow depression for coloristic resonance |
| `touch` | dur < 0.30 IBI, full depth | Very brief: accent or staccato articulation aid |
| `anticipatory` | δ_onset < −0.10 IBI | Pressed before beat: legato across barline |
| `extended` | dur > 1.07 IBI | Pedal held beyond one beat (duration axis) |
| `rhythmic` | δ_onset ≥ −0.10, δ_offset < 0.05, 0.30 ≤ dur ≤ 1.07 | Classical Wechselpedal: at/after beat, clean release |
| `note-blend` | AR median > 2.68 | Notes bleeding past natural IOI (overlap axis) |
| `other` | none of the above | |

## Additional Key Finding

Pedal depth profiles sampled in a ±2-beat window around cadence onsets reveal systematic differences by cadence type across all 36 movements:

- **PAC** (n = 547): pedal lifted at the cadence (depth drops to ~0.11), then restored, confirming the performer marks harmonic resolution with a clean pedal change
- **HC** (n = 410): gentler lift (~0.16 at cadence), consistent with the harmonic tension remaining unresolved
- **IAC / DC**: more varied profiles reflecting their intermediate cadential weight

This pedal-depth analogue of the local tempo analysis in Hu & Widmer (2023, Fig. 4) confirms that pedaling, like tempo, is a musically structured expressive parameter that responds systematically to harmonic syntax.

## Setup

```bash
conda create -n pedal-enc python=3.11
conda activate pedal-enc
pip install partitura numpy scipy pandas matplotlib jupyter
```

Requires the Batik-plays-Mozart dataset (match files + annotated score parts). Update the `DATASET` path at the top of each notebook.

```bash
jupyter notebook notebooks/
```

Run the notebooks in order: 01 → 02 → 03.

## References

- Zhang et al. (2026). *A Context-Aware Piano Performance Encoding Framework for Sustain Pedaling.* MEC 2026.
- Hu, Q. & Widmer, G. (2023). *A Continuous Representation of Tempo in the Batik-plays-Mozart Dataset.* ISMIR 2023.
