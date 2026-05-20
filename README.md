# Pedal Encoding Demo

> **Note:** This is an exploratory implementation accompanying the paper, not a canonical reference implementation. Thresholds and label definitions reflect design decisions made during development and may not be the only reasonable choices.

A working implementation of the three-layer sustain pedal encoding framework applied to the **Batik-plays-Mozart** corpus (Hu & Widmer, 2023).

---

## The Three Layers

| Layer | Name | What it captures |
|-------|------|-----------------|
| I | Physical | Raw CC64 event segmentation: onset, offset, depth trajectory |
| II | Contextual | Score-aligned descriptors: δ_onset, δ_offset (IBI-normalized), articulation ratio AR |
| III | Semantic | Interpretable intent labels |

### Layer II descriptors

- **δ_onset / δ_offset**: timing of pedal press/release relative to nearest beat, normalized by local IBI (inter-beat interval). Positive = after beat, negative = before beat.
- **AR** (articulation ratio): `d_eff / IOI` per voice, where `d_eff` is the note's effective sounding duration (natural end or pedal release, whichever is later) and IOI is the inter-onset interval to the next note in the same voice. AR > 1 means the pedal extends notes past their natural duration.

### Layer III labels

Labels are assigned in order; an event can carry multiple labels reflecting compound function.

| Label | Conditions | Interpretation |
|-------|-----------|---------------|
| `rhythmic` | δ_onset ≥ −0.30, 0.30 ≤ dur ≤ 1.50 IBI | Near-beat press, moderate duration; classical rhythmic pedal |
| `half-pedal` | max depth < 0.50 | Shallow depression; coloristic use |
| `touch` | dur < 0.30 IBI | Very brief press at any depth; accent or articulation aid |
| `anticipatory` | δ_onset < −0.10, AR > 1, full depth | Pressed before beat, notes carry over; legato across barline (δ_offset not required) |
| `pedaled-legato` | δ_onset > 0, δ_offset > −0.20, AR > 1, full depth | Pressed after beat, sustained with note blending; syncopated sustain |
| `other` | none of the above | Does not fit any category clearly |

**Design notes:**
- `rhythmic` is checked first — most timing-constrained label. No AR or depth guard; rhythmic pedaling can be performed at any depth.
- `AR` is computed from score-performance alignment (voice labels + matched note pairs from `.match` files). Requires the Batik corpus `.match` files; not available from raw MIDI.
- `δ_offset` is not used for `rhythmic` or `anticipatory` — for these labels, the exact release position relative to the nearest beat does not define the intent. A small tolerance (`δ_offset > −0.20`) is applied to `pedaled-legato` to handle long pedals where the release landing slightly before a beat is noise.
- `half-pedal` covers all low-depth events (max depth < 0.50) regardless of duration. Brief shallow events will carry both `touch` and `half-pedal` as compound labels.

### Corpus-wide label distribution (Batik-plays-Mozart, n = 10,245 events, 36 movements)

| Label | Count | % of events |
|-------|-------|-------------|
| `rhythmic` | 6,342 | 61.9% |
| `pedaled-legato` | 3,330 | 32.5% |
| `anticipatory` | 1,469 | 14.3% |
| `half-pedal` | 1,209 | 11.8% |
| `touch` | 1,101 | 10.7% |
| `other` | 657 | 6.4% |

Percentages sum beyond 100% because labels are not mutually exclusive.

---

## Notebooks

### 01: Data loading and exploration
Tours the corpus structure: loading a match file with `partitura`, inspecting the raw CC64 pedal signal, visualizing the score-to-performance note alignment, and displaying harmony/cadence annotations.

### 02: Three-layer encoding framework
Full implementation of all three layers on KV279 mvt. 1, plus corpus-wide label distribution across all 36 movements.

### 03: Pedal depth profiles at cadences
Loads all 36 movements, samples CC64 depth in a ±2-beat window around each annotated cadence, and averages by cadence type. PAC produces the sharpest pedal lift at the cadence beat; HC a shallower lift; IAC the least pronounced — mirroring the musicological strength ranking.

---

## Structure

```
pedal-encoding-demo/
├── src/
│   └── layers.py                     # Three-layer framework implementation
├── notebooks/
│   ├── 01_data_loading.ipynb         # Dataset tour
│   ├── 02_layer_framework.ipynb      # Full encoding on KV279 mvt. 1 + corpus stats
│   └── 03_cadence_analysis.ipynb     # Pedal depth at cadences, all 36 movements
```

---

## Setup

```bash
conda create -n pedal python=3.11
conda activate pedal
pip install partitura numpy scipy pandas matplotlib jupyter
```

Set `DATASET = '../../batik_plays_mozart-main'` at the top of each notebook.

---

## References
- Hu, Q. & Widmer, G. (2023). *A Continuous Representation of Tempo in the Batik-plays-Mozart Dataset.* ISMIR 2023.
- Zhang et al. (2026). *A Context-Aware Piano Performance Encoding Framework for Sustain Pedaling.* MEC 2026.

