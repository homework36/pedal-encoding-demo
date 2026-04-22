"""
Three-layer pedal encoding framework (Zhang et al., MEC 2026).

Layer I  – physical:    raw CC64 pedal events (onset, offset, depth trajectory)
Layer II – contextual:  score-aligned descriptors (δ_onset, δ_offset, AR)
Layer III– semantic:    interpretable intent labels
"""

import numpy as np
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Layer I
# ---------------------------------------------------------------------------

@dataclass
class PedalEvent:
    onset_sec: float
    offset_sec: float
    times: np.ndarray          # absolute timestamps of depth samples
    depths: np.ndarray         # normalised [0,1] depth values
    # Layer II descriptors (filled in by layer2_descriptors)
    delta_onset: Optional[float] = None
    delta_offset: Optional[float] = None
    dur_ibi: Optional[float] = None              # duration in IBI units
    ar_notes: list = field(default_factory=list) # AR values for notes sounding during event
    # Layer III
    labels: list = field(default_factory=list)


def extract_pedal_events(controls: list, threshold: int = 10) -> list[PedalEvent]:
    """
    Segment raw CC64 control messages into discrete PedalEvents.

    A new event begins when the pedal crosses `threshold` from below,
    and ends when it drops back below `threshold`.
    """
    cc64 = sorted(
        [c for c in controls if c['number'] == 64],
        key=lambda c: c['time']
    )
    if not cc64:
        return []

    times  = np.array([c['time']  for c in cc64])
    values = np.array([c['value'] for c in cc64], dtype=float) / 127.0

    events = []
    in_event = False
    start_idx = None

    for i, (t, v) in enumerate(zip(times, values)):
        raw = v * 127
        if not in_event and raw >= threshold:
            in_event = True
            start_idx = i
        elif in_event and raw < threshold:
            in_event = False
            seg_t = times[start_idx:i+1]
            seg_v = values[start_idx:i+1]
            events.append(PedalEvent(
                onset_sec=seg_t[0],
                offset_sec=seg_t[-1],
                times=seg_t,
                depths=seg_v,
            ))

    # still open at end of recording
    if in_event:
        seg_t = times[start_idx:]
        seg_v = values[start_idx:]
        events.append(PedalEvent(
            onset_sec=seg_t[0],
            offset_sec=seg_t[-1],
            times=seg_t,
            depths=seg_v,
        ))

    return events


# ---------------------------------------------------------------------------
# Layer II helpers
# ---------------------------------------------------------------------------

def build_score_to_perf_map(sna, pna, alignment) -> interp1d:
    """
    Build a linear interpolator: score onset_quarter → performance onset_sec.
    Uses matched note pairs; deduplicates by taking median perf time per quarter.
    """
    s_id2q = {n['id']: float(n['onset_quarter']) for n in sna}
    p_id2t = {n['id']: float(n['onset_sec'])     for n in pna}

    from collections import defaultdict
    q_to_times = defaultdict(list)
    for a in alignment:
        if a['label'] == 'match':
            sid, pid = a['score_id'], a['performance_id']
            if sid in s_id2q and pid in p_id2t:
                q_to_times[s_id2q[sid]].append(p_id2t[pid])

    qs = np.array(sorted(q_to_times.keys()))
    ts = np.array([np.median(q_to_times[q]) for q in qs])

    return interp1d(qs, ts, kind='linear', bounds_error=False,
                    fill_value=(ts[0], ts[-1]))


def beat_unit_from_timesig(beats: int, beat_type: int) -> float:
    """Return beat duration in quarter notes for a given time signature."""
    quarter_per_beat = 4.0 / beat_type
    # compound meters: 6/8, 9/8, 12/8 → beat = dotted quarter
    if beat_type == 8 and beats % 3 == 0:
        return quarter_per_beat * 3
    return quarter_per_beat


def beat_unit_from_timesig_str(ts_str: str) -> float:
    beats, beat_type = (int(x) for x in ts_str.split('/'))
    return beat_unit_from_timesig(beats, beat_type)


def build_beat_grid(sna, q2t: interp1d, beats_per_measure: int = 4,
                    beat_unit: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Return arrays of (beat_quarter_positions, beat_perf_times) for each beat
    in the score.  beat_unit=1.0 means one beat = one quarter note (4/4 time).
    """
    max_q = float(np.max(sna['onset_quarter'])) + 4 * beat_unit
    beat_qs = np.arange(0.0, max_q, beat_unit)
    beat_ts = q2t(beat_qs)
    valid = np.isfinite(beat_ts)
    return beat_qs[valid], beat_ts[valid]


def layer2_descriptors(events: list[PedalEvent], beat_qs: np.ndarray,
                       beat_ts: np.ndarray, sna, pna, alignment,
                       q2t: interp1d) -> None:
    """
    Fill PedalEvent.delta_onset, .delta_offset, .ar_notes in-place.

    δ = (t_actual − t_grid) / IBI
    AR(n) = d_eff(n) / IOI(n, n+1)  per voice
    """
    # --- timing deltas and duration in IBI units ---
    for ev in events:
        # nearest beat to onset
        idx_on = int(np.argmin(np.abs(beat_ts - ev.onset_sec)))
        t_grid_on = beat_ts[idx_on]
        ibi_on = (beat_ts[idx_on + 1] - beat_ts[idx_on - 1]) / 2 \
            if 0 < idx_on < len(beat_ts) - 1 else 1.0

        idx_off = int(np.argmin(np.abs(beat_ts - ev.offset_sec)))
        t_grid_off = beat_ts[idx_off]
        ibi_off = (beat_ts[idx_off + 1] - beat_ts[idx_off - 1]) / 2 \
            if 0 < idx_off < len(beat_ts) - 1 else 1.0

        ev.delta_onset  = (ev.onset_sec  - t_grid_on)  / max(ibi_on,  1e-6)
        ev.delta_offset = (ev.offset_sec - t_grid_off) / max(ibi_off, 1e-6)
        ev.dur_ibi      = (ev.offset_sec - ev.onset_sec) / max(ibi_on, 1e-6)

    # --- articulation ratio per note ---
    # AR(n) = d_eff(n) / IOI(n, n+1)
    # d_eff = natural played duration, extended to pedal offset if pedal holds note longer.
    # Only computed when a strictly positive IOI exists (skips chordal/simultaneous notes).
    p_id2on  = {n['id']: float(n['onset_sec'])                            for n in pna}
    p_id2off = {n['id']: float(n['onset_sec']) + float(n['duration_sec']) for n in pna}
    s_id2q   = {n['id']: float(n['onset_quarter'])                        for n in sna}
    s_id2v   = {n['id']: n['voice']                                       for n in sna}

    matched = [(a['score_id'], a['performance_id'])
               for a in alignment if a['label'] == 'match']

    from collections import defaultdict
    voice_notes = defaultdict(list)
    for sid, pid in matched:
        if sid in s_id2q and pid in p_id2on:
            voice_notes[s_id2v.get(sid, 0)].append(
                (p_id2on[pid], p_id2off[pid])
            )
    for v in voice_notes:
        voice_notes[v].sort(key=lambda x: x[0])

    for ev in events:
        ars = []
        for v, notes in voice_notes.items():
            for i, (t_on, t_off) in enumerate(notes):
                if t_on > ev.offset_sec or t_off < ev.onset_sec:
                    continue
                # find a valid positive IOI (skip simultaneous/chord notes)
                ioi = None
                for j in range(i + 1, len(notes)):
                    candidate = notes[j][0] - t_on
                    if candidate > 1e-3:   # at least 1 ms gap
                        ioi = candidate
                        break
                if ioi is None:
                    continue
                # effective sounding duration: note natural end or pedal end, whichever later
                d_eff = max(t_off, ev.offset_sec) - t_on
                ars.append(d_eff / ioi)
        ev.ar_notes = ars


# ---------------------------------------------------------------------------
# Layer III  (Mozart-calibrated thresholds)
# ---------------------------------------------------------------------------
# Thresholds derived from full Batik-plays-Mozart corpus (n=10,245 events):
#   delta_onset:  p25=-0.10, med=0.06,  p75=0.18
#   delta_offset: p25=-0.21, med=-0.06, p75=0.06
#   max_depth:    p10=0.37,  p25=1.00  (>75 % reach full depth)
#   dur_ibi:      p10=0.30,  med=0.75, p75=1.07, p90=1.92
#   AR_median:    p25=1.03,  med=1.60, p75=2.68

# Labels stay anchored to Zhang et al.'s framework but thresholds are
# corpus-calibrated. Priority order: half-pedal > touch > anticipatory >
# extended > rhythmic > note-blend > other.

_AR_CORPUS_P75  = 2.68   # note-blend: top quartile of AR
_DUR_TOUCH      = 0.30   # touch: bottom decile of dur_ibi
_DUR_SUSTAIN    = 1.07   # extended: top quartile of dur_ibi
_DEPTH_HALF     = 0.50   # half-pedal: well below corpus median depth
_DON_ANTICIPATE = -0.10  # anticipatory: p25 of delta_onset


def layer3_labels(events: list[PedalEvent]) -> None:
    """
    Assign Mozart-calibrated semantic labels in-place.
    Events may carry multiple labels reflecting compound function.

      half-pedal   – shallow depression (max_d < 0.50); coloristic use
      touch        – very brief event (dur_ibi < 0.30); accent/articulation
      anticipatory – pressed clearly before beat (δ_onset < -0.10); legato across barline
      extended     – held > 1 beat (dur_ibi > 1.07); pedal duration exceeds one beat unit
      rhythmic     – Wechselpedal: at/after beat, released before next (classical standard)
      note-blend   – high note overlap (AR_median > p75); notes bleeding past natural IOI
      other        – does not fit any category clearly
    """
    for ev in events:
        if ev.delta_onset is None:
            continue

        d_on    = ev.delta_onset
        d_off   = ev.delta_offset
        max_d   = float(np.max(ev.depths)) if len(ev.depths) > 0 else 0.0
        dur_ibi = ev.dur_ibi if ev.dur_ibi is not None else 1.0
        ar_med  = float(np.median(ev.ar_notes)) if ev.ar_notes else 1.0

        labels = []

        # half-pedal: shallow depression for coloristic resonance
        if max_d < _DEPTH_HALF:
            labels.append('half-pedal')

        # touch: very brief — accent or staccato articulation aid
        if dur_ibi < _DUR_TOUCH and max_d >= _DEPTH_HALF:
            labels.append('touch')

        # anticipatory: pressed before beat — legato connection across barline
        if d_on < _DON_ANTICIPATE and max_d >= _DEPTH_HALF and dur_ibi >= _DUR_TOUCH:
            labels.append('anticipatory')

        # extended: pedal held beyond one beat — duration-based, independent of overlap
        if dur_ibi > _DUR_SUSTAIN and max_d >= _DEPTH_HALF:
            labels.append('extended')

        # rhythmic (Wechselpedal): at/after beat, released before next
        if (d_on >= _DON_ANTICIPATE and d_off < 0.05
                and _DUR_TOUCH <= dur_ibi <= _DUR_SUSTAIN):
            labels.append('rhythmic')

        # note-blend: high note overlap relative to corpus norm (AR-based, not duration)
        if ar_med > _AR_CORPUS_P75 and max_d >= _DEPTH_HALF and dur_ibi >= _DUR_TOUCH:
            labels.append('note-blend')

        ev.labels = labels if labels else ['other']
