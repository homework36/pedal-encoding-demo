"""
Score-free Layer II for MAESTRO MIDI files.

MAESTRO files carry a fixed 120-BPM MIDI tempo map that does not reflect
actual musical tempo.  Beat positions are estimated from note-onset density
via librosa beat tracking.

AR is approximated from MIDI note IOI without score-voice separation: for each
note sounding during a pedal event the next distinct note onset in the full
polyphonic stream is used as the IOI denominator.  This is noisier than the
score-aligned version but correctly distinguishes legato (AR > 1) from
staccato passages.
"""
from __future__ import annotations

import numpy as np
import librosa


def _beat_grid_from_onsets(pm, sr: int = 22050, hop: int = 512) -> np.ndarray:
    """Return musical beat times estimated from note-onset density."""
    notes    = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    end_time = pm.get_end_time()
    n_frames = int(end_time * sr / hop) + 1
    onset_sig = np.zeros(n_frames)
    for n in notes:
        idx = int(n.start * sr / hop)
        if 0 <= idx < n_frames:
            onset_sig[idx] += n.velocity / 127.0
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_sig, sr=sr, hop_length=hop, start_bpm=80
    )
    beat_ts = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    if len(beat_ts) < 2:
        beat_ts = np.arange(0.0, end_time + 0.5, 0.5)
        return beat_ts

    # Extrapolate at median IBI so the grid covers the full piece.
    # Librosa only finds beats in active sections; sparse openings/codas otherwise
    # produce delta_onset values of -50 to -200, which are nonsensical.
    ibi = float(np.median(np.diff(beat_ts)))
    pre, t = [], beat_ts[0] - ibi
    while t > -ibi:
        pre.append(t)
        t -= ibi
    post, t = [], beat_ts[-1] + ibi
    while t <= end_time + ibi:
        post.append(t)
        t += ibi
    if pre or post:
        beat_ts = np.concatenate([pre[::-1], beat_ts, post])
    return beat_ts


def _nearest_beat(t: float, beat_ts: np.ndarray):
    """Return (t_grid, local_ibi) for time t."""
    idx = int(np.argmin(np.abs(beat_ts - t)))
    lo  = max(0, idx - 1)
    hi  = min(len(beat_ts) - 1, idx + 1)
    ibi = (beat_ts[hi] - beat_ts[lo]) / max(hi - lo, 1)
    return beat_ts[idx], max(ibi, 1e-6)


def layer2_beats_proxy(events, pm) -> None:
    """
    Fill PedalEvent.delta_onset, .delta_offset, .dur_ibi, .ar_notes in-place.

    IBI proxy  — librosa beat tracking on note-onset density.
    AR proxy   — MIDI note IOI from the full polyphonic note list (no voice
                 separation); simultaneous notes (IOI < 1 ms) are skipped.
    """
    beat_ts = _beat_grid_from_onsets(pm)

    for ev in events:
        t_grid_on,  ibi_on  = _nearest_beat(ev.onset_sec,  beat_ts)
        t_grid_off, ibi_off = _nearest_beat(ev.offset_sec, beat_ts)
        ev.delta_onset  = (ev.onset_sec  - t_grid_on)  / ibi_on
        ev.delta_offset = (ev.offset_sec - t_grid_off) / ibi_off
        ev.dur_ibi      = (ev.offset_sec - ev.onset_sec) / ibi_on

    notes_all = sorted(
        [n for inst in pm.instruments if not inst.is_drum for n in inst.notes],
        key=lambda n: n.start,
    )
    if not notes_all:
        return

    onsets = np.array([n.start for n in notes_all])

    for ev in events:
        active_idx = [
            i for i, n in enumerate(notes_all)
            if n.start < ev.offset_sec and n.end > ev.onset_sec
        ]
        ars = []
        for i in active_idx:
            t_on  = notes_all[i].start
            t_off = notes_all[i].end
            j = int(np.searchsorted(onsets, t_on + 1e-3))
            if j >= len(notes_all):
                continue
            ioi = notes_all[j].start - t_on
            if ioi <= 1e-6:
                continue
            d_eff = max(t_off, ev.offset_sec) - t_on
            ars.append(d_eff / ioi)
        ev.ar_notes = ars
