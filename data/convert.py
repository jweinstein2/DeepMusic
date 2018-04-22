import os
import pretty_midi
import numpy as np
import midi
from itertools import combinations
from sklearn import preprocessing

import pdb

def _compress(grid):
    reduced_data = []
    for i in range(12):
        reduced_data.append(np.max(grid[:,i::12], axis=1))
    return np.stack(reduced_data, axis=1)

def one_to_multihot(array, n, l=128):
    combos = []
    for i in range(n + 1):
        combos += combinations(range(l), i)

    mlb = preprocessing.MultiLabelBinarizer()
    labels = mlb.fit_transform(combos)

    return np.apply_along_axis(lambda x: labels[x], axis=0, arr=array)


def multi_to_onehot(array, n, l=128):
    combos = []
    for i in range(n + 1):
        combos += combinations(range(l), i)
    look_up = {k: v for v, k in enumerate(combos)}
    # mlb = preprocessing.MultiLabelBinarizer()
    # labels = mlb.fit_transform(combos)

    def ntuple(vector):
        t = tuple(np.nonzero(vector)[0][:n])
        return look_up[t]

    return np.apply_along_axis(ntuple, axis=1, arr=array)

def encode(midifile, compress=False):
    pattern = midi.read_midifile(midifile)
    total_ticks = 0
    musical_events = []
    track = pattern[0]

    for event in track:
        if isinstance(event, midi.SetTempoEvent):
            bpm = event.get_bpm()
            print "tempo event " + str(bpm)
        if isinstance(event, midi.Event):
            total_ticks += event.tick
            musical_events.append(event)

    grid = np.zeros((total_ticks, 128), dtype=np.int16)
    current_vector = np.zeros(128)
    position_in_grid = 0

    for event in musical_events:
        if not isinstance(event, midi.Event):
            position_in_grid += event.tick
        else:
            if event.tick != 0:
                for i in range(event.tick):
                    grid[position_in_grid, :] = current_vector
                    position_in_grid += 1
            if isinstance(event, midi.NoteOffEvent):
                current_vector[event.pitch] = 0
            if isinstance(event, midi.NoteOnEvent):
                current_vector[event.pitch] = event.velocity
    # print grid.shape
    # print total_ticks

    if compress:
        # collapse octaves by taking max for each note
        grid = _compress(grid)

    hold = np.clip(grid, a_min=0, a_max=1)
    previous = np.roll(grid, 1, axis=0)
    previous[0,:] = 0
    change = (previous != grid).astype(np.int32)
    hit = np.multiply(change, hold)

    print("encoded shape {}".format(np.asarray(grid).shape))

    attributes = {
            'bpm': bpm / 2,
            'compressed': compress
    }
    return hold.tolist(), hit.tolist(), attributes


def decode(hold, hit, attributes):
    trans = int(attributes['compressed']) * 72
    bpm = attributes['bpm']

    print("decoded shape {}".format(np.asarray(hold).shape))
    if np.array(hold).shape != np.array(hit).shape:
        print("Decode: Assumption violated")

    pattern = midi.Pattern()
    track = midi.Track()
    tempoEvent = midi.SetTempoEvent()
    tempoEvent.set_bpm(bpm)
    track.append(tempoEvent)
    pattern.append(track)

    n_noteon = 0
    n_noteoff = 0
    prev_hold = np.zeros(len(hold[0]))
    tick_offset = 0

    for tick in range(len(hit)):
        hit_step = hit[tick]
        hold_step = hold[tick]

        for note in range(len(prev_hold)):
            pitch = note + trans
            if hit_step[note] == 1:
                track.append(midi.NoteOnEvent(tick=tick_offset, velocity=100, pitch=pitch))
                n_noteon += 1
                tick_offset = 0
            if hold_step[note] == 0 and prev_hold[note] == 1:
                track.append(midi.NoteOffEvent(tick=tick_offset, pitch=pitch))
                n_noteoff += 1
                tick_offset = 0

        tick_offset += 1
        prev_hold = hold_step

    track.append(midi.EndOfTrackEvent(tick=1))

    print("on decode {} played {} released".format(n_noteon, n_noteoff))

    return pattern

songs_dir = './songs/'

if __name__ == "__main__":
    print("demo")
    arr = [[0,0,0,0],[1,1,1,1],[0,1,0,1]]
    oh = multi_to_onehot(arr, l=4, n=3)
    mh = one_to_multihot(oh, l=4, n=3)
    print arr
    print mh

    for filename in os.listdir(songs_dir):
        if filename.endswith('.mid'):
            hold, hit, a = encode(songs_dir + filename, False)
            oh_hold = multi_to_onehot(hold, 3)
            oh_hit = multi_to_onehot(hit, 3)
            print("onehot shape {}".format(oh_hold.shape))
            mh_hold = one_to_multihot(oh_hold, 3)
            mh_hit = one_to_multihot(oh_hit, 3)
            pattern = decode(mh_hold, hit, a)
            midi.write_midifile(filename.replace(".mid", "_sample.mid"), pattern)
            # songMatrix = midi_to_matrix.midiToNoteStateMatrix(songs_dir + filename)
            # midi_to_matrix.noteStateMatrixToMidi(songMatrix, name="test")
