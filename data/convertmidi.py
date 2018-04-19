import os
import pretty_midi
# import midi_to_matrix
import numpy as np
import midi




def midi_encode(midifile):
    # prettymidi = pretty_midi.PrettyMIDI(songs_dir + filename)
    # piano_roll = np.array(prettymidi.get_piano_roll(fs=10000), dtype=np.int16) #returns 128 x length
    # print piano_roll.shape
    # transposed = piano_roll.T #want length x 128
    # return transposed.tolist()

    pattern = midi.read_midifile(midifile)
    total_ticks = 0
    musical_events = []
    track = pattern[0]

    for event in track:
        # if isinstance(event, midi.SetTempoEvent):
        #   print "tempo event" + str(event.get_bpm())
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
    print grid.shape
    print total_ticks
    return grid.tolist()


def midi_decode(grid):
    pattern = midi.Pattern()
    track = midi.Track()
    tempoEvent = midi.SetTempoEvent()
    tempoEvent.set_bpm(30)
    track.append(tempoEvent)
    pattern.append(track)

    previous_vector = grid[0] # first vector
    for note_index in range(len(previous_vector)):
        if previous_vector[note_index] != 0: #velocity is not 0
            track.append(midi.NoteOnEvent(tick=0, velocity=previous_vector[note_index], pitch=note_index))

    tickoffset = 0
    for vector in grid:
        if previous_vector == vector: #if vectors are same, no new events
            tickoffset += 1
        else:
            for note_index in range(len(previous_vector)):
                if previous_vector[note_index] == vector[note_index]: #if same velocity, hold the note rather than rearticulate (no new event)
                    continue
                if previous_vector[note_index] != 0 and vector[note_index] == 0:
                    if note_index > 127:
                        print "BROKEN ASSUMPTION"
                    track.append(midi.NoteOffEvent(tick=tickoffset, pitch=note_index))
                else:
                    track.append(midi.NoteOnEvent(tick=0, velocity=vector[note_index], pitch=note_index))
                tickoffset = 0
            tickoffset += 1
        previous_vector = vector
    track.append(midi.EndOfTrackEvent(tick=1))
    return pattern

songs_dir = './songs/'

if __name__ == "__main__":

    for filename in os.listdir(songs_dir):
        if filename.endswith('.mid'):
            matrix = midi_encode(songs_dir + filename)
            pattern = midi_decode(matrix)
            pdb.set_trace()
            midi.write_midifile("decoded_" + filename, pattern)
            # songMatrix = midi_to_matrix.midiToNoteStateMatrix(songs_dir + filename)
            # midi_to_matrix.noteStateMatrixToMidi(songMatrix, name="test")
