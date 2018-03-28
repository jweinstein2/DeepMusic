#converts all midi files in the current folder
import sys, os, shutil
import music21, pretty_midi
import fnmatch
import multiprocessing, tqdm
import pdb, warnings

# Disable initial warning messages
music21.environment.UserSettings()['warnings'] = 0

#os.chdir("./")
src_dir = 'clean_midi'
dst_dir = 'transposed_midi'
should_replace = False
sample = True

# Converts.mid into the key of C major or A minor
# TODO: time signature change on non-zero track?
def preprocess(f):
    root, dirnames, filenames = f
    success = 0
    total = 0
    for filename in fnmatch.filter(filenames, '*.mid'):
        success = transpose_pretty(root, filename)
        total += 1

# Attempts to transpose a given root/filename
# uses music21 to recognize key
# pretty_midi to transpose the data
#
# Returns: 1 if successful, 0 otherwise
def transpose_pretty(root, filename):
    src_filepath = (os.path.join(root, filename))
    dst_filepath = src_filepath.replace(src_dir, dst_dir, 1)

    if os.path.isfile(dst_filepath):
        if should_replace == False:
            print str(filename) + " already transposed"
            return 1

    # convert to music21 and pretty_midi
    try:
        midi_data = pretty_midi.PrettyMIDI(src_filepath)
    except IOError as e:
        print e
        print src_filepath + " skipped b/c pretty_midi error"
        return 0
    except KeyError as e:
        print e
        print src_filepath + " skipped b/c pretty_midi error"
        return 0
    except ValueError as e:
        print e
        print src_filepath + " skipped b/c pretty_midi error"
        return 0
    except RuntimeWarning as w:
        print w
        print src_filepath + " attempting to continue"
    try:
        score = music21.converter.parse(src_filepath)
    except music21.midi.MidiException as e:
        print src_filepath + " skipped b/c music21 error"
        return 0

    # analyze key and transpose
    key = score.analyze('key')
    key_string = "{} {}".format(key.tonic.name, key.mode)
    key_string = key_string.replace("-", "b")
    from_num = pretty_midi.key_name_to_key_number(key_string)
    if key.mode == "major":
        to_num = pretty_midi.key_name_to_key_number("C major")
    elif key.mode == "minor":
        to_num = pretty_midi.key_name_to_key_number("A minor")
    half_steps = (from_num - to_num + 6) % 12 - 6
    for instrument in midi_data.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += half_steps
                # prevent transposing beyond bound
                if note.pitch >= 127: note.pitch -= 12
                if note.pitch <= 0: note.pitch += 12

    filename = filename.replace(".mid", "_pretty.mid", 1)
    midi_data.write(filename)
    try:
        os.makedirs(os.path.dirname(dst_filepath))
    except OSError:
        pass
    os.rename(filename, dst_filepath)
    return 1

if __name__ == '__main__':
    # computationally expensive but accurate
    # elements = list(os.walk(src_dir))
    # elements_count = len(elements)
    # print "total elements: " + str(elements_count)
    elements_count = 2199
    warnings.simplefilter("once")

    if sample == True:
        sample_filepath = "clean_midi/The Beatles/Here Comes the Sun.1.mid"

        midi_data = pretty_midi.PrettyMIDI(sample_filepath)
        midi_data.write("original_sample.mid")

        # get the key signature using music21 (there might be a faster way)
        score = music21.converter.parse(sample_filepath)
        key = score.analyze('key')
        key_string = "{} {}".format(key.tonic.name, key.mode)
        from_num = pretty_midi.key_name_to_key_number(key_string)
        if key.mode == "major":
            to_num = pretty_midi.key_name_to_key_number("C major")
        elif key.mode == "minor":
            to_num = pretty_midi.key_name_to_key_number("A minor")
        half_steps = from_num - to_num
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note.pitch += half_steps
            print instrument

        midi_data.write("transposed_sample.mid")
        audio_data = midi_data.synthesize()

    pool_num = multiprocessing.cpu_count()
    print "Running with {} cpus".format(pool_num)
    pool = multiprocessing.Pool(pool_num)
    for _ in tqdm.tqdm(pool.imap_unordered(preprocess, os.walk(src_dir)),
                       total = elements_count):
        pass
    pool.close()
