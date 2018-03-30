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
should_mk_sample = True
sample_dest_dir = '/home/accts/jtw37/shared/'
parallelize = True
verbose = False

# Converts.mid into the key of C major or A minor
# TODO: time signature change on non-zero track?
def preprocess(f):
    root, dirnames, filenames = f
    for filename in fnmatch.filter(filenames, '*.mid'):
        src = (os.path.join(root, filename))
        dst = src.replace(src_dir, dst_dir, 1)

        if verbose:
            print src

        # check if the file already exists
        if os.path.isfile(dst):
            if not should_replace:
                if verbose:
                    print str(filename) + " already transposed"
                return 1

        # XXX: Still missing an exception but multiprocessing makes traceback difficult
        try:
            transpose(src, dst)
        except Exception as e:
            print e

# Attempts to transpose a given root/filename
# uses music21 to recognize key
# pretty_midi to transpose the data
#
# Returns: 1 if successful, 0 otherwise
def transpose(src_filepath, dst_filepath, generate_mp3 = False):
    # convert to music21 and pretty_midi
    try:
        # ignore invalid tempo change warning
        warnings.filterwarnings("ignore", category=RuntimeWarning)
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
    try:
        score = music21.converter.parse(src_filepath)
    except music21.midi.MidiException as e:
        print src_filepath + " skipped b/c music21 error"
        return 0

    # analyze key
    key = score.analyze('key')
    key_string = "{} {}".format(key.tonic.name, key.mode)
    key_string = key_string.replace("-", "b")
    if verbose: print('before: ' + key_string)

    # transpose to target
    from_num = pretty_midi.key_name_to_key_number(key_string)
    if key.mode == "major":
        to_num = pretty_midi.key_name_to_key_number("C major")
    elif key.mode == "minor":
        to_num = pretty_midi.key_name_to_key_number("A minor")
    half_steps = to_num - from_num
    if verbose: print('transposing: ' + str(half_steps))
    for instrument in midi_data.instruments:
        if instrument.is_drum: continue
        for note in instrument.notes:
            note.pitch += half_steps
            # prevent transposing beyond bound
            if note.pitch >= 127: note.pitch -= 12
            if note.pitch <= 0: note.pitch += 12

    # create the directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(dst_filepath))
    except OSError:
        pass
    midi_data.write(dst_filepath)

    if verbose:
        score = music21.converter.parse(dst_filepath)
        key = score.analyze('key')
        key_string = "{} {}".format(key.tonic.name, key.mode)
        key_string = key_string.replace("-", "b")
        print('after: ' + key_string)

    return 1

if __name__ == '__main__':
    # computationally expensive but accurate
    # elements = list(os.walk(src_dir))
    # elements_count = len(elements)
    # print "total elements: " + str(elements_count)
    elements_count = 2199

    if should_mk_sample:
        sample_filepath = "clean_midi/The Beatles/Here Comes the Sun.1.mid"

        # convert to pretty_midi and save without transposing
        midi_data = pretty_midi.PrettyMIDI(sample_filepath)
        midi_data.write(os.path.join(sample_dest_dir, "original_sample.mid"))

        # transpose
        dst = os.path.join(sample_dest_dir, "transposed_sample.mid")
        transpose(sample_filepath, dst)


    if parallelize:
        pool_num = multiprocessing.cpu_count()
        print "running on {} cpus".format(pool_num)
        pool = multiprocessing.Pool(pool_num)
        for _ in tqdm.tqdm(pool.imap_unordered(preprocess, os.walk(src_dir)), total=elements_count):
            pass
        pool.close()
        pool.join()
    else:
        print "running on 1 cpu"
        for i in os.walk(src_dir):
            preprocess(i)
