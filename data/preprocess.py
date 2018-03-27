#converts all midi files in the current folder
import os
import music21
import fnmatch
import multiprocessing
import tqdm
import pdb

# converting everything into the key of C major or A minor
# TODO: time signature seems to change
#       handle program change
#       fails on certain channels like drums

# major conversions
majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("G-", 3),("G", 2)])

sharp_convert = dict([('C#', 'B-'),('D#', 'E-'),('E#', 'F'),('F#', 'G-'),('G#', 'A-'),('A#', 'B-'),('B#', 'C')])

# Disable initial warning messages
music21.environment.UserSettings()['warnings'] = 0

#os.chdir("./")
src_dir = 'clean_midi'
dst_dir = 'transposed_midi'
should_replace = False

def sanitize(key):
    if '#' in key:
        print "FOUND"
        key = sharp_convert[key]
    return key

def preprocess(f):

    root, dirnames, filenames = f
    for filename in fnmatch.filter(filenames, '*.mid'):
        src_filepath = (os.path.join(root, filename))
        dst_filepath = src_filepath.replace(src_dir, dst_dir, 1)

        if os.path.isfile(dst_filepath):
            if should_replace == False:
                print 'already transposed. skipping ' + str(filename)
                return

        score = music21.converter.parse(src_filepath)
        key = score.analyze('key')

        print key.tonic.name, key.mode
        keyname = sanitize(key.tonic.name)
        try:
            if key.mode == "major":
                halfSteps = majors[keyname]
            elif key.mode == "minor":
                halfSteps = minors[keyname]
        except KeyError as e:
            print "INVALID KEY: " + keyname
            print src_filepath + " skipped"
            return

        newscore = score.transpose(halfSteps)
        key = newscore.analyze('key')
        print key.tonic.name, key.mode
        newscore.write('midi', filename)
        try:
            os.makedirs(os.path.dirname(dst_filepath))
        except OSError:
            pass
        os.rename(filename, dst_filepath)

        return

if __name__ == '__main__':
    elements = list(os.walk(src_dir))
    elements_count = len(elements)
    # print "total elements: " + str(element_count)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for _ in tqdm.tqdm(pool.imap_unordered(preprocess, elements), total=elements_count):
        pass
    pool.close()
