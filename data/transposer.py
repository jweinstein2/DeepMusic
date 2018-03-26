#converts all midi files in the current folder
import os
import music21
import fnmatch

# converting everything into the key of C major or A minor
# TODO: time signature seems to change
#       handle program change
#       fails on certain channels like drums

# major conversions
majors = dict([("A-", 4),("A", 3),("B-", 2),("B", 1),("C", 0),("D-", -1),("D", -2),("E-", -3),("E", -4),("F", -5),("G-", 6),("G", 5)])
minors = dict([("A-", 1),("A", 0),("B-", -1),("B", -2),("C", -3),("D-", -4),("D", -5),("E-", 6),("E", 5),("F", 4),("G-", 3),("G", 2)])

# Disable initial warning messages
music21.environment.UserSettings()['warnings'] = 0

#os.chdir("./")
src_dir = 'clean_midi'
dst_dir = 'transposed_midi'
for root, dirnames, filenames in os.walk(src_dir):
    for filename in fnmatch.filter(filenames, '*.mid'):
        src_filepath = (os.path.join(root, filename))
        dst_filepath = src_filepath.replace(src_dir, dst_dir, 1)

        score = music21.converter.parse(src_filepath)
        key = score.analyze('key')

        # print key.tonic.name, key.mode

        try:
            if key.mode == "major":
                halfSteps = majors[key.tonic.name]
            elif key.mode == "minor":
                halfSteps = minors[key.tonic.name]
        except KeyError as e:
            print "INVALID KEY: " + key.tonic.name
            print src_filepath + " skipped"
            continue

        newscore = score.transpose(halfSteps)
        key = newscore.analyze('key')
        print key.tonic.name, key.mode
        new_filename = "C_" + filename
        newscore.write('midi', new_filename)
        try:
            os.makedirs(os.path.dirname(dst_filepath))
        except OSError:
            pass
        os.rename(new_filename, dst_filepath)
