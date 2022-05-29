import itertools
import operator

def _extract_ranges(data):
    # convert a list of integers into a list of contiguous ranges
    # [2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 20] -> [(2,5), (12,17), (20, 20)]
    # http://stackoverflow.com/a/2154437
    ranges = []
    for key, group in itertools.groupby(enumerate(sorted(data)), lambda x: x[0] - x[1]):
        group = list(map(operator.itemgetter(1), group))
        if len(group) > 1:
            ranges.append((group[0], group[-1]))
        else:
            ranges.append((group[0], group[0]))

    return ranges