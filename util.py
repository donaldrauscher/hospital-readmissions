import pandas as pd
import numpy as np

# flatten array of items, some of which can be arrays
def flatten(y):
    y = [i if isinstance(i, (np.ndarray, list, tuple)) else [i] for i in y]
    cuts = list(np.cumsum([len(i) for i in y])[:-1])
    flat = [item for sublist in y for item in sublist]
    return flat, cuts if len(flat) > (len(cuts) + 1) else None

# unflattens array
def unflatten(y, cuts):
    return np.split(y, cuts) if cuts else y

# add prefix to keys in a dictionary
def add_dict_prefix(x, px):
    return {'%s__%s' % (px, k) : v for k,v in x.items()}

# https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)
def stars_and_bars(bins, stars, allowEmpty=True):
    """
    Non-recursive generator that returns the possible ways that n indistinguishable objects
    can be distributed between k distinguishible bins (allowing empty bins)
    Total number of arrangements = (n+k-1)! / n!(k-1)! if empty bins are allowed
    Total number of arrangements = (n-1)! / (n-k)!(k-1)! if empty bins are not allowed
    Parameters
    ----------
    bins : int
        Number of distinguishible bins (k)
    stars : int
        Number of indistinguishible objects (n)
    allowEmpty : boolean
        If True, empty bins are allowed; default is True
    """

    if bins < 1 or stars < 1:
        raise ValueError("Number of objects (stars) and bins must both be greater than or equal to 1.")
    if not allowEmpty and stars < bins:
        raise ValueError("Number of objects (stars) must be greater than or equal to the number of bins.")

    # If there is only one bin, there is only one arrangement!
    if bins == 1:
        yield stars,
        return

    # If empty bins are not allowed, distribute (star-bins) stars and add an extra star to each bin when yielding.
    if not allowEmpty:
        if stars == bins:
            # If same number of stars and bins, then there is only one arrangement!
            yield tuple([1] * bins)
            return
        else:
            stars -= bins

    # 'bars' holds the queue or stack of positions of the bars in the stars and bars arrangement
    # (including a bar at the beginning and end) and the level of iteration that this stack item has reached.
    # Initial stack holds a single arrangement ||...||*****...****| with an iteration level of 1.
    bars = [([0]*bins + [stars], 1)]

    # Iterate through the current queue of arrangements until no more are left (all arrangements have been yielded).
    while len(bars) > 0:
        newBars = []

        for b in bars:
            # Iterate through inner arrangements of b, yielding each arrangement and queuing each
            # arrangement for further iteration except the very first
            for x in range(b[0][-2], stars+1):
                newBar = b[0][1:bins] + [x, stars]
                if b[1] < bins-1 and x > 0:
                    newBars.append((newBar, b[1]+1))

                # Translate the stars and bars into a tuple
                yield tuple(newBar[y] - newBar[y-1] + (0 if allowEmpty else 1) for y in range(1, bins+1))

        bars = newBars
