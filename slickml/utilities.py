def join_dictionaries(dict1, dict2):
    """
    Join Two Dictionaries
    Function to join two input dictionaries. For the pairs
    with the same keys, the set of values will be stored in a list.
    --------------------------
    Parameters:
               - dict1: dictionary or key-value pairs
               - dict2: dictionary or key-value pairs

    """
    if not (isinstance(dict1, dict) and isinstance(dict2, dict)):
        raise TypeError("The Type for dict1 and dict2 should be dict!")

    dictionary = {}
    d1Keys = list(dict1.keys())
    d2Keys = list(dict2.keys())
    combinedKeys = list(set(d1Keys + d2Keys))

    for key in combinedKeys:
        d1Vals = []
        d2Vals = []
        if key in d1Keys:
            d1Vals = dict1[key]
            if isinstance(d1Vals, (int, float, str)):
                d1Vals = [d1Vals]

        if key in d2Keys:
            d2Vals = dict2[key]
            if isinstance(d2Vals, (int, float, str)):
                d2Vals = [d2Vals]

        dictionary[key] = list(set(d1Vals + d2Vals))

    return dictionary
