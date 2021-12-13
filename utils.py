import gc


def cleanup(*args):
    for var in args:
        del var
        gc.collect()


def split_to_batches(data, batch):
    for i in range(0, len(data), batch):
        yield data[i:i + batch]
