import Levenshtein


class CompressionEvaluator:
    def evaluate(self, cfg, impl_cls):
        codec = impl_cls()
        print('loading data')
        with open(cfg.test_filename, 'rb') as infile:
            doc = infile.read()
        print('compressing')
        compressed = codec.compress(doc)
        print('decompressing')
        recon = codec.decompress(compressed)
        print('evaluating')
        lossless = compressed == recon
        if lossless:
            distance = 0
        else:
            distance = Levenshtein.distance(doc, recon)
        compression_ratio = len(compressed) / len(doc)
        winner = compression_ratio < 0.1

        return dict(
            lossless=lossless, distance=distance,
            compression_ratio=compression_ratio, winner=winner
        )
