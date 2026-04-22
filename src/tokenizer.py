class WordTokenizer:
    def __init__(self, text):
        # split text into words
        words = text.split()

        # unique words
        vocab = sorted(set(words))

        # mappings
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}

        self.vocab_size = len(vocab)

    def encode(self, text):
        return [self.stoi.get(w, 0) for w in text.split()]

    def decode(self, tokens):
        return " ".join([self.itos[t] for t in tokens])


# class CharTokenizer:
#     def __init__(self, text):
#         chars = sorted(list(set(text)))
#         self.stoi = {ch: i for i, ch in enumerate(chars)}
#         self.itos = {i: ch for ch, i in self.stoi.items()}
#         self.vocab_size = len(chars)

#     def encode(self, text):
#         return [self.stoi[c] for c in text]

#     def decode(self, tokens):
#         return ''.join([self.itos[t] for t in tokens])
