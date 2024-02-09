from tqdm import tqdm
from piraye import NormalizerBuilder
from piraye.normalizer_builder import Config
from piraye.nltk_tokenizer import NltkTokenizer
import re
import os


class Preprocessor:
    def __init__(self):
        self.normalizer = NormalizerBuilder(
            [Config.PUNCTUATION_FA, Config.ALPHABET_FA, Config.DIGIT_FA, Config.ALPHABET_EN, Config.DIGIT_EN,
             Config.DIGIT_FA, Config.DIACRITIC_DELETE, Config.SPACE_NORMAL, Config.PUNCTUATION_FA,
             Config.PUNCTUATION_EN],
            remove_extra_spaces=True,
            tokenization=True).build()
        self.tokenizer = NltkTokenizer()

    def preprocess_line(self, text: str):
        text = self.normalizer.normalize(text)
        text = re.sub(r'\b[A-Z]+\b', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        list_of_sentences = [self.tokenizer.word_tokenize(sen) for sen in self.tokenizer.sentence_tokenize(text)]
        tokens = [item for sublist in list_of_sentences for item in sublist]
        text = ' '.join(tokens)
        return text

    def preprocess_document(self, text: str):
        lines = text.splitlines()
        return '\n'.join([self.preprocess_line(text_line) for text_line in lines])

preprocessor = Preprocessor()
text = "این یک متن تست است. مواظب باشید. تروخدا"
print(preprocessor.preprocess_line(text))