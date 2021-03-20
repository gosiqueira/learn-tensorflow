"""
TF.Text
---
TensorFlow Text provides a collection of text related classes and ops ready to use
with TensorFlow 2.0. The library can perform the preprocessing regularly required by
text-based models, and includes other features useful for sequence modeling not provided
by core TensorFlow.
The benefit of using these ops in your text preprocessing is that they are done in the
TensorFlow graph. You do not need to worry about tokenization in training being different
than the tokenization at inference, or managing preprocessing scripts.
---
Remember to install TensorFlow Text:
$ pip install -q tensorflow-text
---
Original source: https://www.tensorflow.org/tutorials/tensorflow_text/intro
"""

import tensorflow as tf
import tensorflow_text as text


def main():
    # Unicode
    docs = tf.constant([
        u'Everything not saved will be lost.'.encode('UTF-16-BE'),
        u'Sad☹'.encode('UTF-16-BE')
    ])
    _ = tf.strings.unicode_transcode(docs, input_encoding='UTF-16-BE', output_encoding='UTF-8')

    # Tokenization
    # WhitespaceTokenizer
    tokenizer = text.UnicodeScriptTokenizer()
    tokens = tokenizer.tokenize(['everything not saved will be lost', u'Sad☹'.encode('UTF-8')])
    print(f'Tokens: {tokens.to_list()}')

    # Unicode split
    tokens = tf.strings.unicode_split([u"仅今年前".encode('UTF-8')], 'UTF-8')
    print(f'Tokens: {tokens.to_list()}')

    # Offsets
    tokenizer = text.UnicodeScriptTokenizer()
    (tokens, _, end_offsets) = tokenizer.tokenize_with_offsets([
        'everything not saved will be lost.', u'Sad☹'.encode('UTF-8')
    ])

    print(f'Tokens: {tokens.to_list()}')
    print(f'Offsets: {end_offsets.to_list()}')

    # TF.Data Example
    docs = tf.data.Dataset.from_tensor_slices([
        ['Never tell me the odds.'],
        ["It's a trap!"]
    ])
    tokenizer = text.WhitespaceTokenizer()
    tokenized_docs = docs.map(lambda x: tokenizer.tokenize(x))
    iterator = iter(tokenized_docs)
    print(f'First sentence tokens: {next(iterator).to_list()}')
    print(f'Seconds sentence tokens: {next(iterator).to_list()}')

    # Other Text Ops
    # Wordshape
    tokenizer = text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize([
        'Everything not saved will be lost.',
        u'Sad☹'.encode('UTF-8')
    ])

    # Is capitalized?
    f1 = text.wordshape(tokens, text.WordShape.HAS_TITLE_CASE)
    # Are all letters uppercased
    f2 = text.wordshape(tokens, text.WordShape.IS_UPPERCASE)
    # Does the token contain punctuation?
    f3 = text.wordshape(tokens, text.WordShape.HAS_SOME_PUNCT_OR_SYMBOL)
    # Is the token a number?
    f4 = text.wordshape(tokens, text.WordShape.IS_NUMERIC_VALUE)

    print(f'Is capitalized? {f1.to_list()}')
    print(f'Are all letters uppercased? {f2.to_list()}')
    print(f'Does the token contain punctuation? {f3.to_list()}')
    print(f'Is the token a number? {f4.to_list()}')

    # N-grams & Sliding Window
    tokenizer = text.WhitespaceTokenizer()
    tokens = tokenizer.tokenize([
        'Everything not saved will be lost.',
        u'Sad☹'.encode('UTF-8')
    ])

    # Ngrams, in this case bi-gram (n = 2)
    bigrams = text.ngrams(tokens, 2, reduction_type=text.Reduction.STRING_JOIN)

    print(f'Bi-grams: {bigrams.to_list()}')


if __name__ == '__main__':
    main()
