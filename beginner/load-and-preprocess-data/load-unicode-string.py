"""
Unicode strings
---
Models that process natural language often handle different languages with different
character sets. Unicode is a standard encoding system that is used to represent
character from almost all languages. Each character is encoded using a unique integer
code point between 0 and 0x10FFFF. A Unicode string is a sequence of zero or more code
points.
This tutorial shows how to represent Unicode strings in TensorFlow and manipulate them
using Unicode equivalents of standard string ops. It separates Unicode strings into tokens
based on script detection.
---
Original source: https://www.tensorflow.org/tutorials/load_data/unicode
"""

import tensorflow as tf


def main():
    # The `tf.string` data type
    tf.constant(u'Thanks 😊')
    tf.constant([u"You're", u"welcome!"]).shape

    # Representing Unicode
    # Unicode string, represented as a UTF-8 encoded string scalar.
    text_utf8 = tf.constant(u"语言处理")
    text_utf8

    # Unicode string, represented as a UTF-16-BE encoded string scalar.
    text_utf16be = tf.constant(u"语言处理".encode("UTF-16-BE"))
    text_utf16be

    # Unicode string, represented as vector of Unicode code points.
    text_chars = tf.constant([ord(char) for char in u"语言处理"])
    text_chars

    # Converting between representations
    tf.strings.unicode_decode(text_utf8, input_encoding='UTF-8')

    tf.strings.unicode_encode(text_chars, output_encoding='UTF-8')

    tf.strings.unicode_transcode(text_utf8, input_encoding='UTF8', output_encoding='UTF-16-BE')

    # Batch dimensions
    # A batch of Unicode strings, each represented as a UTF8-encoded string.
    batch_utf8 = [
        s.encode('UTF-8') for s in
        [u'hÃllo',  u'What is the weather tomorrow',  u'Göödnight', u'😊']
    ]
    batch_chars_ragged = tf.strings.unicode_decode(batch_utf8, input_encoding='UTF-8')

    for sentence_chars in batch_chars_ragged.to_list():
        print(sentence_chars)

    batch_chars_padded = batch_chars_ragged.to_tensor(default_value=-1)
    print(batch_chars_padded.numpy())

    batch_chars_sparse = batch_chars_ragged.to_sparse()

    tf.strings.unicode_encode(
        [[99, 97, 116], [100, 111, 103], [99, 111, 119]],
        output_encoding='UTF-8'
    )

    tf.strings.unicode_encode(batch_chars_ragged, output_encoding='UTF-8')

    tf.strings.unicode_encode(
        tf.RaggedTensor.from_sparse(batch_chars_sparse),
        output_encoding='UTF-8'
    )

    tf.strings.unicode_encode(
        tf.RaggedTensor.from_tensor(batch_chars_padded, padding=-1),
        output_encoding='UTF-8'
    )

    # Unicode operations
    # Character length

    # Note that the final character takes up 4 bytes in UTF8.
    thanks = u'Thanks 😊'.encode('UTF-8')
    num_bytes = tf.strings.length(thanks).numpy()
    num_chars = tf.strings.length(thanks, unit='UTF8_CHAR').numpy()
    print(f'{num_bytes} bytes; {num_chars} UTF-8 characters')

    # Character substrings

    # default: unit='BYTE'. With len=1, we return a single byte.
    tf.strings.substr(thanks, pos=7, len=1).numpy()

    # Specifying unit='UTF8_CHAR', we return a single character, which in this case
    # is 4 bytes.
    print(tf.strings.substr(thanks, pos=7, len=1, unit='UTF8_CHAR').numpy())

    # Split Unicode strings
    tf.strings.unicode_split(thanks, 'UTF-8').numpy()

    # Byte offsets for characters
    codepoints, offsets = tf.strings.unicode_decode_with_offsets(u"🎈🎉🎊", 'UTF-8')

    for (codepoint, offset) in zip(codepoints.numpy(), offsets.numpy()):
        print(f'At byte offset {offset}: codepoint {codepoint}')

    # Unicode scripts
    uscript = tf.strings.unicode_script([33464, 1041])  # ['芸', 'Б']
    print(uscript.numpy())                              # [17, 8] == [USCRIPT_HAN, USCRIPT_CYRILLIC]

    print(tf.strings.unicode_script(batch_chars_ragged))

    # Example: Simple segmentation
    # dtype: string/ shape: [num_sentences]
    #
    # The sentences to process. Edit this line to try out different inputs!
    sentence_texts = [u'Hello, world.', u'世界こんにちは']

    # dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
    #
    # sentence_char_codepoint[i, j] is the codepoint for the j'th character in
    # the i'th sentence.
    sentence_char_codepoint = tf.strings.unicode_decode(sentence_texts, 'UTF-8')
    print(sentence_char_codepoint)

    # dtype: int32; shape: [num_sentences, (num_chars_per_sentence)]
    #
    # sentence_char_scripts[i, j] is the unicode script of the j'th character in
    # the i'th sentence.
    sentence_char_scripts = tf.strings.unicode_script(sentence_char_codepoint)
    print(sentence_char_scripts)

    # dtype: bool; shape: [num_sentences, (num_chars_per_sentence)]
    #
    # sentence_char_starts_word[i, j] is True if the j'th character in the i'th
    # sentence is the start of a word.
    sentence_char_starts_word = tf.concat([
        tf.fill([sentence_char_scripts.nrows(), 1], True),
        tf.not_equal(sentence_char_scripts[:, 1:], sentence_char_scripts[:, :-1])
    ], axis=1)

    # dtype: int64; shape: [num_words]
    #
    # word_starts[i] is the index of the character that starts the i'th word (in
    # the flattened list of characters from all sentences).
    word_starts = tf.squeeze(tf.where(sentence_char_starts_word.values), axis=1)
    print(word_starts)

    # dtype: int32; shape: [num_words, (num_chars_per_word)]
    #
    # word_char_codepoint[i, j] is the codepoint for the j'th character in the
    # i'th word.
    word_char_codepoint = tf.RaggedTensor.from_row_starts(
        values=sentence_char_codepoint.values,
        row_starts=word_starts
    )
    print(word_char_codepoint)

    # dtype: int64; shape: [num_sentences]
    #
    # sentence_num_words[i] is the number of words in the i'th sentence.
    sentence_num_words = tf.reduce_sum(
        tf.cast(sentence_char_starts_word, tf.int64),
        axis=1
    )

    # dtype: int32; shape: [num_sentences, (num_words_per_sentence), (num_chars_per_word)]
    #
    # sentence_word_char_codepoint[i, j, k] is the codepoint for the k'th character
    # in the j'th word in the i'th sentence.
    sentence_word_char_codepoint = tf.RaggedTensor.from_row_lengths(
        values=word_char_codepoint,
        row_lengths=sentence_num_words
    )
    print(sentence_word_char_codepoint)

    tf.strings.unicode_encode(sentence_word_char_codepoint, 'UTF-8').to_list()


if __name__ == '__main__':
    main()
