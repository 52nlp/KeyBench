#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import codecs
import inist
import os
import pickle
import string
import sys
import tempfile
import train_refs

from nltk import data
from nltk.corpus import stopwords
from os import path

SCRIPT_DIRECTORY = path.join(path.dirname(__file__))
BONSAI_CMD = path.join(SCRIPT_DIRECTORY, "bonsai_tokenizer.pl")

if len(sys.argv) == 3:
  train_dir = sys.argv[1]
  output = sys.argv[2]

  # number of documents where each ngram, keyphrase and (ngram, keyphrase) pair
  # appears
  nb_documents = 0.0
  ngram_counts = {}
  keyphrase_counts = {}
  pair_counts = {}

  #-- load documents -----------------------------------------------------------
  for filename in os.listdir(train_dir):
    if filename.endswith(".xml"):
      filepath = path.join(train_dir, filename)
      xml_file = inist.INISTFileRep(); xml_file.parse_file(filepath)
      raw_text = xml_file.title() + "."
      document_ngrams = []
      document_keyphrases = []

      if xml_file.abstract() != "":
        raw_text += " " + xml_file.abstract()
      if xml_file.content() != "":
        raw_text += " " + xml_file.content()

      nb_documents += 1.0

      # tokenize document
      sentence_tokenizer = data.load('tokenizers/punkt/french.pickle')
      sentences = sentence_tokenizer.tokenize(raw_text)
      input_filepath = path.join(tempfile.gettempdir(), "input.tmp")
      output_filepath = path.join(tempfile.gettempdir(), "output.tmp")
      input_file = codecs.open(input_filepath, "w", "utf-8")

      input_file.write("\n".join(sentences))
      input_file.close()
      os.system("%s %s > %s"%(BONSAI_CMD, input_filepath, output_filepath))

      output_file = codecs.open(output_filepath, "r", "utf-8")
      ascii_output_file = open(output_filepath, "r")
      sentences = output_file.read().splitlines()
      ascii_sentences = ascii_output_file.read().splitlines()

      output_file.close()
      ascii_output_file.close()

      #-- extract document n-grams ---------------------------------------------
      for sentence_index, sentence in enumerate(sentences):
        words = sentence.split()
        ascii_words = ascii_sentences[sentence_index].split()

        for n in range(1, 7):#len(words) + 1):
          for i in range(n, 7):#len(words) + 1):
            start = i - n
            end = i
            ngram = " ".join(words[start:end])
            ascii_ngram = " ".join(ascii_words[start:end])

            # filter stopword sequences (only PRE, DET, etc.)
            if len(set(ascii_ngram.split()) & set(stopwords.words("french") + list(string.punctuation))) < len(ngram.split()):
              # increment n-gram document count
              if ngram not in document_ngrams:
                document_ngrams.append(ngram)

                if ngram not in ngram_counts:
                  ngram_counts[ngram] = 0.0
                ngram_counts[ngram] += 1.0

      #-- retrieve document keyphrases -----------------------------------------
      keyphrases = train_refs.train_references[filename]

      for keyphrase in keyphrases:
        # increment keyphrase document count
        if keyphrase not in document_keyphrases:
          document_keyphrases.append(keyphrase)

          if keyphrase not in keyphrase_counts:
            keyphrase_counts[keyphrase] = 0.0
          keyphrase_counts[keyphrase] += 1.0

      #-- create (n-gram, keyphrase) pairs -------------------------------------
      for ngram in document_ngrams:
        for keyphrase in document_keyphrases:
          # increment pair document count
          if ngram not in pair_counts:
            pair_counts[ngram] = {}
          if keyphrase not in pair_counts[ngram]:
            pair_counts[ngram][keyphrase] = 0.0
          pair_counts[ngram][keyphrase] += 1.0

  #-- serialize counts ---------------------------------------------------------
  model = open(output, "w")
  pickle.dump((nb_documents, ngram_counts, keyphrase_counts, pair_counts), model)
  model.close()

  #print "## ngram_counts ######################################################"
  #for ngram, count in ngram_counts.items():
  #  print "%s\t\t\t\t%d"%(ngram, count)
  #print "## keyphrase_counts ##################################################"
  #for keyphrase, count in keyphrase_counts.items():
  #  print "%s\t\t\t\t%d"%(keyphrase, count)
  #print "## pair_counts #######################################################"
  #for ngram in pair_counts:
  #  for keyphrase in pair_counts[ngram]:
  #    print "(%s, %s)\t\t\t\t%d"%(ngram, keyphrase, pair_counts[ngram][keyphrase])
else:
  print "Usage: %s <train_directory> <output_filepath>"%(sys.argv[0])

