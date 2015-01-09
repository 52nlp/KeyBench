#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import codecs
import networkx
import os
import pickle
import sys
import train_refs

from os import path

if len(sys.argv) == 2:
  output = sys.argv[1]

  #-- document counts of keyphrase pairs ---------------------------------------
  keyphrase_counts = {}
  pair_counts = {}

  for document in train_refs.train_references:
    document_keyphrases = []

    for keyphrase1 in train_refs.train_references[document]:
      # keyphrase count
      if keyphrase1 not in document_keyphrases:
        document_keyphrases.append(keyphrase1)

        if keyphrase1 not in keyphrase_counts:
          keyphrase_counts[keyphrase1] = 0.0
        keyphrase_counts[keyphrase1] += 1.0

      # pair count
      associated_keyphrases = []
      for keyphrase2 in train_refs.train_references[document]:
        if keyphrase1 != keyphrase2:
          if keyphrase2 not in associated_keyphrases:
            associated_keyphrases.append(keyphrase2)

            if keyphrase1 not in pair_counts:
              pair_counts[keyphrase1] = {}
            if keyphrase2 not in pair_counts[keyphrase1]:
              pair_counts[keyphrase1][keyphrase2] = 0.0
            pair_counts[keyphrase1][keyphrase2] += 1.0

  #-- create graph -------------------------------------------------------------
  graph = networkx.DiGraph()

  for keyphrase1 in pair_counts:
    if not graph.has_node(keyphrase1):
      graph.add_node(keyphrase1, {"type": "keyphrase"})

    for keyphrase2 in pair_counts[keyphrase1]:
      if not graph.has_node(keyphrase2):
        graph.add_node(keyphrase2, {"type": "keyphrase"})

      p_k1 = keyphrase_counts[keyphrase1] / len(train_refs.train_references)
      p_k2_given_k1 = pair_counts[keyphrase1][keyphrase2] / keyphrase_counts[keyphrase1]
      p_k1_k2 = p_k1 * p_k2_given_k1

      graph.add_edge(keyphrase1, keyphrase2, {"type": "intra", "weight": p_k1_k2})

  #networkx.write_dot(graph, "debug_model_intra_uni.dot")

  #-- serialize graph ----------------------------------------------------------
  model = open(output, "w")
  pickle.dump(graph, model)
  model.close()
else:
  print "Usage: %s <output_filepath>"%(sys.argv[0])

