# -*- encoding: utf-8 -*-

import math
import networkx

from keybench import RankerC
from keybench import util as keybench_util

class TopicRankPP(RankerC):
  """
  """

  def __init__(self,
               name,
               is_lazy,
               debug,
               domain_graph,
               domain_model, # nb_documents, ngram_counts, keyphrase_counts
                             # and (ngram, keyphrase) pair counts
               oriented,
               convergence_threshold=0.001,
               recomendation_weight=0.85,
               max_iteration=1000000):
    """
    """

    super(TopicRankPP, self).__init__(name, is_lazy, lazy_directory, debug)

    self._domain_graph = domain_graph
    self._domain_model = domain_model
    self._convergence_threshold = convergence_threshold
    self._lambda = recomendation_weight
    self._max_iteration = max_iteration

  def weighting(self, pre_processed_file, candidates, clusters):
    """
    Takes a pre-processed text (list of POS-tagged sentences) and gives a weight
    to its candidates keyphrases.

    @param    pre_processed_file: The pre-processed file.
    @type     pre_processed_file: C{PreProcessedFile}
    @param    candidates:         The keyphrase candidates.
    @type     candidates:         C{list(string)}
    @param    clusters:           The clustered candidates.
    @type     clusters:           C{list(list(string))}

    @return:  A dictionary of terms as key and weight as value.
    @rtype:   C{dict(string, float)}
    """

    topic_indexing = {}
    # initialize the graph with the domain graph (interconnected keyphrases)
    graph = networkx.DiGraph(self._domain_graph)

    # index topics with sentence appearances
    for sentence_index, sentence in enumerate(sentences):
      ngrams = keybench_util.n_to_m_grams(sentence, 1, len(sentence))

      for topic_id, topic in enumerate(clusters):
        if len(set(topic) & set(ngrams)) > 0:
          if topic_id not in topic_indexing:
            topic_indexing[topic_id] = []
          topic_indexing[topic_id].append(sentence_index)

    #-- graph creation ---------------------------------------------------------
    nb_documents, ngram_counts, keyphrase_counts, ngram_keyphrase_pair_counts = self._domain_model
    sentences = pre_processed_file.full_text()

    # add document information within the domain graph
    for topic_id_1 in topic_indexing:
      count_1 = float(len(topic_indexing[topic_id_1]))
      p_1 = count_1 / float(len(sentences))

      # document connections
      for topic_id_2 in topic_indexing:
        if topic_d_1 != topic_id_2:
          pair_count_1_2 = float(len(set(topic_indexing[topic_id_1]) & set(topic_indexing[topic_id_2])))
          p_2_given_1 = pair_count_1_2 / count_1
          p_1_2 = p_1 * p_2_given_1
          weight = 0.0

          if oriented:
            weight = p_2_given_1
          else:
            weight = p_1_2

          if weight != 0.0:
            if not graph.has_node(topic_id_1):
              graph.add_node(topic_id_1, {"type": "candidate"})
            if not graph.has_node(topic_id_2):
              graph.add_node(topic_id_2, {"type": "candidate"})

            graph.add_edge(topic_id_1,
                           topic_id_2,
                           {"type": "intra", "weight": weight})
      # domain connections
      for keyphrase in keyphrase_counts:
        for tagged_candidate in clusters[topic_id]:
          candidate = " ".join(wt.rsplit(pre_processed_file.tag_separator(), 1)[0] for wt in tagged_candidate.split(" "))

          if candidate in ngram_keyphrase_pair_counts \
             and keyphrase in ngram_keyphrase_pair_counts[ngram][keyphrase]:
            p_candidate = ngram_counts[candidate] / nb_documents
            p_keyphrase_given_candidate = ngram_keyphrase_pair_counts[candidate][keyphrase] / ngram_counts[candidate]
            p_candidate_keyphrase = p_candidate * p_keyphrase_given_candidate
            weight = p_candidate_keyphrase

            if weight != 0.0:
              graph.add_edge(topic_id_1,
                             keyphrase,
                             {"type": "extra", "weight": weight})
              graph.add_edge(keyphrase,
                             topic_id_1,
                             {"type": "extra", "weight": weight})

    #-- random walk ------------------------------------------------------------
    stabilized = False
    nb_iterations = 0
    scores = {}

    for node in graph.nodes():
      scores[node] = 1.0

    while not stabilized and nb_iterations < self._max_iterations:
      stabilized = True
      previous_scores = scores.copy()

      for node in graph.nodes():
        previous_score = previous_scores[node]
        new_score = 0.0

        intra_recommendation_sum = 0.0
        extra_recommendation_sum = 0.0

        # compute the intra- and extra-recommendation
        for source1, target1, data1 in graph.in_edges(node, data=True):
          if data1["type"] == "intra":
            out_sum = 0.0

            for source2, target2, data2 in graph.out_edges(target1, data=True):
              if data2["type"] == "intra":
                out_sum += data2["weight"]

            intra_recommendation_sum += (data1["weight"] * previous_score) / out_sum
          if data1["type"] == "extra":
            out_sum = 0.0

            for source2, target2, data2 in graph.out_edges(target1, data=True):
              if data2["type"] == "extra":
                out_sum += data2["weight"]

            extra_recommendation_sum += (data1["weight"] * previous_score) / out_sum

        new_score = ((1.0 - self._lambda) * extra_recommendation_sum) \
                    + (self._lambda * intra_recommendation_sum)

        if math.fabs(new_score - previous_score) > self._convergence_threshold:
          stabilized = False

        scores[identifier] = new_score

      nb_iterations += 1

    #-- post-processing --------------------------------------------------------
    # TODO report topic scores to each candidate

    # TODO return ...
    pass

  def ordering(self, weights, clusters):
    """
    Takes the weighted terms of the analysed text and ordered them.

    @param    weights:  A dictionary of weighted candidates.
    @type     weights:  C{dict(string, float)}
    @param    clusters: The clustered candidates.
    @type     clusters: C{list(list(string))}

    @return:  A ordered list of weighted terms.
    @rtype:   C{list(tuple(string, float))}
    """

    pass

