# -*- encoding: utf-8 -*-

import math
import networkx
import pickle

from keybench import RankerC
from keybench.default import util as keybench_util

class TopicRankPPRanker(RankerC):
  """
  """

  def __init__(self,
               name,
               is_lazy,
               lazy_directory,
               debug,
               domain_graph_filepath,
               domain_model_filepath, # nb_documents, ngram_counts,
                                      # keyphrase_counts and (ngram, keyphrase)
                                      # pair counts
               oriented,
               controlled_keyphrase_number=float("inf"),
               convergence_threshold=0.001,
               # TODO test multiple values
               recomendation_weight=0.5, # 0.5 = equi-probability
               max_iterations=1000000):
    """
    """

    super(TopicRankPPRanker, self).__init__(name,
                                            is_lazy,
                                            lazy_directory,
                                            debug)

    domain_graph_file = open(domain_graph_filepath, "r")
    domain_model_file = open(domain_model_filepath, "r")
    domain_graph = pickle.load(domain_graph_file)
    domain_model = pickle.load(domain_model_file)

    domain_graph_file.close()
    domain_model_file.close()

    self._domain_graph = domain_graph
    self._domain_model = domain_model
    self._oriented = oriented
    self._controlled_keyphrase_number = controlled_keyphrase_number
    self._convergence_threshold = convergence_threshold
    self._lambda = recomendation_weight
    self._max_iterations = max_iterations

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

    sentences = pre_processed_file.full_text()
    nb_sentences = float(len(sentences))
    topic_indexing = {}
    # initialize the graph with the domain graph (interconnected keyphrases)
    graph = networkx.DiGraph(self._domain_graph)

    # index topics with sentence appearances
    for sentence_index, sentence in enumerate(sentences):
      ngrams = keybench_util.n_to_m_grams(sentence.split(" "), 1, len(sentence.split(" ")))

      for topic_id, topic in enumerate(clusters):
        if len(set(topic) & set(ngrams)) > 0:
          if topic_id not in topic_indexing:
            topic_indexing[topic_id] = []
          topic_indexing[topic_id].append(sentence_index)

    #-- graph creation ---------------------------------------------------------
    nb_documents, ngram_counts, keyphrase_counts, ngram_keyphrase_pair_counts = self._domain_model

    # add document information within the domain graph
    for topic_id_1 in topic_indexing:
      count_1 = float(len(topic_indexing[topic_id_1]))
      p_1 = count_1 / nb_sentences

      # document connections
      for topic_id_2 in topic_indexing:
        if topic_id_1 != topic_id_2:
          pair_count_1_2 = float(len(set(topic_indexing[topic_id_1]) & set(topic_indexing[topic_id_2])))
          p_2_given_1 = pair_count_1_2 / count_1
          p_1_2 = p_1 * p_2_given_1
          weight = 0.0

          if self._oriented:
            weight = p_2_given_1
          else:
            weight = p_1_2

          if weight != 0.0:
            if not graph.has_node(topic_id_1):
              graph.add_node(topic_id_1, {"type": "topic"})
            if not graph.has_node(topic_id_2):
              graph.add_node(topic_id_2, {"type": "topic"})

            graph.add_edge(topic_id_1,
                           topic_id_2,
                           {"type": "intra", "weight": weight})
      # domain connections
      for keyphrase in keyphrase_counts:
        weight_min = 1.0
        weight_max = 0.0
        weight_pro = 1.0
        weight_avg = 0.0

        for tagged_candidate in clusters[topic_id]:
          candidate = " ".join(wt.rsplit(pre_processed_file.tag_separator(), 1)[0] for wt in tagged_candidate.split(" "))

          if candidate in ngram_keyphrase_pair_counts \
             and keyphrase in ngram_keyphrase_pair_counts[candidate]:
            p_candidate = ngram_counts[candidate] / nb_documents
            p_keyphrase_given_candidate = ngram_keyphrase_pair_counts[candidate][keyphrase] / ngram_counts[candidate]
            p_candidate_keyphrase = p_candidate * p_keyphrase_given_candidate

            weight_min = min(weight_min, p_candidate_keyphrase)
            weight_max = max(weight_max, p_candidate_keyphrase)
            weight_pro *= p_candidate_keyphrase
            weight_avg += p_candidate_keyphrase / float(len(clusters[topic_id]))
            # TODO test multiple values
            weight = weight_max

        if weight != 0.0:
          if not graph.has_node(topic_id_1):
            graph.add_node(topic_id_1, {"type": "topic"})

          graph.add_edge(topic_id_1,
                         keyphrase,
                         {"type": "extra", "weight": weight})
          graph.add_edge(keyphrase,
                         topic_id_1,
                         {"type": "extra", "weight": weight})

    #-- random walk ------------------------------------------------------------
    stabilized = False
    nb_iterations = 0
    in_edge_indexing = {}
    out_edge_indexing = {}
    out_sum_indexing = {}
    scores = {}

    # in and out edge indexing for faster processing
    for node in graph.nodes():
      if node not in in_edge_indexing:
        in_edge_indexing[node] = {"intra": [], "extra": []}
      if node not in out_edge_indexing:
        out_edge_indexing[node] = {"intra": [], "extra": []}

      for source, target, data in graph.in_edges(node, data=True):
        in_edge_indexing[node][data["type"]].append((source, data["weight"]))
      for source, target, data in graph.out_edges(node, data=True):
        out_edge_indexing[node][data["type"]].append((target, data["weight"]))

    # initialization
    for node, data in graph.nodes(data=True):
      scores[node] = 1.0

      out_sum_indexing[node] = {}
      for edge_type in out_edge_indexing[node]:
        if edge_type not in out_sum_indexing[node]:
          out_sum_indexing[node][edge_type] = 0.0

        for target, weight in out_edge_indexing[node][edge_type]:
          out_sum_indexing[node][edge_type] += weight

    while not stabilized and nb_iterations < self._max_iterations:
      stabilized = True
      previous_scores = scores.copy()

      for node in graph.nodes():
        previous_score = previous_scores[node]
        new_score = 0.0

        intra_recommendation_sum = 0.0
        extra_recommendation_sum = 0.0

        # compute the intra-recommendation
        for source, weight1 in in_edge_indexing[node]["intra"]:
          out_sum = out_sum_indexing[source]["intra"]
          #out_sum = 0.0

          #for target, weight2 in out_edge_indexing[source]["intra"]:
          #  out_sum += weight2

          intra_recommendation_sum += (weight1 * previous_scores[source]) \
                                      / out_sum
        # compute the extra-recommendation
        for source, weight1 in in_edge_indexing[node]["extra"]:
          out_sum = out_sum_indexing[source]["extra"]
          #out_sum = 0.0

          #for target, weight2 in out_edge_indexing[source]["extra"]:
          #  out_sum += weight2

          extra_recommendation_sum += (weight1 * previous_scores[source]) \
                                      / out_sum

        new_score = ((1.0 - self._lambda) * extra_recommendation_sum) \
                    + (self._lambda * intra_recommendation_sum)

        # look for convergence
        if math.fabs(new_score - previous_score) > self._convergence_threshold:
          stabilized = False

        scores[node] = new_score

      nb_iterations += 1

    ##-- post-processing -------------------------------------------------------
    ranking_results = {}
    tagged_text = " ".join(pre_processed_file.full_text_words())
    untagged_text = " ".join(wt.rsplit(pre_processed_file.tag_separator(), 1)[0] for wt in pre_processed_file.full_text_words())
    sorted_nodes = sorted(graph.nodes(data=True),
                          key=lambda (n, d): (len(d["type"]), scores[n]),
                          reverse=True)
    score_max = max(scores[n] for n, d in sorted_nodes)

    # - treat keyphrases first
    # - add the best score to every keyphrase so that candidates are always
    #   ranked below
    for node, node_data in sorted_nodes:
      # add reference keyphrases if it has a score above 0.0
      if node_data["type"] == "keyphrase" \
         and scores[node] > 0.0 \
         and len(ranking_results) < self._controlled_keyphrase_number \
         and (self._controlled_keyphrase_number != float("inf") \
              or not untagged_text.count(node) > 0):
        ranking_results[node] = scores[node]
        
        # ensure to rank keyphrases first when a subset must be extracted
        if self._controlled_keyphrase_number != float("inf"):
          ranking_results[node] += score_max

      # put only a few candidates per topic
      if node_data["type"] == "topic":
        cluster = sorted(clusters[node],
                         key=lambda c: len(c.split(" ")),
                         reverse=True)
        found_in_domain = False

        # extract (every) candidate already used as keyphrase
        for candidate in cluster:
          untagged_candidate = " ".join(wt.rsplit(pre_processed_file.tag_separator(), 1)[0] for wt in candidate.split(" "))

          if untagged_candidate in keyphrase_counts \
             and untagged_candidate not in ranking_results: # check if the
                                                            # keyphrase has
                                                            # already been added
            found_in_domain = True

            ranking_results[candidate] = scores[node]

        # if no candidate is already used, extract novelty (only one candidate)
        if not found_in_domain:
          best_candidate = ""
          best_first_position = float("inf")

          # find the first appearing candidate
          for candidate in cluster:
            untagged_candidate = " ".join(wt.rsplit(pre_processed_file.tag_separator(), 1)[0] for wt in candidate.split(" "))
            first_position = tagged_text.find(candidate)

            if first_position < best_first_position \
               and untagged_candidate not in ranking_results: # exclude already
                                                              # extracted
                                                              # keyphrases
              best_candidate = candidate
              best_first_position = first_position

          if best_candidate != "":
            ranking_results[best_candidate] = scores[node]

    return ranking_results

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

    return sorted(weights.items(), key=lambda row: row[1], reverse=True)

