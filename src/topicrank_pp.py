# -*- encoding: utf-8 -*-

from keybench import RankerC

class TopicRankPP(RankerC):
  """
  """

  def __init__(self,
               name,
               is_lazy,
               debug,
               domain_graph,
               domain_model, # ngram_counts, keyphrase_counts
                             # and (ngram, keyphrase) pair counts
               convergence_threshold=0.001,
               recomendation_weight=0.85,
               max_iteration=1000000):
    """
    """

    super(TopicRankPP, self).__init__(name, is_lazy, lazy_directory, debug)

    self._domain_graph = domain_graph
    self._domain_model = domain_model
    self._convergence_threshold = convergence_threshold
    self._recomendation_weight = recomendation_weight
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

    # TODO assign topic IDs to candidates (__TOPICRANKPP_CLUSTER_1__, etc.)
    # TODO properly identify topics within the document
    # TODO count n-grams and n-gram pairs for probability (or occurrence counts)
    #      weighting using a sentence window
    # TODO create a graph containing the domain_graph and add topic nodes and
    #      inter-document edges
    # TODO ranking using a modified random walk
    # TODO report topic scores to each candidate

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

