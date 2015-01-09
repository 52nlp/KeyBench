#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
from corpus_file import CorpusFileRep
from lxml import etree

class INISTFileRep(CorpusFileRep):
  """
  """

  def __init__(self):
    """
    Constructor.
    """

    super(INISTFileRep, self).__init__()

  def parse_file(self, filepath):
    """
    Parses a corpus file and initialize the object.
    
    @param  filepath: The path of the corpus file to parse.
    @type   filepath: C{string}
    """

    xml_file = open(filepath, "r")
    xml = re.sub(" (corresp|xml:id)=\"[^>]*\">", ">", xml_file.read())
    xml = re.sub("</?sup>", "", xml)
    xml = re.sub("</?sub>", "", xml)
    xml_file.close()
    doc = etree.fromstring(xml)

    # parse title
    title = doc.xpath("//tei:sourceDesc/tei:biblStruct/tei:analytic/tei:title[@type=\"main\"][@xml:lang=\"fr\"]",
                       namespaces={"tei": "http://www.tei-c.org/ns/1.0"})[0].text + "."
    self.set_title(title)

    # parse the abstract
    abstract = ""
    for p in doc.xpath("//tei:profileDesc/tei:abstract[@xml:lang=\"fr\"]/tei:p",
                       namespaces={"tei": "http://www.tei-c.org/ns/1.0"}):
      if not p.text == None:
        if abstract != "":
          abstract += " "
        abstract += p.text.strip()
    self.set_abstract(abstract)

