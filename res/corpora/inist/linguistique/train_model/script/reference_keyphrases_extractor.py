#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import codecs
import re
from sys import argv
from os import listdir
from os import path
from lxml import etree

FILE_EXTENSION = ".xml"

if len(argv) != 3:
  print "Usage:\n\t%s <document_directory> <output_directory>"%argv[0]
else:
  document_directory = argv[1]
  output_directory = argv[2]
  inist_output_file = codecs.open(path.join(output_directory, "ref_inist"), "w", "utf-8")
  author_output_file = codecs.open(path.join(output_directory, "ref_auteur"), "w", "utf-8")
  combined_output_file = codecs.open(path.join(output_directory, "ref_inist_et_auteur"), "w", "utf-8")

  for filename in listdir(document_directory):
    if filename.count(FILE_EXTENSION) > 0:
      filepath = path.join(document_directory, filename)
      txt_file = open(filepath, "r")
      txt_content = re.sub(" (corresp|xml:id)=\"[^>]*\">", ">", txt_file.read()).lower()
      xml_content = etree.fromstring(txt_content)
      inist_keyphrases = xml_content.xpath("//tei:keywords[@scheme=\"inist-francis\"][@xml:lang=\"fr\"]/tei:term/text()",
                                           namespaces={"tei": "http://www.tei-c.org/ns/1.0"})
      if len(inist_keyphrases) == 0:
        inist_keyphrases = xml_content.xpath("//tei:keywords[@scheme=\"inist-pascal\"][@xml:lang=\"fr\"]/tei:term/text()",
                                             namespaces={"tei": "http://www.tei-c.org/ns/1.0"})
      author_keyphrases = xml_content.xpath("//tei:keywords[@scheme=\"author\"][@xml:lang=\"fr\"]/tei:term/text()",
                                            namespaces={"tei": "http://www.tei-c.org/ns/1.0"})
      combined_keyphrases = list(set(inist_keyphrases) | set(author_keyphrases))

      txt_file.close()

      if len(inist_keyphrases) != 0:
        inist_output_file.write("%s\t%s\n"%(filename,
                                      reduce(lambda i, k: i + ";" + k,
                                             inist_keyphrases[1:],
                                             inist_keyphrases[0])))
      else:
        inist_output_file.write("%s\t\n"%filename)
      if len(author_keyphrases) != 0:
        author_output_file.write("%s\t%s\n"%(filename,
                                      reduce(lambda i, k: i + ";" + k,
                                             author_keyphrases[1:],
                                             author_keyphrases[0])))
      else:
        author_output_file.write("%s\t\n"%filename)
      if len(combined_keyphrases) != 0:
        combined_output_file.write("%s\t%s\n"%(filename,
                                      reduce(lambda i, k: i + ";" + k,
                                             combined_keyphrases[1:],
                                             combined_keyphrases[0])))
      else:
        combined_output_file.write("%s\t\n"%filename)

  inist_output_file.close()
  author_output_file.close()
  combined_output_file.close()

