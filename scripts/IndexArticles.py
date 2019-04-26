from __future__ import division
import xml.etree.ElementTree as ET
import re, glob, time, os
import json
from collections import defaultdict
import nltk
import dewiki
from pyspark import SparkContext, SparkConf
import pickle
sc = SparkContext("local[20]", "Counting Indicators")
# sc = SparkContext("local[4]", "Counting Indicators")

def read_article(article_ID, root_dir = "/data/ClinicalTrialsWikipediaArticles/articles/"):
    article = json.load(open(root_dir + article_ID + ".json", "r"))
    wikitext = article["text"]
    text = dewiki.from_string(wikitext)
    text = remove_refs(text)
    text = remove_lists(text)
    return text

def remove_refs(text):
    text = re.sub('<ref( [^>]*)?>([^<>]*</ref>)?', '', text)
    text = re.sub("\[(.*?)\]", "", text)
    return text

def remove_lists(text):
    text = re.sub('(^|(\{\{[^\n]*)?\n)\s*(\*|\|)\s*.*(\n\}\}|(?=\n|$))', "", text)
    return text

def create_regex_format(x):
    regex = re.compile(r"[^a-z0-9](" + x + r")[^a-z0-9]")
    return regex

def find_indicators(element, indicators_dict = {}, current = "root"):
    if type(element) == dict:
        subelements = element.keys()
        if "indicators" in subelements and element["indicators"]:
            indicators_dict[re.sub("refinements>", "",current)] = (map(create_regex_format, element["indicators"]))
        if len(subelements) > 1:
            for subelement in subelements:
                if type(element[subelement]) == dict:
                    indicators_dict = find_indicators(
                        element[subelement], indicators_dict, current = current + ">" + subelement
                    )
    return indicators_dict

def match_indicators(lines, indicators_dict):
    found = defaultdict(list)
    for sentNum, sentence in enumerate(nltk.sent_tokenize(lines)):
        for group in indicators_dict:
            matches = []
            for n, indicator in enumerate(indicators_dict[group]):
                padded = " " + sentence.lower().strip() + " "
                if indicator.search(padded):
                    nuggets = indicator.findall(padded)
                    matches.append(re.sub(r"^.*?\]\((.*?)\)\[.*?$",r"\1",indicator.pattern))
            if matches:
                found[group].append((matches, str(sentNum)))
    return dict(found)

def extract_matches(article_ID, root_dir = "/data/ClinicalTrialsWikipediaArticles/articles/"):
    schema = json.load(open("../structured-output-schema.json", "r"))
    indicators_dict = find_indicators(schema)

    article_text = read_article(article_ID, root_dir)

    found = match_indicators(article_text, indicators_dict)
    return found

def compound_results(results, compounded):
    article_ID, found = results
    for group in found:
        for indicators, sentID in found[group]:
            for indicator in indicators:
                compounded[group][indicator].append(article_ID + "." + sentID)
    return compounded

def build_compounded(processed):
    compounded = defaultdict(lambda : defaultdict(list))

    for results in processed:
        compounded = compound_results(results, compounded)

    compounded = {k : dict(compounded[k]) for k in compounded}
    return compounded

# control code

root_dir = "/data/ClinicalTrialsWikipediaArticles/articles/"
article_IDs = open("../article_IDs.txt", "r").readlines()
article_IDs = [article_ID.strip() for article_ID in article_IDs]
start = time.time()
total_time = 0
processed = []
count = 0
batchsize = 10000
batch_article_IDs = []

for article_ID in article_IDs:
    batch_article_IDs.append(article_ID)

    if not (count + 1) % batchsize:
        processed.extend(
            sc.parallelize(batch_article_IDs, 15).map(lambda x: (x, extract_matches(x, root_dir = root_dir))).collect()
        )
        batch_article_IDs = []

        print "Right now, we're " + str(100 * (count + 1) / len(article_IDs)) + "%% done."
        end = time.time()
        print "This batch of " + str(batchsize) + " took " + str(end - start) + " seconds."
        print "The rest should take about " + str((len(article_IDs) - (count + 1)) * (end - start) / (60 * batchsize)) + " minutes."
        print ""
        total_time += len(article_IDs) * (end - start) / (60 * batchsize)
        start = time.time()

    count += 1

if batch_article_IDs:
    processed.extend(
        sc.parallelize(batch_article_IDs, 15).map(lambda x: (x, extract_matches(x, root_dir = root_dir))).collect()
    )

end = time.time()

# total_time += len(article_IDs) * (end - start) / (60 * batchsize)
# print "This whole process took " + str(total_time) + " minutes."
# print ""

json.dump(processed, open("../data/processed_articles.json", "w"))

compounded = build_compounded(processed)

json.dump(compounded, open("../data/compounded_articles.json", "w"))
