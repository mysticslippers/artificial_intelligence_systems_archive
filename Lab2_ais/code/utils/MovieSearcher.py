from typing import List, Iterable

from rdflib import Graph, URIRef, Namespace


class MovieSearcher:
    def __init__(self, graph: Graph, namespace: Namespace, collection: list):
        self.graph = graph
        self.namespace = namespace
        self.collection = collection
