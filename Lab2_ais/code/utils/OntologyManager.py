from rdflib import Graph, Namespace, URIRef, RDF
from rdflib.namespace import NamespaceManager


class OntologyManager:
    def __init__(self, rdf_filepath: str):
        self.rdf_filepath = rdf_filepath
        self.graph = Graph()
        self._load_graph()

        self.namespace = Namespace("http://ontologies/movies.owl#")
        self.namespace_manager = NamespaceManager(self.graph)
        self.namespace_manager.bind("movies", self.namespace)

    def _load_graph(self):
        self.graph.parse(self.rdf_filepath, format="application/rdf+xml")

    def get_by_type(self, name_class: str) -> list:
        collection = list()
        objectClass = URIRef("http://ontologies/movies.owl#" + name_class)

        for subject in self.graph.subjects(RDF.type, objectClass):
            collection.append(subject)

        return collection
