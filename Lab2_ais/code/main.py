from utils.MovieSearcher import MovieSearcher
from utils.OntologyManager import OntologyManager

FILEPATH = "resources/ontology.rdf"

if __name__ == "__main__":
    ontology_manager = OntologyManager(FILEPATH)
    movie_searcher = MovieSearcher(ontology_manager.graph, ontology_manager.namespace, ontology_manager.get_by_type("Movie"))
