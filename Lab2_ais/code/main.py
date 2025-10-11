from ui.UserInputManager import UserInputManager
from utils.MovieSearcher import MovieSearcher
from utils.OntologyManager import OntologyManager

FILEPATH = "resources/ontology.rdf"

if __name__ == "__main__":
    ontology_manager = OntologyManager(FILEPATH)
    movie_searcher = MovieSearcher(ontology_manager.graph, ontology_manager.namespace, ontology_manager.get_by_type("Movie"))
    userInputManager = UserInputManager()
    userInputManager.parse("Мне 24 года. Мне нравятся жанры: Fantasy, Action; страна производства: USA; с рейтинг: 6.0; с годом производства: 2001; с название: Pulp Fiction; с режиссёр: Christopher Nolan; с актёр: Cillian Murphy; с кинокомпания: A Band Apart")
    dictionary = userInputManager.get_criteria()
    keys = dictionary.keys()
    for i in keys:
        print(i, dictionary.get(i))
