from typing import Dict, Any

from ui.UserInputManager import UserInputManager
from utils.MovieSearcher import MovieSearcher
from utils.OntologyManager import OntologyManager

FILEPATH = "resources/ontology.rdf"


def prepare_search_args(criteria: dict) -> Dict[str, Any]:
    args = {}

    if "genres" in criteria and criteria["genres"] is not None:
        args["genres"] = list(criteria["genres"])

    if "age" in criteria and criteria["age"] is not None:
        args["age_limit"] = int(criteria["age"])

    if "country" in criteria and criteria["country"] is not None:
        args["country"] = str(criteria["country"])

    if "rating" in criteria and criteria["rating"] is not None:
        args["rating"] = float(criteria["rating"])

    if "release_year" in criteria and criteria["release_year"] is not None:
        args["release_year"] = int(criteria["release_year"])

    if "title" in criteria and criteria["title"] is not None:
        args["title"] = str(criteria["title"])

    if "director" in criteria and criteria["director"] is not None:
        args["director_uri"] = str(criteria["director"])

    if "actor" in criteria and criteria["actor"] is not None:
        args["actor_uri"] = str(criteria["actor"])

    if "production_company" in criteria and criteria["production_company"] is not None:
        args["production_company_uri"] = str(criteria["production_company"])

    return args


def main():
    userInputManager = UserInputManager()
    ontology_manager = OntologyManager(FILEPATH)
    movie_searcher = MovieSearcher(ontology_manager.graph,
                                   ontology_manager.namespace,
                                   ontology_manager.get_by_type("Movie"))

    print("Введите запрос в формате: ")
    print('"Мне (int) лет/года. Мне нравятся жанры: (жанры через запятую на английском); страна производства: (английский); рейтинг: (float); годом производства: (int); название: (английский); режиссёр: (на английском); актёр: (на английском); кинокомпания: (на английском)"')

    while True:
        try:
            user_input = input("\nВведите запрос: ").strip()
            if not user_input:
                continue

            userInputManager.parse(user_input)
            criteria: Dict[str, Any] = userInputManager.get_criteria()


            print("Критерии пользователя:")
            for key, value in criteria.items():
                print(f"  {key}: {value}")

            prepared_args = prepare_search_args(criteria)
            movies = movie_searcher.search_movies(**prepared_args)

            if movies is not None:
                print("Вот список фильмов, сформированных на ваших критериях:")
                for movie in movies:
                    for movie_title in ontology_manager.graph.objects(movie, ontology_manager.namespace.hasTitle):
                        print(str(movie_title))
            else:
                print("В данный момент фильмы, соотвествующие вашим критериям отсутствуют.")

        except KeyboardInterrupt:
            print("\nКонец работы программы.")
            break
        except Exception as exception:
            print(f"Ошибка обработки запроса: {exception}")
            continue


if __name__ == "__main__":
    main()
