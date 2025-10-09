from typing import List, Iterable

from rdflib import Graph, URIRef, Namespace


class MovieSearcher:
    def __init__(self, graph: Graph, namespace: Namespace, collection: list):
        self.graph = graph
        self.namespace = namespace
        self.collection = collection

    def search_movies(self,
                      genres: Iterable[str] = None,
                      age_limit: int = None,
                      country: str = None,
                      rating: float = None,
                      release_year: int = None,
                      title: str = None,
                      director_uri: str = None,
                      actor_uri: str = None) -> List[URIRef]:
        result: List[URIRef] = []


        if genres is not None:
            filtered = list()
            genres_uris = {URIRef(str(self.namespace) + str(genre)) for genre in genres}
            for movie in self.collection:
                movie_genres = set(self.graph.objects(movie, self.namespace.hasGenreIn))
                if genres_uris.issubset(movie_genres):
                    filtered.append(movie)
            result = filtered

        if age_limit is not None:
            filtered = list()
            for movie in self.collection:
                for al in self.graph.objects(movie, self.namespace.hasAgeLimit):
                    try:
                        if int(al) <= int(age_limit):
                            filtered.append(movie)
                            break
                    except (TypeError, ValueError):
                        continue
            result = filtered

        if country is not None:
            filtered = list()
            tmp = country.lower()
            for movie in self.collection:
                for movie_country in self.graph.objects(movie, self.namespace.hasCountry):
                    if isinstance(movie_country, str):
                        country_str = movie_country.lower()
                    else:
                        country_str = str(movie_country).lower()
                    if tmp in country_str:
                        filtered.append(movie)
                        break
            result = filtered

        if rating is not None:
            filtered = list()
            for movie in self.collection:
                for r in self.graph.objects(movie, self.namespace.hasRating):
                    try:
                        if float(r) >= float(rating):
                            filtered.append(movie)
                            break
                    except (TypeError, ValueError):
                        continue
            result = filtered

        if release_year is not None:
            filtered = list()
            for movie in self.collection:
                for r_year in self.graph.objects(movie, self.namespace.hasReleaseYear):
                    try:
                        if int(r_year) == int(release_year):
                            filtered.append(movie)
                            break
                    except (TypeError, ValueError):
                        continue
                result = filtered

        if title is not None:
            filtered = list()
            tmp = title.lower()
            for movie in self.collection:
                for movie_title in self.graph.objects(movie, self.namespace.hasTitle):
                    if isinstance(movie_title, str):
                        title_str = movie_title.lower()
                    else:
                        title_str = str(movie_title).lower()
                    if tmp in title_str:
                        filtered.append(movie)
                        break
            result = filtered

        if director_uri is not None:
            filtered = list()
            director = URIRef(str(self.namespace) + "".join(director_uri.split()))
            for movie in self.collection:
                for directed_in in self.graph.objects(director, self.namespace.hasDirectedIn):
                    if directed_in == movie:
                        filtered.append(movie)
                        break
            result = filtered

        if actor_uri is not None:
            filtered = list()
            actor = URIRef(str(self.namespace) + "".join(actor_uri.split()))
            for movie in self.collection:
                for acted_in in self.graph.objects(actor, self.namespace.hasActedIn):
                    if acted_in == movie:
                        filtered.append(movie)
                        break
            result = filtered

        return result
