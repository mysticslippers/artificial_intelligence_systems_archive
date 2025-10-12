from typing import List, Iterable

from rdflib import Graph, URIRef, Namespace


class MovieSearcher:
    def __init__(self, graph: Graph, namespace: Namespace, collection: list):
        self.graph = graph
        self.namespace = namespace
        self.collection = collection

    def _uri(self, s: str) -> URIRef:
        return URIRef(str(self.namespace) + "".join(s.split()))

    def search_movies(self,
                      genres: Iterable[str] = None,
                      age_limit: int = None,
                      country: str = None,
                      rating: float = None,
                      release_year: int = None,
                      title: str = None,
                      director_uri: str = None,
                      actor_uri: str = None,
                      production_company_uri: str = None) -> List[URIRef]:
        result_set = set(self.collection)

        if genres is not None:
            required_genres = {URIRef(str(self.namespace) + str(genre).strip()) for genre in genres}
            filtered = set()
            for movie in result_set:
                movie_genres = set(self.graph.objects(movie, self.namespace.hasGenreIn))
                if required_genres.issubset(movie_genres):
                    filtered.add(movie)
            result_set = filtered

        if age_limit is not None:
            filtered = set()
            for movie in result_set:
                for al in self.graph.objects(movie, self.namespace.hasAgeLimit):
                    try:
                        if int(al) <= int(age_limit):
                            filtered.add(movie)
                            break
                    except (TypeError, ValueError):
                        continue
            result_set = filtered

        if country is not None:
            tmp = country.lower()
            filtered = set()
            for movie in result_set:
                for movie_country in self.graph.objects(movie, self.namespace.hasCountry):
                    country_str = str(movie_country).lower() if not isinstance(movie_country, str) else movie_country.lower()
                    if tmp in country_str:
                        filtered.add(movie)
                        break
            result_set = filtered

        if rating is not None:
            filtered = set()
            for movie in result_set:
                for r in self.graph.objects(movie, self.namespace.hasRating):
                    try:
                        if float(r) >= float(rating):
                            filtered.add(movie)
                            break
                    except (TypeError, ValueError):
                        continue
            result_set = filtered

        if release_year is not None:
            filtered = set()
            for movie in result_set:
                for r_year in self.graph.objects(movie, self.namespace.hasReleaseYear):
                    try:
                        if int(r_year) == int(release_year):
                            filtered.add(movie)
                            break
                    except (TypeError, ValueError):
                        continue
            result_set = filtered

        if title is not None:
            tmp = title.lower()
            filtered = set()
            for movie in result_set:
                for movie_title in self.graph.objects(movie, self.namespace.hasTitle):
                    title_str = str(movie_title).lower() if not isinstance(movie_title, str) else movie_title.lower()
                    if tmp in title_str:
                        filtered.add(movie)
                        break
            result_set = filtered

        if director_uri is not None:
            director = self._uri(director_uri)
            filtered = set()
            for movie in result_set:
                for directed_in in self.graph.objects(director, self.namespace.hasDirectedIn):
                    if directed_in == movie:
                        filtered.add(movie)
                        break
            result_set = filtered

        if actor_uri is not None:
            actor = self._uri(actor_uri)
            filtered = set()
            for movie in result_set:
                for acted_in in self.graph.objects(actor, self.namespace.hasActedIn):
                    if acted_in == movie:
                        filtered.add(movie)
                        break
            result_set = filtered

        if production_company_uri is not None:
            production_company = self._uri(production_company_uri)
            filtered = set()
            for movie in result_set:
                for producedIn in self.graph.objects(production_company, self.namespace.hasProducedIn):
                    if producedIn == movie:
                        filtered.add(movie)
                        break
            result_set = filtered

        return list(result_set)
