from typing import Iterable, List, Optional, Dict, Any
import re

class UserInputManager:
    def __init__(self) -> None:
        self.age: Optional[int] = None
        self.genres: Optional[Iterable[str]] = None
        self.country: Optional[str] = None
        self.rating: Optional[float] = None
        self.release_year: Optional[int] = None
        self.title: Optional[str] = None
        self.director: Optional[str] = None
        self.actor: Optional[str] = None
        self.production_company: Optional[str] = None

    def _parse_int(self, s: str) -> Optional[int]:
        try:
            return int(s)
        except (ValueError, TypeError):
            return None

    def _parse_float(self, s: str) -> Optional[float]:
        try:
            return float(s)
        except (ValueError, TypeError):
            return None

    def _split_items(self, s: str) -> List[str]:
        items = [it.strip() for it in s.split(',')]
        return [it for it in items if it]

    def _normalize_title(self, s: str) -> str:
        return s.strip()

    def parse(self, input_text: str) -> None:
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text должен быть непустой строкой")

        text = input_text.strip()

        m_age = re.search(r'\b(\d{1,3})\s*(?:лет|год|годов|года)?\b', text, flags=re.IGNORECASE)
        if m_age:
            val = self._parse_int(m_age.group(1))
            if val is not None:
                self.age = val

        m_genres = re.search(r'жанры\s*[:\-]\s*(.*?)(;|\n|$)', text, flags=re.IGNORECASE)
        if m_genres:
            raw = m_genres.group(1)
            items = self._split_items(raw)
            genres = [self._normalize_title(it) for it in items if it]
            self.genres = genres if genres else None
        else:
            m_genres = re.search(r'genres\s*[:\-]\s*(.*?)(;|\n|$)', text, flags=re.IGNORECASE)
            if m_genres:
                raw = m_genres.group(1)
                items = self._split_items(raw)
                genres = [self._normalize_title(it) for it in items if it]
                self.genres = genres if genres else None

        m_country = re.search(r'страна\s*производства\s*[:\-]\s*(.*?)(;|\n|$)', text, flags=re.IGNORECASE)
        if m_country:
            raw = m_country.group(1)
            self.country = self._normalize_title(raw) if raw.strip() else None
        else:
            m_country = re.search(r'country\s*[:\-]\s*(.*?)(;|\n|$)', text, flags=re.IGNORECASE)
            if m_country:
                raw = m_country.group(1)
                self.country = self._normalize_title(raw) if raw.strip() else None

        m_rating = re.search(r'(?:рейтинг|rating)\s*(?::|=|-?\s*)?\s*([+-]?\d+(?:\.\d+)?)', text, flags=re.IGNORECASE)
        if m_rating:
            self.rating = self._parse_float(m_rating.group(1))

        m_year = re.search(r'(?:годом производства|release year|production year|year of release)\s*[:\-]?\s*(\d{4})', text, flags=re.IGNORECASE)
        if m_year:
            self.release_year = self._parse_int(m_year.group(1))

        m_title = re.search(r'(?:название|title)\s*[:\-]\s*([^\n;]+)', text, flags=re.IGNORECASE)
        if m_title:
            self.title = self._normalize_title(m_title.group(1))

        m_director = re.search(r'(?:режиссёр|director)\s*[:\-]\s*([^\n;]+)', text, flags=re.IGNORECASE)
        if m_director is not None:
            line = "".join(self._normalize_title(m_director.group(1)).split())
            self.director = line

        m_actor = re.search(r'(?:актёр|actor)\s*[:\-]\s*([^\n;]+)', text, flags=re.IGNORECASE)
        if m_actor:
            line = "".join(self._normalize_title(m_actor.group(1)).split())
            self.actor = line

        m_production_company = re.search(r'(?:кинокомпания|production company)\s*[:\-]\s*([^\n;]+)', text, flags=re.IGNORECASE)
        if m_production_company:
            line = "".join(self._normalize_title(m_production_company.group(1)).split())
            self.production_company = line


        filled = any([
            self.age is not None,
            self.genres is not None,
            self.country is not None,
            self.rating is not None,
            self.release_year is not None,
            self.title is not None,
            self.director is not None,
            self.actor is not None,
            self.production_company is not None
        ])
        if not filled:
            raise ValueError("Должен быть введён хотя бы один критерий!")

    def get_criteria(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.age is not None:
            result['age'] = self.age
        if self.genres is not None:
            result['genres'] = list(self.genres)
        if self.country is not None:
            result['country'] = self.country
        if self.rating is not None:
            result['rating'] = self.rating
        if self.release_year is not None:
            result['release_year'] = self.release_year
        if self.title is not None:
            result['title'] = self.title
        if self.director is not None:
            result['director'] = self.director
        if self.actor is not None:
            result['actor'] = self.actor
        if self.production_company is not None:
            result['production_company'] = self.production_company
        return result