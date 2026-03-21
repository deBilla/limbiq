"""
Generic web search client — unified interface for any search API.

Supports multiple providers out of the box:
  - SearXNG      (self-hosted, default)
  - Brave Search (https://api.search.brave.com)
  - Tavily       (https://api.tavily.com)
  - Google CSE   (https://www.googleapis.com/customsearch)

Usage:
    from limbiq.search import SearchClient

    search = SearchClient(base_url="http://localhost:8888", provider="searxng")
    results = search("latest SpaceX launch")
    for r in results:
        print(f"{r.title}: {r.snippet}")
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.request import Request, urlopen
from urllib.parse import urlencode, urlparse
from urllib.error import URLError

logger = logging.getLogger(__name__)


# ── Uncertainty detection ─────────────────────────────────

UNCERTAINTY_PHRASES = [
    "i don't know",
    "i don't have information",
    "i'm not sure",
    "i am not sure",
    "i cannot find",
    "i can't find",
    "i don't have access to",
    "my knowledge cutoff",
    "as of my last",
    "as of my training",
    "i'm unable to",
    "i am unable to",
    "i apologize, but i don't",
    "i don't have enough context",
    "i'm not aware",
    "i am not aware",
    "i don't have any information",
    "i cannot provide",
    "i can't provide",
    "beyond my knowledge",
    "outside my knowledge",
    "i don't have details",
    "i have no information",
    "i lack information",
    "i don't have data",
    "i don't have the latest",
    "i don't have real-time",
    "i don't have current",
    "i cannot access the internet",
    "i can't access the internet",
    "i can't browse",
    "i cannot browse",
]


def detect_uncertainty(text: str) -> bool:
    """
    Check if an LLM response indicates lack of knowledge.

    Returns True if the text contains known uncertainty phrases.
    """
    lower = text.lower()
    return any(phrase in lower for phrase in UNCERTAINTY_PHRASES)


# ── Search result ─────────────────────────────────────────

@dataclass
class SearchResult:
    """A single web search result."""
    title: str
    url: str
    snippet: str
    source: str = ""


# ── Search client ─────────────────────────────────────────

class SearchClient:
    """
    Lightweight web search client using only stdlib (no SDK dependencies).

    Works with any supported search provider via a unified interface.
    """

    PROVIDERS = ("searxng", "brave", "tavily", "google_cse")

    def __init__(
        self,
        base_url: str = "http://localhost:8888",
        provider: str = "searxng",
        api_key: Optional[str] = None,
        max_results: int = 5,
        timeout: int = 15,
    ):
        self.base_url = base_url.rstrip("/")
        self.provider = provider.lower()
        self.api_key = api_key
        self.max_results = max_results
        self.timeout = timeout

        if self.provider not in self.PROVIDERS:
            raise ValueError(
                f"Unknown provider '{self.provider}'. "
                f"Supported: {', '.join(self.PROVIDERS)}"
            )

    def __call__(self, query: str) -> list[SearchResult]:
        """Search and return results. Primary interface."""
        return self.search(query)

    def search(self, query: str) -> list[SearchResult]:
        """
        Execute a web search and return structured results.

        Args:
            query: The search query string.

        Returns:
            List of SearchResult objects.
        """
        dispatch = {
            "searxng": self._search_searxng,
            "brave": self._search_brave,
            "tavily": self._search_tavily,
            "google_cse": self._search_google_cse,
        }
        fn = dispatch[self.provider]
        try:
            results = fn(query)
            logger.info(f"Search [{self.provider}] '{query}' → {len(results)} results")
            return results[:self.max_results]
        except Exception as e:
            logger.error(f"Search failed [{self.provider}]: {e}")
            raise

    def is_available(self) -> bool:
        """Check if the search endpoint is reachable."""
        try:
            if self.provider == "searxng":
                url = f"{self.base_url}/healthz"
            elif self.provider == "brave":
                url = f"{self.base_url}/res/v1/web/search?q=test&count=1"
            elif self.provider == "tavily":
                # Tavily has no health endpoint; try a minimal search
                return self._try_request(f"{self.base_url}/search", method="POST",
                                         body={"query": "test", "max_results": 1})
            else:
                url = self.base_url

            return self._try_request(url)
        except Exception:
            return False

    # ── Provider implementations ──────────────────────────

    def _search_searxng(self, query: str) -> list[SearchResult]:
        """SearXNG: GET /search?q=...&format=json"""
        params = urlencode({
            "q": query,
            "format": "json",
            "categories": "general",
        })
        url = f"{self.base_url}/search?{params}"
        data = self._get_json(url)

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source=urlparse(item.get("url", "")).netloc,
            ))
        return results

    def _search_brave(self, query: str) -> list[SearchResult]:
        """Brave Search: GET /res/v1/web/search?q=..."""
        params = urlencode({
            "q": query,
            "count": self.max_results,
        })
        url = f"{self.base_url}/res/v1/web/search?{params}"
        headers = {}
        if self.api_key:
            headers["X-Subscription-Token"] = self.api_key

        data = self._get_json(url, headers=headers)

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
                source=urlparse(item.get("url", "")).netloc,
            ))
        return results

    def _search_tavily(self, query: str) -> list[SearchResult]:
        """Tavily: POST /search with JSON body."""
        url = f"{self.base_url}/search"
        payload = {
            "query": query,
            "max_results": self.max_results,
            "include_answer": False,
        }
        if self.api_key:
            payload["api_key"] = self.api_key

        data = self._post_json(url, payload)

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                source=urlparse(item.get("url", "")).netloc,
            ))
        return results

    def _search_google_cse(self, query: str) -> list[SearchResult]:
        """Google Custom Search: GET /customsearch/v1?q=..."""
        params = {
            "q": query,
            "num": min(self.max_results, 10),
        }
        if self.api_key:
            params["key"] = self.api_key

        url = f"{self.base_url}/customsearch/v1?{urlencode(params)}"
        data = self._get_json(url)

        results = []
        for item in data.get("items", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source=urlparse(item.get("link", "")).netloc,
            ))
        return results

    # ── HTTP helpers ──────────────────────────────────────

    def _get_json(self, url: str, headers: dict = None) -> dict:
        """GET request, return parsed JSON."""
        hdrs = {"Accept": "application/json"}
        if headers:
            hdrs.update(headers)
        req = Request(url, headers=hdrs)
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, url: str, payload: dict, headers: dict = None) -> dict:
        """POST JSON request, return parsed JSON."""
        hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            hdrs.update(headers)
        data = json.dumps(payload).encode("utf-8")
        req = Request(url, data=data, headers=hdrs, method="POST")
        with urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _try_request(self, url: str, method: str = "GET", body: dict = None) -> bool:
        """Try a request and return True if it succeeds."""
        try:
            if method == "POST" and body:
                data = json.dumps(body).encode("utf-8")
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                req = Request(url, data=data, headers=headers, method="POST")
            else:
                headers = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                req = Request(url, headers=headers)
            with urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def __repr__(self):
        return f"SearchClient(base_url={self.base_url!r}, provider={self.provider!r})"
