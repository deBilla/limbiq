"""Tests for the web search client and uncertainty detection."""

import json
from unittest.mock import patch, MagicMock
from urllib.error import URLError

import pytest

from limbiq.search import SearchClient, SearchResult, detect_uncertainty


# ─── Uncertainty Detection ─────────────────────────────────

class TestDetectUncertainty:
    def test_detects_dont_know(self):
        assert detect_uncertainty("I don't know the answer to that question.")

    def test_detects_not_sure(self):
        assert detect_uncertainty("I'm not sure about this, but...")

    def test_detects_knowledge_cutoff(self):
        assert detect_uncertainty("As of my last training data, I cannot confirm.")

    def test_detects_no_access(self):
        assert detect_uncertainty("I don't have access to real-time information.")

    def test_detects_cannot_browse(self):
        assert detect_uncertainty("I can't browse the internet to look that up.")

    def test_no_false_positive_on_confident(self):
        assert not detect_uncertainty("The capital of France is Paris.")

    def test_no_false_positive_on_detailed(self):
        assert not detect_uncertainty(
            "SpaceX launched Starship on March 14, 2025. The mission was a success."
        )

    def test_case_insensitive(self):
        assert detect_uncertainty("I DON'T KNOW the answer.")

    def test_empty_string(self):
        assert not detect_uncertainty("")


# ─── SearchResult ──────────────────────────────────────────

class TestSearchResult:
    def test_creation(self):
        sr = SearchResult(
            title="Test", url="https://example.com", snippet="A snippet", source="example.com"
        )
        assert sr.title == "Test"
        assert sr.url == "https://example.com"
        assert sr.snippet == "A snippet"
        assert sr.source == "example.com"

    def test_default_source(self):
        sr = SearchResult(title="T", url="https://x.com", snippet="S")
        assert sr.source == ""


# ─── SearchClient Init ────────────────────────────────────

class TestSearchClientInit:
    def test_default_provider(self):
        client = SearchClient(base_url="http://localhost:8888")
        assert client.provider == "searxng"

    def test_custom_provider(self):
        client = SearchClient(base_url="http://x.com", provider="brave")
        assert client.provider == "brave"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            SearchClient(base_url="http://x.com", provider="bing")

    def test_strips_trailing_slash(self):
        client = SearchClient(base_url="http://localhost:8888/")
        assert client.base_url == "http://localhost:8888"

    def test_repr(self):
        client = SearchClient(base_url="http://localhost:8888", provider="searxng")
        assert "searxng" in repr(client)


# ─── SearXNG Provider ────────────────────────────────────

class TestSearXNGSearch:
    def _mock_searxng_response(self):
        return json.dumps({
            "results": [
                {
                    "title": "SpaceX Launch Update",
                    "url": "https://spacex.com/launches",
                    "content": "SpaceX launched Starship successfully.",
                },
                {
                    "title": "NASA Coverage",
                    "url": "https://nasa.gov/spacex",
                    "content": "NASA provided live coverage of the launch.",
                },
            ]
        }).encode("utf-8")

    @patch("limbiq.search.urlopen")
    def test_searxng_search(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_searxng_response()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = SearchClient(base_url="http://localhost:8888", provider="searxng")
        results = client("SpaceX launch")

        assert len(results) == 2
        assert results[0].title == "SpaceX Launch Update"
        assert results[0].source == "spacex.com"
        assert "Starship" in results[0].snippet

    @patch("limbiq.search.urlopen")
    def test_max_results_cap(self, mock_urlopen):
        many_results = {
            "results": [
                {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Snippet {i}"}
                for i in range(20)
            ]
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(many_results).encode("utf-8")
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = SearchClient(base_url="http://localhost:8888", max_results=3)
        results = client("test")
        assert len(results) == 3


# ─── Brave Provider ──────────────────────────────────────

class TestBraveSearch:
    @patch("limbiq.search.urlopen")
    def test_brave_search(self, mock_urlopen):
        brave_response = json.dumps({
            "web": {
                "results": [
                    {
                        "title": "Brave Result",
                        "url": "https://brave.com/result",
                        "description": "A brave search result.",
                    }
                ]
            }
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = brave_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = SearchClient(
            base_url="https://api.search.brave.com",
            provider="brave",
            api_key="test-key",
        )
        results = client("test query")

        assert len(results) == 1
        assert results[0].title == "Brave Result"
        assert results[0].source == "brave.com"


# ─── Tavily Provider ─────────────────────────────────────

class TestTavilySearch:
    @patch("limbiq.search.urlopen")
    def test_tavily_search(self, mock_urlopen):
        tavily_response = json.dumps({
            "results": [
                {
                    "title": "Tavily Result",
                    "url": "https://tavily.com/result",
                    "content": "Found via Tavily.",
                }
            ]
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = tavily_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        client = SearchClient(
            base_url="https://api.tavily.com",
            provider="tavily",
            api_key="tvly-test",
        )
        results = client("test query")

        assert len(results) == 1
        assert results[0].title == "Tavily Result"


# ─── Error Handling ──────────────────────────────────────

class TestSearchErrors:
    @patch("limbiq.search.urlopen")
    def test_connection_error_propagates(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused")

        client = SearchClient(base_url="http://localhost:9999")
        with pytest.raises(Exception):
            client("test")

    @patch("limbiq.search.urlopen")
    def test_is_available_false_on_error(self, mock_urlopen):
        mock_urlopen.side_effect = URLError("Connection refused")

        client = SearchClient(base_url="http://localhost:9999")
        assert client.is_available() is False


# ─── Context Builder Integration ─────────────────────────

class TestContextBuilderWithSearch:
    def test_search_results_in_context(self):
        from limbiq.context.builder import ContextBuilder

        builder = ContextBuilder()
        search_results = [
            SearchResult(
                title="SpaceX Launch",
                url="https://spacex.com/news",
                snippet="Starship launched successfully on March 14.",
                source="spacex.com",
            ),
        ]

        context = builder.build(
            priority_memories=[],
            relevant_memories=[],
            suppressed_ids=set(),
            search_results=search_results,
        )

        assert "WEB SEARCH" in context
        assert "SpaceX Launch" in context
        assert "spacex.com" in context

    def test_no_search_section_when_empty(self):
        from limbiq.context.builder import ContextBuilder

        builder = ContextBuilder()
        context = builder.build(
            priority_memories=[],
            relevant_memories=[],
            suppressed_ids=set(),
            search_results=None,
        )
        assert context == ""

    def test_search_results_capped_at_3(self):
        from limbiq.context.builder import ContextBuilder

        builder = ContextBuilder()
        search_results = [
            SearchResult(title=f"R{i}", url=f"https://x.com/{i}", snippet=f"S{i}")
            for i in range(10)
        ]

        context = builder.build(
            priority_memories=[],
            relevant_memories=[],
            suppressed_ids=set(),
            search_results=search_results,
        )

        # Should only have 3 results (domain extracted as x.com)
        assert context.count("(x.com)") == 3
