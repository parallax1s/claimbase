import hashlib
import json

from urllib.error import URLError

import mole.feeds as feeds


class _FakeResponse:
    def __init__(self, body: str):
        self._body = body.encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self._body


def _make_urlopen(fake_body_map):
    calls = []

    def _urlopen(request, timeout=30):
        calls.append(request)
        if isinstance(fake_body_map, Exception):
            raise fake_body_map
        if callable(fake_body_map):
            body = fake_body_map(request)
        elif isinstance(fake_body_map, list):
            body = fake_body_map.pop(0)
        else:
            body = fake_body_map

        if isinstance(body, Exception):
            raise body
        return _FakeResponse(body)

    _urlopen.calls = calls
    return _urlopen


def test_graphql_feed_parsing_and_since_filter(monkeypatch):
    graphql_payload = {
        "data": {
            "posts": {
                "results": [
                    {
                        "title": "Old forum post",
                        "pageUrl": "https://example.com/old",
                        "postedAt": "2026-05-20T10:00:00Z",
                        "user": {"displayName": "Forum Author"},
                        "htmlBody": "<p>Should not be visible</p>",
                    },
                    {
                        "title": "New forum post",
                        "pageUrl": "https://example.com/new",
                        "postedAt": "2026-06-02T12:00:00Z",
                        "user": {"displayName": "Recent Author"},
                        "htmlBody": "<p>First paragraph.</p><p>Second paragraph.</p>",
                    },
                ]
            }
        }
    }

    monkeypatch.setattr(
        "urllib.request.urlopen",
        _make_urlopen(json.dumps(graphql_payload)),
    )

    items, warnings = feeds.fetch_all_with_warnings(
        {
            "feeds": [
                {
                    "key": "lesswrong-af",
                    "kind": "graphql-forum",
                    "url": "https://www.alignmentforum.org/graphql",
                    "view": "new",
                    "limit": 25,
                }
            ]
        },
        "2026-06-01",
    )

    assert warnings == []
    assert len(items) == 1
    assert items[0]["item_key"] == "https://example.com/new"
    assert items[0]["url"] == "https://example.com/new"
    assert items[0]["title"] == "New forum post"
    assert items[0]["author"] == "Recent Author"
    assert items[0]["published"] == "2026-06-02T12:00:00+00:00"
    assert items[0]["text"] == "First paragraph.\n\nSecond paragraph."
    expected = hashlib.sha256(items[0]["text"].encode("utf-8")).hexdigest()
    assert items[0]["content_sha256"] == expected


def test_arxiv_feed_parses_title_plus_abstract(monkeypatch):
    arxiv_xml = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>First</title>
    <summary>Old abstract should be filtered</summary>
    <id>http://arxiv.org/abs/old.0001</id>
    <published>2026-05-20T00:00:00Z</published>
    <author><name>Old Author</name></author>
  </entry>
  <entry>
    <title>Boundary case</title>
    <summary>Abstract keeps focus.</summary>
    <id>http://arxiv.org/abs/new.0002</id>
    <published>2026-06-03T00:00:00Z</published>
    <author><name>Alex</name></author>
  </entry>
</feed>
"""

    fake_urlopen = _make_urlopen(arxiv_xml)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    items, warnings = feeds.fetch_all_with_warnings(
        {
            "feeds": [
                {
                    "key": "arxiv-csai",
                    "kind": "arxiv",
                    "query": "cat:cs.AI",
                    "limit": 10,
                }
            ]
        },
        "2026-06-01",
    )

    assert warnings == []
    assert len(items) == 1
    assert items[0]["feed"] == "arxiv-csai"
    assert items[0]["url"] == "http://arxiv.org/abs/new.0002"
    assert items[0]["author"] == "Alex"
    assert items[0]["text"] == "Boundary case\n\nAbstract keeps focus."
    assert items[0]["item_key"] == "http://arxiv.org/abs/new.0002"
    assert items[0]["published"] == "2026-06-03T00:00:00+00:00"
    expected = hashlib.sha256(items[0]["text"].encode("utf-8")).hexdigest()
    assert items[0]["content_sha256"] == expected
    assert "sortBy=submittedDate" in str(fake_urlopen.calls[0].full_url)


def test_rss_feed_prefers_content_encoded_and_strips_html(monkeypatch):
    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
<rss>
  <channel>
    <item>
      <title>RSS item</title>
      <link>https://example.com/rss</link>
      <content:encoded xmlns:content="http://purl.org/rss/1.0/modules/content/">
        <p>From encoded</p><p>still preferred</p>
      </content:encoded>
      <description><p>Fallback description</p></description>
      <pubDate>Tue, 02 Jun 2026 10:00:00 +0000</pubDate>
      <dc:creator xmlns:dc="http://purl.org/dc/elements/1.1/">RSS Author</dc:creator>
    </item>
  </channel>
</rss>
"""

    monkeypatch.setattr("urllib.request.urlopen", _make_urlopen(rss_xml))

    items, warnings = feeds.fetch_all_with_warnings(
        {
            "feeds": [
                {
                    "key": "blog-rss",
                    "kind": "rss",
                    "url": "https://thezvi.substack.com/feed",
                    "limit": 10,
                }
            ]
        },
        "2026-06-01",
    )

    assert warnings == []
    assert len(items) == 1
    assert items[0]["feed"] == "blog-rss"
    assert items[0]["url"] == "https://example.com/rss"
    assert items[0]["author"] == "RSS Author"
    assert items[0]["text"] == "From encoded\n\nstill preferred"
    assert items[0]["published"] == "2026-06-02T10:00:00+00:00"


def test_dead_feed_records_warning_and_continues(monkeypatch):
    good_rss = """<?xml version='1.0'?>
<rss>
  <channel>
    <item>
      <title>Recovered item</title>
      <link>https://example.com/recovered</link>
      <description>Recovered feed entry.</description>
      <pubDate>2026-06-04T12:00:00Z</pubDate>
    </item>
  </channel>
</rss>
"""

    call_count = 0

    def fake_urlopen(request, timeout=30):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise URLError("server unavailable")
        return _FakeResponse(good_rss)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    items, warnings = feeds.fetch_all_with_warnings(
        {
            "feeds": [
                {
                    "key": "broken-feed",
                    "kind": "graphql-forum",
                    "url": "https://broken.example/graphql",
                    "view": "new",
                    "limit": 5,
                },
                {
                    "key": "good-rss",
                    "kind": "rss",
                    "url": "https://thezvi.substack.com/feed",
                    "limit": 5,
                },
            ]
        },
        "2026-06-01",
    )

    assert len(warnings) == 1
    assert warnings[0].startswith("broken-feed:")
    assert any(item["feed"] == "good-rss" for item in items)
    assert len(items) == 1
