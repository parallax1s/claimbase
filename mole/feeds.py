from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from typing import Any
from xml.etree import ElementTree as ET

from email.utils import parsedate_to_datetime


USER_AGENT = "claimbase-mole/0.1 (+github.com/parallax1s/claimbase)"
REQUEST_TIMEOUT = 30


def fetch_all(feeds_config: dict, since: str) -> list[dict]:
    items, _ = fetch_all_with_warnings(feeds_config, since)
    return items


def fetch_all_with_warnings(feeds_config: dict, since: str) -> tuple[list[dict], list[str]]:
    since_dt = _parse_datetime(since)
    feeds = feeds_config.get("feeds", []) if isinstance(feeds_config, dict) else []
    all_items: list[dict] = []
    warnings: list[str] = []

    for feed in feeds:
        try:
            kind = feed.get("kind")
            if kind == "graphql-forum":
                items = _fetch_graphql_forum(feed, since_dt)
            elif kind == "arxiv":
                items = _fetch_arxiv(feed, since_dt)
            elif kind == "rss":
                items = _fetch_rss(feed, since_dt)
            else:
                raise ValueError(f"unsupported feed kind: {kind}")

            all_items.extend(items)
        except Exception as exc:
            key = feed.get("key", feed.get("url", "unknown"))
            warnings.append(f"{key}: {exc}")

    return all_items, warnings


def _fetch_graphql_forum(feed: dict[str, Any], since_dt: datetime) -> list[dict]:
    limit = int(feed.get("limit", 10))
    query = (
        "query {\n"
        "  posts(input: { view: \"%s\", limit: %d }) {\n"
        "    results {\n"
        "      title\n"
        "      pageUrl\n"
        "      postedAt\n"
        "      user { displayName }\n"
        "      htmlBody\n"
        "    }\n"
        "  }\n"
        "}\n"
        % (feed.get("view", "new"), limit)
    )

    payload = json.dumps({"query": query}).encode("utf-8")
    response = _fetch_json(feed["url"], payload, headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    })

    results = (
        response.get("data", {})
        .get("posts", {})
        .get("results", [])
    )
    if not isinstance(results, list):
        return []

    items: list[dict] = []
    for post in results[:limit]:
        item = _coerce_item(
            feed=feed["key"],
            item_key=post.get("pageUrl"),
            title=post.get("title", ""),
            author=(post.get("user") or {}).get("displayName", ""),
            published=_parse_datetime(post.get("postedAt", "")),
            since_dt=since_dt,
            url=post.get("pageUrl"),
            raw_text=post.get("htmlBody", ""),
            force_text=True,
        )
        if item is not None:
            items.append(item)
    return items


def _fetch_arxiv(feed: dict[str, Any], since_dt: datetime) -> list[dict]:
    limit = int(feed.get("limit", 10))
    query = urllib.parse.urlencode(
        {
            "search_query": feed["query"],
            "sortBy": "submittedDate",
            "max_results": str(limit),
        }
    )
    url = f"http://export.arxiv.org/api/query?{query}"
    xml_text = _fetch_text(url, headers={"User-Agent": USER_AGENT})
    root = ET.fromstring(xml_text)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", namespace)
    if not entries:
        entries = [node for node in root.iter() if _localname(node.tag) == "entry"]

    items: list[dict] = []
    for entry in entries[:limit]:
        title = _first_child_text(entry, ["title"], default="")
        abstract = _first_child_text(entry, ["summary"], default="")
        url = _first_child_text(entry, ["id"], default="")
        author = _first_child_text(entry, ["name"], default="")
        if not author:
            author = _first_child_text(entry, ["author"], default="")
        published_raw = _first_child_text(entry, ["published"], default="")
        item_text = (
            f"{title}\n\n{_strip_html(str(abstract))}".strip()
            if abstract or title
            else _strip_html(str(abstract))
        )
        item = _coerce_item(
            feed=feed["key"],
            item_key=url,
            title=title,
            author=author,
            published=_parse_datetime(published_raw),
            since_dt=since_dt,
            url=url,
            raw_text=item_text,
        )
        if item is not None:
            items.append(item)
    return items


def _fetch_rss(feed: dict[str, Any], since_dt: datetime) -> list[dict]:
    limit = int(feed.get("limit", 10))
    limit = max(0, limit)
    xml_text = _fetch_text(feed["url"], headers={"User-Agent": USER_AGENT})
    root = ET.fromstring(xml_text)

    candidates: list[ET.Element] = []
    for node in root.iter():
        if _localname(node.tag) in {"item", "entry"}:
            candidates.append(node)
    if not candidates:
        return []

    items: list[dict] = []
    for node in candidates[:limit]:
        title = _first_child_text(node, ["title"], default="")
        if _localname(node.tag) == "entry":
            url = _extract_atom_link(node) or _first_child_text(node, ["link"], default="")
            author = _first_child_text(node, ["name"], default="")
            if not author:
                author = _first_child_text(node, ["author"], default="")
            published_raw = _first_child_text(node, ["published", "updated"], default="")
        else:
            url = _first_child_text(node, ["link"], default="")
            author = _first_child_text(
                node,
                ["creator", "name", "author"],
                default="",
                namespace_uris={"http://purl.org/dc/elements/1.1/"},
            )
            published_raw = _first_child_text(node, ["pubDate", "published", "updated"], default="")

        raw_text = _extract_rss_text(node)
        item = _coerce_item(
            feed=feed["key"],
            item_key=url,
            title=title,
            author=author,
            published=_parse_datetime(published_raw),
            since_dt=since_dt,
            url=url,
            raw_text=raw_text,
        )
        if item is not None:
            items.append(item)
    return items


def _coerce_item(
    feed: str,
    item_key: str,
    title: str,
    author: str,
    published: datetime,
    since_dt: datetime,
    url: str,
    raw_text: str,
    force_text: bool = False,
) -> dict[str, Any] | None:
    if published < since_dt:
        return None

    cleaned_text = _strip_html(raw_text)
    if force_text and not cleaned_text:
        cleaned_text = _strip_html(str(raw_text))

    content_bytes = cleaned_text.encode("utf-8")
    return {
            "feed": feed,
            "item_key": item_key or _stable_key(feed, title, url),
            "url": url,
        "title": title,
        "author": author,
        "published": _normalize_dt(published),
        "text": cleaned_text,
        "content_sha256": hashlib.sha256(content_bytes).hexdigest(),
    }


def _stable_key(feed: str, title: str, url: str) -> str:
    return hashlib.sha256(f"{feed}::{url}::{title}".encode("utf-8")).hexdigest()


def _fetch_text(url: str, *, headers: dict[str, str] | None = None) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        return response.read().decode("utf-8", errors="replace")


def _fetch_json(url: str, data: bytes, *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": USER_AGENT, **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        payload = response.read().decode("utf-8", errors="replace")
    return json.loads(payload)


def _parse_datetime(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return datetime.fromisoformat(text + "T00:00:00+00:00")

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        return datetime.now(timezone.utc)


def _normalize_dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _first_child_text(
    node: ET.Element,
    names: list[str],
    *,
    default: str = "",
    namespace_uris: set[str] | None = None,
) -> str:
    targets = set(names)
    for child in node:
        child_name = _localname(child.tag)
        if child_name in targets:
            text = _collapse_text(child)
            if text:
                return text

    for child in node.iter():
        if child is node:
            continue
        child_name = _localname(child.tag)
        if child_name in targets:
            text = _collapse_text(child)
            if text:
                return text

    if namespace_uris:
        for uri in namespace_uris:
            for name in names:
                element = node.find(f"{{{uri}}}{name}")
                if element is not None:
                    text = _collapse_text(element)
                    if text:
                        return text

    return default


def _collapse_text(element: ET.Element) -> str:
    return "".join(element.itertext()).strip()


def _extract_rss_text(node: ET.Element) -> str:
    for preferred in {"encoded", "content"}:
        for child in node.iter():
            if child is node:
                continue
            if _localname(child.tag) == preferred:
                text = _strip_html(ET.tostring(child, encoding="unicode", method="html"))
                if text:
                    return text

    for child in node.iter():
        if child is node:
            continue
        if _localname(child.tag) in {"description", "summary"}:
            text = _strip_html(ET.tostring(child, encoding="unicode", method="html"))
            if text:
                return text

    return _strip_html(ET.tostring(node, encoding="unicode", method="html"))


def _extract_atom_link(node: ET.Element) -> str:
    for child in node.findall("link"):
        href = child.attrib.get("href")
        if href:
            return href
        text = (child.text or "").strip()
        if text:
            return text
    for child in list(node):
        if _localname(child.tag) == "link":
            href = child.attrib.get("href")
            if href:
                return href
    return ""


def _localname(tag: str) -> str:
    if tag.startswith("{"):
        return tag.rsplit("}", 1)[1]
    return tag


def _strip_html(html_text: str) -> str:
    parser = _HTMLToText()
    parser.feed(html_text or "")
    parser.close()
    text = parser.value()
    text = unescape(text)
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class _HTMLToText(HTMLParser):
    _BREAK_TAGS = {"p", "div", "section", "article", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "pre"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._BREAK_TAGS and tag.lower() != "br":
            self._parts.append("\n\n")
        if tag.lower() == "br":
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._BREAK_TAGS and tag.lower() != "br":
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self._parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._parts.append(f"&#{name};")

    def value(self) -> str:
        return "".join(self._parts)
