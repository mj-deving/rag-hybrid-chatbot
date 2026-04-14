"""In-memory knowledge graph backed by NetworkX with JSON persistence."""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

import networkx as nx

GRAPH_PATH = os.environ.get(
    "GRAPH_PATH", str(Path(__file__).parent.parent / "data" / "graph.json")
)

_graph: "KnowledgeGraph | None" = None


@dataclass
class Entity:
    name: str
    type: str
    document_id: str


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    document_id: str


class KnowledgeGraph:
    """Directed graph storing entities as nodes and relations as edges."""

    def __init__(self, path: str = GRAPH_PATH):
        self._path = path
        self._graph = nx.DiGraph()
        self._load()

    def _load(self):
        """Load graph from JSON file if it exists."""
        if Path(self._path).exists():
            with open(self._path) as f:
                data = json.load(f)
            for node in data.get("nodes", []):
                self._graph.add_node(node["name"], type=node["type"], document_id=node["document_id"])
            for edge in data.get("edges", []):
                self._graph.add_edge(
                    edge["subject"], edge["object"],
                    predicate=edge["predicate"], document_id=edge["document_id"],
                )

    def _save(self):
        """Persist graph to JSON file."""
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        nodes = [
            {"name": n, "type": d.get("type", ""), "document_id": d.get("document_id", "")}
            for n, d in self._graph.nodes(data=True)
        ]
        edges = [
            {"subject": u, "object": v, "predicate": d.get("predicate", ""), "document_id": d.get("document_id", "")}
            for u, v, d in self._graph.edges(data=True)
        ]
        with open(self._path, "w") as f:
            json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=2)

    def add_entities(self, entities: list[Entity]):
        """Add entity nodes to the graph."""
        for e in entities:
            self._graph.add_node(e.name, type=e.type, document_id=e.document_id)
        if entities:
            self._save()

    def add_relations(self, relations: list[Relation]):
        """Add relation edges to the graph. Creates nodes if they don't exist."""
        for r in relations:
            if not self._graph.has_node(r.subject):
                self._graph.add_node(r.subject, type="unknown", document_id=r.document_id)
            if not self._graph.has_node(r.object):
                self._graph.add_node(r.object, type="unknown", document_id=r.document_id)
            self._graph.add_edge(
                r.subject, r.object,
                predicate=r.predicate, document_id=r.document_id,
            )
        if relations:
            self._save()

    def query_entity(self, name: str) -> dict | None:
        """Return entity info and its direct neighbors, or None if not found."""
        if not self._graph.has_node(name):
            return None
        node_data = self._graph.nodes[name]
        neighbors = []
        for _, target, data in self._graph.out_edges(name, data=True):
            neighbors.append({"entity": target, "relation": data.get("predicate", "")})
        for source, _, data in self._graph.in_edges(name, data=True):
            neighbors.append({"entity": source, "relation": data.get("predicate", "")})
        return {
            "name": name,
            "type": node_data.get("type", ""),
            "document_id": node_data.get("document_id", ""),
            "neighbors": neighbors,
        }

    def query_relations(self, entity_a: str, entity_b: str) -> list[dict]:
        """Return all paths between two entities (up to length 3)."""
        if not self._graph.has_node(entity_a) or not self._graph.has_node(entity_b):
            return []
        undirected = self._graph.to_undirected()
        paths = []
        try:
            for path in nx.all_simple_paths(undirected, entity_a, entity_b, cutoff=3):
                path_info = []
                for i in range(len(path) - 1):
                    edge_data = self._graph.get_edge_data(path[i], path[i + 1])
                    if edge_data is None:
                        edge_data = self._graph.get_edge_data(path[i + 1], path[i])
                    predicate = edge_data.get("predicate", "related_to") if edge_data else "related_to"
                    path_info.append({"from": path[i], "relation": predicate, "to": path[i + 1]})
                paths.append(path_info)
        except nx.NetworkXNoPath:
            pass
        return paths

    def get_subgraph(self, document_id: str) -> dict:
        """Return all entities and relations belonging to a document."""
        entities = [
            {"name": n, "type": d.get("type", "")}
            for n, d in self._graph.nodes(data=True)
            if d.get("document_id") == document_id
        ]
        relations = [
            {"subject": u, "predicate": d.get("predicate", ""), "object": v}
            for u, v, d in self._graph.edges(data=True)
            if d.get("document_id") == document_id
        ]
        return {"entities": entities, "relations": relations}

    def find_relevant_entities(self, entity_names: list[str]) -> list[dict]:
        """Find entities matching given names and their neighborhoods."""
        results = []
        seen = set()
        for name in entity_names:
            # Try exact match first
            if self._graph.has_node(name):
                info = self.query_entity(name)
                if info and name not in seen:
                    seen.add(name)
                    results.append(info)
                continue
            # Try case-insensitive substring match
            for node in self._graph.nodes():
                if name.lower() in node.lower() and node not in seen:
                    seen.add(node)
                    info = self.query_entity(node)
                    if info:
                        results.append(info)
        return results

    def delete_document(self, document_id: str):
        """Remove all nodes and edges belonging to a document."""
        edges_to_remove = [
            (u, v) for u, v, d in self._graph.edges(data=True)
            if d.get("document_id") == document_id
        ]
        self._graph.remove_edges_from(edges_to_remove)
        nodes_to_remove = [
            n for n, d in self._graph.nodes(data=True)
            if d.get("document_id") == document_id
        ]
        self._graph.remove_nodes_from(nodes_to_remove)
        self._save()

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()


def get_graph() -> KnowledgeGraph:
    """Get or initialize the knowledge graph singleton."""
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph()
    return _graph
