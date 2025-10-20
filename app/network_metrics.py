from __future__ import annotations
from typing import Dict, List, Tuple
import networkx as nx
from pprint import pformat

BALANCED   = {"+++", "+--", "-+-", "--+"}   # 積が + の三角形
UNBALANCED = {"++-", "+-+", "-++", "---"}   # 積が - の三角形

def _triads_with_struct(G: nx.Graph):
    """グラフ G の三角形を全探索し、構造（符号列）付きで返す"""
    triads = []
    nodes = list(G.nodes())
    n = len(nodes)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a,b,c = nodes[i], nodes[j], nodes[k]
                if G.has_edge(a,b) and G.has_edge(b,c) and G.has_edge(c,a):
                    sab = G[a][b]['score']; sbc = G[b][c]['score']; sca = G[c][a]['score']
                    struct = ('+' if sab>=0 else '-') + ('+' if sbc>=0 else '-') + ('+' if sca>=0 else '-')
                    triads.append(((a,b,c), struct))
    return triads  #　例：[ (('A','B','C'), '+-+'), (('A','C','D'), '+++'), ... ]

def analyze_relations_from_scores(scores: Dict[Tuple[str,str], float], include_nodes: List[str]|None=None, verbose: bool=False, return_trace: bool=False) -> Dict:
    """
    ペアスコア辞書 {(a,b): score} から、三角形構造・安定数・孤立などを返す。
    """
    G = nx.Graph()
    if include_nodes:
        for p in include_nodes:
            G.add_node(p)
    for (a,b), s in scores.items():
        G.add_edge(a,b, score=float(s))

    triads = _triads_with_struct(G)
    balanced_cnt   = sum(1 for _,s in triads if s in BALANCED)
    unbalanced_cnt = sum(1 for _,s in triads if s in UNBALANCED)

    trace: List[str] = []
    if verbose:
        trace.append(f"🧩 三角形一覧: {triads}")
        trace.append(f"✅ 安定三角形: {balanced_cnt} / ⚠️ 不安定三角形: {unbalanced_cnt}")

    # “孤立”っぽいノード（次数>=2 かつ全エッジが負）
    iso = 0
    for n in G.nodes():
        nbr_scores = [G[n][nbr]['score'] for nbr in G.neighbors(n)]
        if len(nbr_scores) >= 2 and all(v < 0.0 for v in nbr_scores):
            iso += 1

    edges = {}
    for u,v,data in G.edges(data=True):
        a,b = sorted([u,v])
        edges[(a,b)] = float(data.get('score',0.0))

    if verbose:
        trace.append(f"🧷 エッジスコア: {pformat(edges)}")
        trace.append(f"🥶 孤立ノード数(簡易): {iso}")
    result = {
        "unstable_triads": int(unbalanced_cnt),
        "balanced_triads": int(balanced_cnt),
        "iso_nodes": int(iso),
        "edges": edges,
        "triangles": [{"nodes": t, "struct": s} for t,s in triads]
    }

    return (result, trace) if return_trace else result
