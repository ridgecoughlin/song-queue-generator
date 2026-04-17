import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Song Queue Generator",
    page_icon="🎵",
    layout="centered"
)

# ── language detection ───────────────────────────────────────────────────────────
import re

LATIN_GENRES = {
    'spanish', 'latin', 'latino', 'reggaeton', 'salsa', 'tango',
    'sertanejo', 'forro', 'pagode', 'samba', 'mpb', 'brazil',
}

OTHER_NON_ENGLISH_GENRES = {
    'cantopop', 'mandopop', 'j-pop', 'j-rock', 'j-idol', 'j-dance',
    'k-pop', 'anime', 'turkish', 'french', 'german', 'swedish',
    'malay', 'iranian', 'indian', 'romance', 'world-music', 'afrobeat',
}

_LATIN_ARTISTS_RE = re.compile(
    r'\b(bad bunny|karol g|j balvin|daddy yankee|ozuna|maluma|anuel aa|'
    r'jhayco|feid|sech|myke towers|rauw alejandro|farruko|nicky jam|'
    r'zion|chencho corleone|manuel turizo|camilo|paulo londra|cnco|'
    r'becky g|natti natasha|lunay|mora|dei v|arcangel|de la ghetto|'
    r'wisin|yandel|cosculluela|alex rose|lyanno|brray|piso 21|'
    r'chris jedi|cris mj|quevedo|bizarrap|sebastián yatra|'
    r'rosalía|c tangana|bad gyal)\b',
    re.IGNORECASE
)

_SPANISH_WORDS_RE = re.compile(
    r'\b(el|la|los|las|un|una|de|del|que|por|con|para|como|todo|pero|'
    r'hay|fue|ser|muy|sin|sobre|entre|cuando|donde|aunque|porque|desde|'
    r'hasta|tambien|solo|siempre|nunca|algo|nada|cada|otro|otra|mismo|'
    r'misma|hace|querer|saber|llegar|pasar|seguir|llamar|venir|pensar|'
    r'poner|parecer|quedar|creer|hablar|llevar|dejar|sentir|conocer|'
    r'vivir|decir|ver|dar|estar|agua|amor|vida|tiempo|mundo|casa|forma|'
    r'parte|lugar|dia|vez|noche|ciudad|pueblo|camino|tierra|cielo|'
    r'corazon|mente|gente|hijo|madre|padre|hermano|amigo|te|tu|mi|'
    r'yo|no|si|me|lo|le|se|al|es|en|su|ya|ni|o)\b',
    re.IGNORECASE
)

def _has_spanish_words(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    if not words:
        return False
    matches = sum(1 for w in words if _SPANISH_WORDS_RE.match(w))
    return matches / len(words) > 0.35

def detect_language(track_name, genre, artists=''):
    """Returns 'latin', 'other', or 'english'."""
    if genre in LATIN_GENRES:
        return 'latin'
    if _LATIN_ARTISTS_RE.search(str(artists)):
        return 'latin'
    if _has_spanish_words(str(track_name)):
        return 'latin'
    if genre in OTHER_NON_ENGLISH_GENRES:
        return 'other'
    if not str(track_name).isascii() or not str(artists).isascii():
        return 'other'
    return 'english'

def allowed_in_queue(track_name, genre, artists, input_language):
    """Whether a song is allowed given the input song's language."""
    song_lang = detect_language(track_name, genre, artists)
    if input_language == 'english':
        return song_lang == 'english'
    elif input_language == 'latin':
        return song_lang in ('english', 'latin')
    else:
        # other language: allow that language + english
        return song_lang in ('english', 'other')

# ── data loading ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_graph():
    nodes = pd.read_parquet("data/nodes.parquet")
    edges = pd.read_parquet("data/edges.parquet")

    G = nx.Graph()

    for _, row in nodes.iterrows():
        G.add_node(row['track_id'], **row.to_dict())

    for _, row in edges.iterrows():
        G.add_edge(row['source'], row['target'], cost=row['cost'], similarity=row['similarity'])

    # precompute neighbor community sets for fast bridge lookups
    neighbor_communities = {
        node: {G.nodes[n].get('community') for n in G.neighbors(node)}
        for node in G.nodes
    }

    return G, nodes, neighbor_communities

@st.cache_data
def load_community_centroids():
    nodes = pd.read_parquet("data/nodes.parquet")
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'mode'
    ]
    return {
        community: {col: group[col].mean() for col in feature_cols}
        for community, group in nodes.groupby('community')
    }

# ── algorithm functions ─────────────────────────────────────────────────────────
def find_target_community(input_community, community_centroids, G, nodes, neighbor_communities):
    input_centroid = np.array(list(community_centroids[input_community].values()))

    distances = []
    for community, centroid_dict in community_centroids.items():
        if community == input_community:
            continue
        centroid = np.array(list(centroid_dict.values()))
        dist = np.linalg.norm(input_centroid - centroid)
        distances.append((community, dist))
    distances.sort(key=lambda x: x[1])
    ranked_communities = [c for c, _ in distances]

    # find reachable communities using precomputed neighbor index
    input_songs = set(nodes[nodes['community'] == input_community]['track_id'])
    valid_communities = set()
    for node_id in input_songs:
        for neighbor_id in G.neighbors(node_id):
            if G.nodes[neighbor_id].get('community') == input_community and G.nodes[neighbor_id].get('betweenness') is not None:
                valid_communities |= neighbor_communities[neighbor_id] - {input_community}

    rng = np.random.default_rng()
    for batch_size in [5, 10, 20, len(ranked_communities)]:
        valid = [c for c in ranked_communities[:batch_size] if c in valid_communities]
        if valid:
            return int(valid[rng.integers(0, len(valid))])

    raise ValueError(f'No reachable target community found from community {input_community}')


def find_bridge_song(input_community, target_community, G, nodes, neighbor_communities):
    candidates = []
    for node_id, data in G.nodes(data=True):
        if data.get('community') not in (input_community, target_community):
            continue
        if data.get('betweenness') is None:
            continue
        comms = neighbor_communities[node_id]
        if input_community in comms and target_community in comms:
            candidates.append((node_id, data['betweenness'], data.get('community')))

    if not candidates:
        raise ValueError(f'No bridge song found between communities {input_community} and {target_community}')

    candidates.sort(key=lambda x: x[1], reverse=True)
    top3 = candidates[:3]
    rng = np.random.default_rng()
    chosen = top3[rng.integers(0, len(top3))]
    return chosen[0], int(chosen[2])


def pick_destination(target_comm, bridge_id, nodes):
    pool = nodes[(nodes['community'] == target_comm) & (nodes['track_id'] != bridge_id)]
    rng = np.random.default_rng()
    return pool.iloc[rng.integers(0, len(pool))]


def shortest_path(start_track_id, target_track_id, G):
    try:
        path_ids = nx.dijkstra_path(G, start_track_id, target_track_id, weight='cost')
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        raise ValueError(f'No path found between {start_track_id} and {target_track_id}')

    rows = []
    for tid in path_ids:
        d = G.nodes[tid]
        rows.append({
            'track_id':   tid,
            'track_name': d.get('track_name'),
            'artist':     d.get('artists'),
            'genre':      d.get('genre'),
            'community':  d.get('community'),
        })
    return pd.DataFrame(rows)


def generate_queue(input_track_id, G, nodes, neighbor_communities, community_centroids, queue_length=10, max_artist_appearances=2):
    if input_track_id not in G.nodes:
        raise ValueError(f'No song found with track_id: {input_track_id}')

    input_data = G.nodes[input_track_id]
    input_community = int(input_data['community'])
    input_language = detect_language(
        input_data.get('track_name', ''),
        input_data.get('genre', ''),
        input_data.get('artists', '')
    )

    def ok(track_name, genre, artists):
        return allowed_in_queue(track_name, genre, artists, input_language)

    target_community = find_target_community(input_community, community_centroids, G, nodes, neighbor_communities)
    bridge_id, bridge_community = find_bridge_song(input_community, target_community, G, nodes, neighbor_communities)
    dest_row = pick_destination(target_community, bridge_id, nodes)

    path1 = shortest_path(input_track_id, bridge_id, G)
    path2 = shortest_path(bridge_id, dest_row['track_id'], G)
    final_path = pd.concat([path1, path2.iloc[1:]], ignore_index=True)

    final_path = final_path.drop_duplicates(subset='track_id', keep='first')
    final_path = final_path.drop_duplicates(subset='track_name', keep='first')
    final_path = final_path[final_path.apply(
        lambda r: ok(r['track_name'], r['genre'], r['artist']), axis=1
    )].reset_index(drop=True)

    # artist repeat limiting
    artist_counts = final_path['artist'].value_counts()
    excess_artists = artist_counts[artist_counts > max_artist_appearances].index.tolist()
    rng = np.random.default_rng()
    for artist in excess_artists:
        artist_rows = final_path[final_path['artist'] == artist]
        to_replace = artist_rows.index[max_artist_appearances:]
        for idx in to_replace:
            current_id = final_path.loc[idx, 'track_id']
            used_ids = set(final_path['track_id'])
            neighbors = list(G.neighbors(current_id))
            replacements = [
                n for n in neighbors
                if G.nodes[n].get('artists') != artist and n not in used_ids
                and ok(G.nodes[n].get('track_name', ''), G.nodes[n].get('genre', ''), G.nodes[n].get('artists', ''))
            ]
            if replacements:
                r_id = replacements[rng.integers(0, len(replacements))]
                d = G.nodes[r_id]
                final_path.loc[idx, ['track_id', 'track_name', 'artist', 'genre', 'community']] = \
                    [r_id, d.get('track_name'), d.get('artists'), d.get('genre'), d.get('community')]

    # enforce fixed queue length
    if len(final_path) > queue_length:
        final_path = final_path.iloc[:queue_length]
    elif len(final_path) < queue_length:
        used_ids = set(final_path['track_id'])
        while len(final_path) < queue_length:
            last_id = final_path.iloc[-1]['track_id']
            neighbors = [
                n for n in list(G.neighbors(last_id))
                if n not in used_ids
                and ok(G.nodes[n].get('track_name', ''), G.nodes[n].get('genre', ''), G.nodes[n].get('artists', ''))
            ]
            if not neighbors:
                break
            neighbors_with_bw = [n for n in neighbors if G.nodes[n].get('betweenness') is not None]
            pool = neighbors_with_bw if neighbors_with_bw else neighbors
            next_id = pool[rng.integers(0, len(pool))]
            d = G.nodes[next_id]
            final_path = pd.concat([final_path, pd.DataFrame([{
                'track_id':   next_id,
                'track_name': d.get('track_name'),
                'artist':     d.get('artists'),
                'genre':      d.get('genre'),
                'community':  d.get('community'),
            }])], ignore_index=True)
            used_ids.add(next_id)

    return final_path.reset_index(drop=True)

# ── startup ─────────────────────────────────────────────────────────────────────
G, nodes, neighbor_communities = load_graph()
community_centroids = load_community_centroids()

# ── search index ────────────────────────────────────────────────────────────────
@st.cache_data
def build_search_df():
    df = pd.read_parquet("data/nodes.parquet")[['track_id', 'track_name', 'artists', 'popularity', 'genre']]
    # exclude only truly unreadable scripts (non-ASCII names with no Latin characters)
    df = df[df['track_name'].apply(lambda x: str(x).isascii())]
    df['label'] = df['track_name'] + ' — ' + df['artists']
    df = df.sort_values('popularity', ascending=False).drop_duplicates(subset='label')
    return df

search_df = build_search_df()
label_to_id = dict(zip(search_df['label'], search_df['track_id']))

# ── ui ──────────────────────────────────────────────────────────────────────────
st.title("🎵 Song Queue Generator")
st.caption("Search for a song and we'll build a queue that takes you on a sonic journey.")

selection = st.selectbox(
    "Search for a song or artist",
    options=search_df['label'].tolist(),
    index=None,
    placeholder="e.g. Bohemian Rhapsody or Queen"
)

selected_track_id = label_to_id.get(selection)

if selected_track_id and st.button("Generate Queue", type="primary"):
    with st.spinner("Building your queue..."):
        try:
            queue = generate_queue(selected_track_id, G, nodes, neighbor_communities, community_centroids)
            st.success(f"Queue generated — {len(queue)} songs")
            rows_html = ""
            for i, row in queue.iterrows():
                rows_html += (
                    f"<div style='padding: 8px 0; border-bottom: 1px solid #e0e0e0;'>"
                    f"<span style='font-weight:600;'>{i + 1}. {row['track_name']}</span>"
                    f" &nbsp; {row['artist']}<br>"
                    f"<span style='font-size:0.8em; color:gray;'><i>{row['genre']}</i></span>"
                    f"</div>"
                )
            st.markdown(rows_html, unsafe_allow_html=True)
        except ValueError as e:
            st.error(f"Error: {e}")
