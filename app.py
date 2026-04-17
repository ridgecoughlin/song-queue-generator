import streamlit as st
import pandas as pd
import numpy as np
from neo4j import GraphDatabase

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Song Queue Generator",
    page_icon="🎵",
    layout="centered"
)

# ── neo4j connection ────────────────────────────────────────────────────────────
@st.cache_resource
def get_driver():
    return GraphDatabase.driver(
        st.secrets["NEO4J_URI"],
        auth=(st.secrets["NEO4J_USER"], st.secrets["NEO4J_PASSWORD"])
    )

def run_query(query, **kwargs):
    driver = get_driver()
    with driver.session(database="neo4j") as session:
        result = session.run(query, **kwargs)
        return pd.DataFrame([r.values() for r in result], columns=result.keys())

def neo4j_run(query, **kwargs):
    driver = get_driver()
    with driver.session(database="neo4j") as session:
        return session.run(query, **kwargs)

# ── graph projection ────────────────────────────────────────────────────────────
@st.cache_resource
def ensure_projection():
    neo4j_run("CALL gds.graph.drop('song_graph', false) YIELD graphName")
    result = run_query("""
        CALL gds.graph.project(
            'song_graph',
            'Song',
            {
                SIMILAR_TO: {
                    orientation: 'UNDIRECTED',
                    properties: ['similarity', 'cost']
                }
            }
        )
        YIELD nodeCount, relationshipCount
        RETURN nodeCount, relationshipCount
    """)
    return result

# ── centroid cache ──────────────────────────────────────────────────────────────
@st.cache_data
def load_community_centroids():
    centroid_df = run_query("""
        MATCH (s:Song)
        WHERE s.community IS NOT NULL
        RETURN s.community             AS community,
               avg(s.danceability)     AS danceability,
               avg(s.energy)           AS energy,
               avg(s.loudness)         AS loudness,
               avg(s.speechiness)      AS speechiness,
               avg(s.acousticness)     AS acousticness,
               avg(s.instrumentalness) AS instrumentalness,
               avg(s.liveness)         AS liveness,
               avg(s.valence)          AS valence,
               avg(s.tempo)            AS tempo,
               avg(s.mode)             AS mode
    """)
    feature_cols = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'mode'
    ]
    return {
        row['community']: {col: row[col] for col in feature_cols}
        for _, row in centroid_df.iterrows()
    }

# ── algorithm functions ─────────────────────────────────────────────────────────
def find_target_community(input_community, community_centroids):
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

    all_bridge_counts = run_query("""
        MATCH (bridge:Song)-[:SIMILAR_TO]-(a:Song),
              (bridge:Song)-[:SIMILAR_TO]-(b:Song)
        WHERE a.community = $input_community
          AND bridge.community IN [$input_community, a.community]
          AND bridge.betweenness IS NOT NULL
        RETURN DISTINCT b.community AS target_community,
               count(DISTINCT bridge) AS bridge_count
    """, input_community=int(input_community))

    valid_communities = set(
        all_bridge_counts[all_bridge_counts['bridge_count'] > 0]['target_community'].tolist()
    )

    rng = np.random.default_rng()
    for batch_size in [5, 10, 20, len(ranked_communities)]:
        valid = [c for c in ranked_communities[:batch_size] if c in valid_communities]
        if valid:
            return int(valid[rng.integers(0, len(valid))])

    raise ValueError(f'No reachable target community found from community {input_community}')


def find_bridge_song(input_community, target_community):
    result = run_query("""
        MATCH (bridge:Song)-[:SIMILAR_TO]-(a:Song),
              (bridge:Song)-[:SIMILAR_TO]-(b:Song)
        WHERE a.community = $input_community
          AND b.community = $target_community
          AND bridge.community IN [$input_community, $target_community]
          AND bridge.betweenness IS NOT NULL
        RETURN DISTINCT
            bridge.id          AS track_id,
            bridge.track_name  AS track_name,
            bridge.genre       AS genre,
            bridge.community   AS community,
            bridge.betweenness AS betweenness
        ORDER BY betweenness DESC
        LIMIT 3
    """, input_community=int(input_community), target_community=int(target_community))

    if result.empty:
        raise ValueError(f'No bridge song found between communities {input_community} and {target_community}')

    rng = np.random.default_rng()
    best = result.iloc[rng.integers(0, len(result))]
    return best['track_id'], int(best['community'])


def pick_destination(target_comm, bridge_id):
    result = run_query("""
        MATCH (s:Song)
        WHERE s.community = $target_community
        AND s.id <> $bridge_track_id
        RETURN s.id AS track_id, s.track_name AS track_name,
               s.genre AS genre, s.popularity AS popularity
    """, target_community=int(target_comm), bridge_track_id=bridge_id)

    rng = np.random.default_rng()
    return result.iloc[rng.integers(0, len(result))]


def shortest_path(start_track_id, target_track_id):
    result = run_query("""
        MATCH (source:Song {id: $source_id}), (target:Song {id: $target_id})
        CALL gds.shortestPath.dijkstra.stream('song_graph',
            {
                sourceNode: source,
                targetNode: target,
                relationshipWeightProperty: 'cost'
            }
        )
        YIELD totalCost, nodeIds
        RETURN
            totalCost,
            [nodeId IN nodeIds | gds.util.asNode(nodeId).id] AS track_ids,
            [nodeId IN nodeIds | gds.util.asNode(nodeId).track_name] AS track_names,
            [nodeId IN nodeIds | gds.util.asNode(nodeId).artists] AS artists,
            [nodeId IN nodeIds | gds.util.asNode(nodeId).genre] AS genres,
            [nodeId IN nodeIds | gds.util.asNode(nodeId).community] AS communities
    """, source_id=start_track_id, target_id=target_track_id)

    if result.empty:
        raise ValueError(f'No path found between {start_track_id} and {target_track_id}')

    path = result.iloc[0]
    return pd.DataFrame({
        'track_id':   path['track_ids'],
        'track_name': path['track_names'],
        'artist':     path['artists'],
        'genre':      path['genres'],
        'community':  path['communities']
    })


def generate_queue(input_track_id, community_centroids, queue_length=10, max_artist_appearances=2):
    song_result = run_query("""
        MATCH (s:Song {id: $track_id})
        RETURN s.id AS track_id, s.track_name AS track_name,
               s.community AS community, s.artists AS artist
    """, track_id=input_track_id)

    if song_result.empty:
        raise ValueError(f'No song found with track_id: {input_track_id}')

    input_song = song_result.iloc[0]
    input_id = input_song['track_id']
    input_community = int(input_song['community'])

    target_community = find_target_community(input_community, community_centroids)
    bridge_id, bridge_community = find_bridge_song(input_community, target_community)
    dest_row = pick_destination(target_community, bridge_id)

    path1 = shortest_path(input_id, bridge_id)
    path2 = shortest_path(bridge_id, dest_row['track_id'])
    final_path = pd.concat([path1, path2.iloc[1:]], ignore_index=True)

    final_path = final_path.drop_duplicates(subset='track_id', keep='first')
    final_path = final_path.drop_duplicates(subset='track_name', keep='first')
    final_path = final_path.reset_index(drop=True)

    # artist repeat limiting
    artist_counts = final_path['artist'].value_counts()
    excess_artists = artist_counts[artist_counts > max_artist_appearances].index.tolist()
    for artist in excess_artists:
        artist_rows = final_path[final_path['artist'] == artist]
        to_replace = artist_rows.index[max_artist_appearances:]
        for idx in to_replace:
            current_id = final_path.loc[idx, 'track_id']
            used_ids = final_path['track_id'].tolist()
            replacement = run_query("""
                MATCH (s:Song {id: $track_id})-[:SIMILAR_TO]->(n:Song)
                WHERE n.artists <> $artist
                  AND NOT n.id IN $used_ids
                RETURN n.id AS track_id, n.track_name AS track_name,
                       n.artists AS artist, n.genre AS genre,
                       n.community AS community
                ORDER BY n.betweenness IS NOT NULL DESC
                LIMIT 1
            """, track_id=current_id, artist=artist, used_ids=used_ids)
            if not replacement.empty:
                r = replacement.iloc[0]
                final_path.loc[idx, ['track_id', 'track_name', 'artist', 'genre', 'community']] = \
                    [r['track_id'], r['track_name'], r['artist'], r['genre'], r['community']]

    # enforce fixed queue length
    if len(final_path) > queue_length:
        final_path = final_path.iloc[:queue_length]
    elif len(final_path) < queue_length:
        used_ids = final_path['track_id'].tolist()
        while len(final_path) < queue_length:
            last_id = final_path.iloc[-1]['track_id']
            neighbors = run_query("""
                MATCH (s:Song {id: $track_id})-[:SIMILAR_TO]->(n:Song)
                WHERE NOT n.id IN $used_ids
                RETURN n.id AS track_id, n.track_name AS track_name,
                       n.artists AS artist, n.genre AS genre,
                       n.community AS community
                ORDER BY n.betweenness IS NOT NULL DESC
                LIMIT 5
            """, track_id=last_id, used_ids=used_ids)
            if neighbors.empty:
                break
            rng = np.random.default_rng()
            next_song = neighbors.iloc[rng.integers(0, len(neighbors))]
            final_path = pd.concat([final_path, pd.DataFrame([{
                'track_id':   next_song['track_id'],
                'track_name': next_song['track_name'],
                'artist':     next_song['artist'],
                'genre':      next_song['genre'],
                'community':  next_song['community'],
            }])], ignore_index=True)
            used_ids.append(next_song['track_id'])

    return final_path.reset_index(drop=True)

# ── startup ─────────────────────────────────────────────────────────────────────
ensure_projection()
community_centroids = load_community_centroids()

# ── ui ──────────────────────────────────────────────────────────────────────────
st.title("🎵 Song Queue Generator")
st.caption("Search for a song and we'll build a queue that takes you on a sonic journey.")

search_term = st.text_input("Search for a song", placeholder="e.g. Bohemian Rhapsody")

selected_track_id = None

if search_term and len(search_term) >= 2:
    results = run_query("""
        MATCH (s:Song)
        WHERE toLower(s.track_name) CONTAINS toLower($search_term)
        RETURN s.id AS track_id, s.track_name AS track_name, s.artists AS artists
        ORDER BY s.popularity DESC
        LIMIT 8
    """, search_term=search_term)

    if results.empty:
        st.warning("No songs found. Try a different search.")
    else:
        options = {
            f"{row['track_name']} — {row['artists']}": row['track_id']
            for _, row in results.iterrows()
        }
        selection = st.radio("Select a song", options=list(options.keys()), index=0)
        selected_track_id = options[selection]

if selected_track_id and st.button("Generate Queue", type="primary"):
    with st.spinner("Building your queue..."):
        try:
            queue = generate_queue(selected_track_id, community_centroids)
            st.success(f"Queue generated — {len(queue)} songs")
            st.dataframe(
                queue[['track_name', 'artist', 'genre', 'community']].rename(columns={
                    'track_name': 'Track',
                    'artist':     'Artist',
                    'genre':      'Genre',
                    'community':  'Community'
                }),
                use_container_width=True,
                hide_index=False
            )
        except ValueError as e:
            st.error(f"Error: {e}")
