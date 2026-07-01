import streamlit as st
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from rapidfuzz import process, fuzz
import graphviz
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = "bolt+ssc://p-mt-2a41353b6b9a-1-0090.production-orch-0695.neo4j.io:7687"
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "f9c5be28"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

@st.cache_data
def get_all_entities():
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("MATCH (n) RETURN n.name AS name, labels(n)[0] AS type")
        return [(r["name"], r["type"]) for r in result if r["name"]]

def fuzzy_match_entities(query, all_entities, threshold=92):
    entity_names = [e[0] for e in all_entities]
    entity_type_map = {e[0]: e[1] for e in all_entities}

    skip_words = {
        "what", "which", "where", "when", "how", "are", "the", "for",
        "against", "effective", "used", "does", "affect", "effect", "control",
        "grow", "need", "require", "is", "in", "of", "and", "or",
        "a", "an", "to", "do", "can", "will", "that", "this",
        "season", "grown", "needed", "required", "suitable", "best",
        "crop", "crops", "plant", "plants", "farm", "farmer"
    }

    words = query.replace("?", "").replace(",", "").split()
    candidates = [w for w in words if w.lower() not in skip_words and len(w) >= 4]

    matched = {}

    for word in candidates:
        # First try exact match (case insensitive)
        for name in entity_names:
            if word.lower() == name.lower():
                if name not in matched:
                    matched[name] = entity_type_map.get(name, "Unknown")
                break
        else:
            # Only fuzzy match single-word entity names to avoid partial matches
            single_word_entities = [n for n in entity_names if len(n.split()) == 1]
            match = process.extractOne(word, single_word_entities, scorer=fuzz.ratio)
            if match and match[1] >= threshold:
                name = match[0]
                if name not in matched:
                    matched[name] = entity_type_map.get(name, "Unknown")

    return matched

def query_graph_direct(matched_entities, question):
    driver = get_driver()
    results = []
    crop_names = [n for n, t in matched_entities.items() if t == "Crop"]
    q_lower = question.lower()

    with driver.session(database=NEO4J_DATABASE) as session:

        if any(w in q_lower for w in ["disease", "pest", "affect", "effect", "problem", "attack", "insect"]):
            for crop in crop_names:
                r = session.run("""
                    MATCH (c:Crop)-[:AFFECTED_BY]->(x)
                    WHERE toLower(c.name) CONTAINS toLower($name)
                    RETURN c.name AS crop, labels(x)[0] AS type, x.name AS result
                    LIMIT 15
                """, name=crop)
                results.extend([dict(row) for row in r])

        if any(w in q_lower for w in ["pesticide", "control", "treat", "chemical", "manage", "spray"]):
            for crop in crop_names:
                r = session.run("""
                    MATCH (c:Crop)-[:CONTROLLED_BY]->(x)
                    WHERE toLower(c.name) CONTAINS toLower($name)
                    RETURN c.name AS crop, labels(x)[0] AS type, x.name AS result
                    LIMIT 15
                """, name=crop)
                results.extend([dict(row) for row in r])

        if any(w in q_lower for w in ["where", "state", "location", "region", "grown", "cultivat", "district"]):
            for crop in crop_names:
                r = session.run("""
                    MATCH (c:Crop)-[:GROWN_IN]->(x)
                    WHERE toLower(c.name) CONTAINS toLower($name)
                    RETURN c.name AS crop, labels(x)[0] AS type, x.name AS result
                    LIMIT 15
                """, name=crop)
                results.extend([dict(row) for row in r])

        if any(w in q_lower for w in ["season", "when", "month", "rabi", "kharif", "time"]):
            for crop in crop_names:
                r = session.run("""
                    MATCH (c:Crop)-[:GROWN_DURING]->(x)
                    WHERE toLower(c.name) CONTAINS toLower($name)
                    RETURN c.name AS crop, labels(x)[0] AS type, x.name AS result
                    LIMIT 15
                """, name=crop)
                results.extend([dict(row) for row in r])

        if any(w in q_lower for w in ["nutrient", "fertilizer", "require", "need", "manure", "nitrogen", "potassium"]):
            for crop in crop_names:
                r = session.run("""
                    MATCH (c:Crop)-[:REQUIRES]->(x)
                    WHERE toLower(c.name) CONTAINS toLower($name)
                    RETURN c.name AS crop, labels(x)[0] AS type, x.name AS result
                    LIMIT 15
                """, name=crop)
                results.extend([dict(row) for row in r])

        # Fallback
        if not results:
            all_names = list(matched_entities.keys())
            r = session.run("""
                MATCH (n)-[rel]->(m)
                WHERE n.name IN $names OR m.name IN $names
                RETURN n.name AS source, type(rel) AS relation, m.name AS result
                LIMIT 20
            """, names=all_names)
            results.extend([dict(row) for row in r])

    return results

def generate_answer(question, graph_context):
    if not graph_context:
        return "I could not find relevant information in the knowledge graph for your question. Please try rephrasing with specific crop names like Rice, Wheat, Cotton, Paddy."
    context_str = "\n".join([str(r) for r in graph_context])
    prompt = f"""You are an agricultural expert assistant.
Based ONLY on the following knowledge graph data, answer the question clearly and concisely.
Do not add any information not present in the data below.

Knowledge Graph Data:
{context_str}

Question: {question}

Answer:"""
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.content

def get_subgraph(entity_names, limit=40):
    if not entity_names:
        return []
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run("""
        MATCH (n)-[r]->(m)
        WHERE n.name IN $names OR m.name IN $names
        RETURN n.name AS source, labels(n)[0] AS src_type,
               type(r) AS relation,
               m.name AS target, labels(m)[0] AS tgt_type
        LIMIT $limit
        """, names=entity_names, limit=limit)
        return [dict(r) for r in result]

def build_graph_viz(subgraph_data):
    dot = graphviz.Digraph(
        graph_attr={"bgcolor": "#1e1e2e", "rankdir": "LR"},
        node_attr={"style": "filled", "fontcolor": "white", "fontsize": "11"},
        edge_attr={"color": "#888888", "fontcolor": "#aaaaaa", "fontsize": "9"}
    )
    color_map = {
        "Crop": "#2ecc71", "Disease": "#e74c3c", "Pest": "#e67e22",
        "Pesticide": "#9b59b6", "Fertilizer": "#3498db", "Soil": "#795548",
        "Season": "#00bcd4", "Location": "#f39c12", "State": "#f39c12",
        "District": "#ff9800", "Country": "#ffeb3b", "Weather": "#607d8b",
        "Irrigation": "#1abc9c", "Nutrient": "#27ae60", "FarmerPractice": "#8e44ad",
    }
    added_nodes = set()
    for row in subgraph_data:
        src, tgt, rel = row["source"], row["target"], row["relation"]
        stype, ttype = row.get("src_type", ""), row.get("tgt_type", "")
        if src and src not in added_nodes:
            dot.node(src, src, fillcolor=color_map.get(stype, "#555555"), shape="ellipse")
            added_nodes.add(src)
        if tgt and tgt not in added_nodes:
            dot.node(tgt, tgt, fillcolor=color_map.get(ttype, "#555555"), shape="ellipse")
            added_nodes.add(tgt)
        if src and tgt:
            dot.edge(src, tgt, label=rel)
    return dot

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="AgriGraphRAG", page_icon="🌾", layout="wide")
st.title("🌾 AgriGraphRAG — Explainable Agricultural QA")
st.markdown("Ask any agricultural question. Answers are grounded in a **Neo4j Knowledge Graph**.")

with st.sidebar:
    st.header("📌 Entity Legend")
    st.markdown("🟢 **Crop**")
    st.markdown("🔴 **Disease**")
    st.markdown("🟠 **Pest**")
    st.markdown("🟣 **Pesticide**")
    st.markdown("🔵 **Fertilizer**")
    st.markdown("🟡 **Location / State**")
    st.markdown("⚪ **Weather**")
    st.markdown("🩵 **Irrigation**")
    st.markdown("🟤 **Soil**")
    st.markdown("---")
    st.markdown("**Model:** Gemini-2.5-flash + Neo4j")
    st.markdown("**Method:** GraphRAG + RapidFuzz")

st.markdown("### 💡 Sample Questions")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🌾 Pesticides for Wheat?"):
        st.session_state["query"] = "What pesticides control Aphids in Wheat?"
with col2:
    if st.button("🍚 Diseases in Rice?"):
        st.session_state["query"] = "What diseases affect Rice?"
with col3:
    if st.button("🌱 Fertilizers for Paddy?"):
        st.session_state["query"] = "What fertilizers are required for Paddy?"
col4, col5, col6 = st.columns(3)
with col4:
    if st.button("🐛 Pests in Cotton?"):
        st.session_state["query"] = "What pests affect Cotton?"
with col5:
    if st.button("🌍 Where is Wheat grown?"):
        st.session_state["query"] = "In which states is Wheat grown?"
with col6:
    if st.button("💊 Control Stem Borer?"):
        st.session_state["query"] = "How to control Stem Borer in Rice?"

query = st.text_input(
    "🔍 Enter your agricultural question:",
    value=st.session_state.get("query", ""),
    placeholder="e.g. What diseases affect Rice?"
)

if st.button("🚀 Get Answer", type="primary") and query:
    with st.spinner("🔄 Querying Knowledge Graph..."):

        all_entities     = get_all_entities()
        matched_entities = fuzzy_match_entities(query, all_entities)

        st.markdown("---")

        if matched_entities:
            cols = st.columns(min(len(matched_entities), 4))
            for i, (name, etype) in enumerate(list(matched_entities.items())[:4]):
                cols[i].metric(label=etype, value=name)
        else:
            st.warning("No entities matched. Try using specific names like Rice, Wheat, Cotton, Paddy.")

        graph_context = query_graph_direct(matched_entities, query)
        answer        = generate_answer(query, graph_context)

        st.markdown("### 📋 Answer")
        st.success(answer)

        if graph_context:
            with st.expander(f"📊 Knowledge Graph Context ({len(graph_context)} records)"):
                st.dataframe(graph_context)

        st.markdown("### 🕸️ Knowledge Subgraph")
        subgraph_data = get_subgraph(list(matched_entities.keys()), limit=40)
        if subgraph_data:
            st.graphviz_chart(build_graph_viz(subgraph_data).source)
        else:
            st.warning("No subgraph found.")

st.markdown("---")
st.markdown(
    "<center>Explainable GraphRAG for Precision Agriculture | "
    "Devdeep Banerjee | DR. B.C. ROY ENGINEERING COLLEGE</center>",
    unsafe_allow_html=True
)