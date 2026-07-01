"""
Comparative Evaluation Script for Explainable GraphRAG Thesis
================================================================
Compares 4 systems on the same 100 agricultural questions:
  1. Gemini Baseline (Standalone Gemini 2.5 Flash, no grounding)
  2. Think-on-Graph (ToG) - Sun et al. 2023 [arXiv:2307.07697]
  3. Proposed GraphRAG (ours) - fuzzy matching + intent detection + fallback

Outputs: Accuracy, Precision, Recall, F1-Score for each system + LaTeX table
"""

import os
import json
import re
import time
import pandas as pd
from neo4j import GraphDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from rapidfuzz import process, fuzz
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = "bolt+ssc://p-mt-2a41353b6b9a-1-0090.production-orch-0695.neo4j.io:7687"
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = "f9c5be28"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ──────────────────────────────────────────────────────────────────────────
# 100 TEST QUESTIONS WITH GROUND TRUTH
# ──────────────────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    {"q": "What pests affect Rice?", "gt": ["Stem Borer", "Brown Plant Hopper", "Leaf Folder", "Gall Midge"]},
    {"q": "What pests attack Wheat?", "gt": ["Aphids", "Termite", "Jassids"]},
    {"q": "What pests affect Cotton?", "gt": ["Bollworm", "Whitefly", "Aphids", "Thrips"]},
    {"q": "What insects affect Maize?", "gt": ["Stem Borer", "Fall Armyworm"]},
    {"q": "What pests attack Sorghum?", "gt": ["Shoot Fly", "Stem Borer"]},
    {"q": "What pests affect Sugarcane?", "gt": ["Early Shoot Borer", "Top Borer", "Pyrilla"]},
    {"q": "What pests attack Groundnut?", "gt": ["Leaf Miner", "Aphids", "Thrips"]},
    {"q": "What pests affect Mustard?", "gt": ["Aphids", "Painted Bug", "Sawfly"]},
    {"q": "What pests attack Soybean?", "gt": ["Stem Fly", "Girdle Beetle", "Whitefly"]},
    {"q": "What pests affect Sunflower?", "gt": ["Capitulum Borer", "Aphids"]},
    {"q": "What diseases affect Rice?", "gt": ["Blast", "Brown Spot", "Sheath Blight"]},
    {"q": "What diseases affect Wheat?", "gt": ["Rust", "Smut", "Loose Smut"]},
    {"q": "What diseases affect Cotton?", "gt": ["Wilt", "Root Rot", "Leaf Curl"]},
    {"q": "What diseases affect Maize?", "gt": ["Downy Mildew", "Blight", "Rust"]},
    {"q": "What diseases affect Tomato?", "gt": ["Early Blight", "Late Blight", "Wilt"]},
    {"q": "What diseases affect Potato?", "gt": ["Late Blight", "Early Blight"]},
    {"q": "What diseases affect Sugarcane?", "gt": ["Red Rot", "Smut", "Wilt"]},
    {"q": "What diseases affect Groundnut?", "gt": ["Tikka Disease", "Rust", "Stem Rot"]},
    {"q": "What diseases affect Mustard?", "gt": ["White Rust", "Downy Mildew"]},
    {"q": "What diseases affect Soybean?", "gt": ["Rust", "Bacterial Pustule"]},
    {"q": "What pesticides control pests in Rice?", "gt": ["Chlorpyriphos", "Carbofuran", "Monocrotophos"]},
    {"q": "What chemicals control Wheat diseases?", "gt": ["Mancozeb", "Carbendazim", "Propiconazole"]},
    {"q": "How to control Bollworm in Cotton?", "gt": ["Endosulfan", "Cypermethrin"]},
    {"q": "What pesticides control Aphids in Mustard?", "gt": ["Imidacloprid", "Dimethoate"]},
    {"q": "How to manage Stem Borer in Rice?", "gt": ["Carbofuran", "Chlorpyriphos"]},
    {"q": "What controls Brown Plant Hopper in Rice?", "gt": ["Imidacloprid", "Buprofezin"]},
    {"q": "What fungicides control Blast in Rice?", "gt": ["Tricyclazole", "Isoprothiolane"]},
    {"q": "What controls Late Blight in Potato?", "gt": ["Mancozeb", "Metalaxyl"]},
    {"q": "What pesticides control Whitefly in Cotton?", "gt": ["Imidacloprid", "Acetamiprid"]},
    {"q": "What controls Downy Mildew in Mustard?", "gt": ["Metalaxyl", "Mancozeb"]},
    {"q": "In which states is Rice grown?", "gt": ["West Bengal", "Punjab", "Uttar Pradesh"]},
    {"q": "Where is Wheat grown in India?", "gt": ["Punjab", "Haryana", "Uttar Pradesh"]},
    {"q": "Which states grow Cotton?", "gt": ["Gujarat", "Maharashtra", "Andhra Pradesh"]},
    {"q": "Where is Sugarcane grown?", "gt": ["Uttar Pradesh", "Maharashtra", "Karnataka"]},
    {"q": "Which states cultivate Maize?", "gt": ["Karnataka", "Andhra Pradesh", "Bihar"]},
    {"q": "Where is Groundnut grown?", "gt": ["Gujarat", "Andhra Pradesh", "Tamil Nadu"]},
    {"q": "Which states grow Mustard?", "gt": ["Rajasthan", "Uttar Pradesh", "Haryana"]},
    {"q": "Where is Soybean cultivated?", "gt": ["Madhya Pradesh", "Maharashtra"]},
    {"q": "Which states grow Potato?", "gt": ["Uttar Pradesh", "West Bengal", "Bihar"]},
    {"q": "Where is Sunflower grown in India?", "gt": ["Karnataka", "Andhra Pradesh"]},
    {"q": "In which season is Rice grown?", "gt": ["Kharif"]},
    {"q": "What season is Wheat grown?", "gt": ["Rabi"]},
    {"q": "When is Maize cultivated?", "gt": ["Kharif"]},
    {"q": "What season is Mustard grown?", "gt": ["Rabi"]},
    {"q": "When is Groundnut grown?", "gt": ["Kharif", "Rabi"]},
    {"q": "What season is Sugarcane planted?", "gt": ["Kharif", "Annual"]},
    {"q": "When is Soybean grown?", "gt": ["Kharif"]},
    {"q": "What season is Potato cultivated?", "gt": ["Rabi"]},
    {"q": "When is Sunflower grown?", "gt": ["Kharif", "Rabi"]},
    {"q": "What season is Cotton grown?", "gt": ["Kharif"]},
    {"q": "What fertilizers does Rice require?", "gt": ["Nitrogen", "Phosphorus", "Potassium"]},
    {"q": "What nutrients does Wheat need?", "gt": ["Nitrogen", "Phosphorus", "Potassium"]},
    {"q": "What fertilizers are needed for Cotton?", "gt": ["Nitrogen", "Phosphorus", "Potassium"]},
    {"q": "What nutrients does Maize require?", "gt": ["Nitrogen", "Phosphorus", "Zinc"]},
    {"q": "What fertilizers are used for Sugarcane?", "gt": ["Nitrogen", "Phosphorus", "Potassium"]},
    {"q": "What nutrients does Groundnut need?", "gt": ["Phosphorus", "Calcium", "Gypsum"]},
    {"q": "What fertilizers are needed for Mustard?", "gt": ["Nitrogen", "Phosphorus", "Sulfur"]},
    {"q": "What nutrients does Soybean require?", "gt": ["Phosphorus", "Potassium"]},
    {"q": "What fertilizers are used for Potato?", "gt": ["Nitrogen", "Phosphorus", "Potassium"]},
    {"q": "What nutrients does Sunflower need?", "gt": ["Nitrogen", "Phosphorus", "Potassium"]},
    {"q": "What pesticides control pests that affect Rice?", "gt": ["Chlorpyriphos", "Carbofuran", "Imidacloprid"]},
    {"q": "What chemicals control diseases of Wheat?", "gt": ["Mancozeb", "Carbendazim"]},
    {"q": "What pesticides are used against Cotton pests?", "gt": ["Endosulfan", "Cypermethrin", "Imidacloprid"]},
    {"q": "What fungicides treat Rice diseases?", "gt": ["Tricyclazole", "Carbendazim"]},
    {"q": "What insecticides protect Sugarcane from pests?", "gt": ["Chlorpyriphos", "Imidacloprid"]},
    {"q": "Where and when is Rice grown?", "gt": ["West Bengal", "Punjab", "Kharif"]},
    {"q": "In which states and seasons is Wheat cultivated?", "gt": ["Punjab", "Haryana", "Rabi"]},
    {"q": "Where and during which season is Cotton grown?", "gt": ["Gujarat", "Maharashtra", "Kharif"]},
    {"q": "In which regions and seasons is Maize cultivated?", "gt": ["Karnataka", "Andhra Pradesh", "Kharif"]},
    {"q": "Where is Mustard grown and in which season?", "gt": ["Rajasthan", "Uttar Pradesh", "Rabi"]},
    {"q": "What are all the problems that affect Rice including pests and diseases?", "gt": ["Stem Borer", "Brown Plant Hopper", "Blast"]},
    {"q": "What pests and diseases affect Cotton?", "gt": ["Bollworm", "Whitefly", "Wilt"]},
    {"q": "What are the major pests and diseases of Wheat?", "gt": ["Aphids", "Rust", "Smut"]},
    {"q": "What problems affect Mustard crop?", "gt": ["Aphids", "Painted Bug", "White Rust"]},
    {"q": "What are all threats to Groundnut?", "gt": ["Leaf Miner", "Tikka Disease"]},
    {"q": "What are symptoms of Rice Blast?", "gt": ["Lesions", "Spots", "Neck Rot"]},
    {"q": "What symptoms does Wheat Rust show?", "gt": ["Orange Pustules", "Yellow Stripes"]},
    {"q": "What are the signs of Cotton Wilt?", "gt": ["Wilting", "Yellowing"]},
    {"q": "What symptoms does Brown Spot disease cause?", "gt": ["Brown Spots", "Lesions"]},
    {"q": "What are symptoms of Sheath Blight in Rice?", "gt": ["Water Soaked Lesions"]},
    {"q": "What type of soil does Rice grow in?", "gt": ["Clay", "Loamy", "Alluvial"]},
    {"q": "What soil is suitable for Wheat?", "gt": ["Loamy", "Clay Loam"]},
    {"q": "What irrigation does Cotton need?", "gt": ["Drip Irrigation"]},
    {"q": "What soil does Groundnut grow in?", "gt": ["Sandy Loam"]},
    {"q": "What water requirement does Sugarcane have?", "gt": ["Heavy Irrigation"]},
    {"q": "How can Rice Blast be prevented?", "gt": ["Resistant Varieties", "Tricyclazole"]},
    {"q": "How to prevent Wheat Rust?", "gt": ["Resistant Varieties", "Propiconazole"]},
    {"q": "How can Cotton Wilt be prevented?", "gt": ["Seed Treatment", "Crop Rotation"]},
    {"q": "Which pesticide controls Stem Borer which affects Rice?", "gt": ["Carbofuran", "Chlorpyriphos"]},
    {"q": "What diseases affect crops grown in Punjab?", "gt": ["Rust", "Smut", "Blast"]},
    {"q": "Which crops grown in West Bengal are affected by Blast?", "gt": ["Rice"]},
    {"q": "What pests affect Kharif crops?", "gt": ["Stem Borer", "Bollworm"]},
    {"q": "Which fertilizers are used for Rabi crops?", "gt": ["Nitrogen", "Phosphorus"]},
    {"q": "What crops are grown in Maharashtra?", "gt": ["Cotton", "Sugarcane", "Soybean"]},
    {"q": "Which pests affect crops grown in Rajasthan?", "gt": ["Aphids", "Painted Bug"]},
    {"q": "What are all diseases that affect Rabi crops?", "gt": ["Rust", "Smut"]},
    {"q": "Which crops require Nitrogen fertilizer?", "gt": ["Rice", "Wheat", "Maize", "Cotton"]},
    {"q": "What crops are affected by Aphids?", "gt": ["Wheat", "Mustard", "Cotton"]},
]

# ──────────────────────────────────────────────────────────────────────────
# SHARED UTILITIES
# ──────────────────────────────────────────────────────────────────────────
def run_query(cypher, params=None):
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), max_connection_lifetime=60)
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(cypher, **(params or {}))
            return [dict(r) for r in result]
    except Exception as e:
        print(f"Query error: {e}")
        return []
    finally:
        driver.close()

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0
        )
    return _llm

_all_entities = None
def get_all_entities():
    global _all_entities
    if _all_entities is None:
        rows = run_query("MATCH (n) RETURN n.name AS name, labels(n)[0] AS type")
        _all_entities = [(r["name"], r["type"]) for r in rows if r["name"]]
    return _all_entities

def evaluate_answer(answer, ground_truth):
    if not answer or answer.startswith("Error:"):
        return 0, 0, 0, 0
    answer_lower = answer.lower()
    gt_lower = [g.lower() for g in ground_truth]
    tp = sum(1 for g in gt_lower if g in answer_lower)
    fp = max(0, len(answer_lower.split()) // 12 - tp)
    fn = len(gt_lower) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = 1.0 if tp >= max(1, len(gt_lower) // 2) else tp / max(1, len(gt_lower))
    return round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3)

# ──────────────────────────────────────────────────────────────────────────
# SYSTEM 1: GEMINI BASELINE (Standalone LLM, no grounding)
# ──────────────────────────────────────────────────────────────────────────
def gemini_baseline_answer(question):
    prompt = f"""You are an agricultural expert. Answer the following question based on your general knowledge.
Be specific and list relevant items.

Question: {question}

Answer:"""
    return get_llm().invoke(prompt).content

# ──────────────────────────────────────────────────────────────────────────
# SYSTEM 2: THINK-ON-GRAPH (ToG) — Sun et al. 2023, arXiv:2307.07697
# Simplified reimplementation of the core ToG algorithm:
#   - Beam search over graph relations starting from matched entities
#   - At each hop, LLM prunes relations/entities to keep most relevant
#   - Iterates up to max_depth hops, then LLM reasons over collected paths
# ──────────────────────────────────────────────────────────────────────────
def tog_fuzzy_entities(question, all_entities, threshold=80):
    entity_names = [e[0] for e in all_entities]
    words = re.findall(r"[A-Za-z]{4,}", question)
    matched = []
    for w in words:
        m = process.extractOne(w, entity_names, scorer=fuzz.token_sort_ratio)
        if m and m[1] >= threshold and m[0] not in matched:
            matched.append(m[0])
    return matched

def tog_get_neighbors(entity_name, depth_limit=20):
    rows = run_query("""
        MATCH (n)-[r]->(m)
        WHERE n.name = $name
        RETURN n.name AS source, type(r) AS relation, m.name AS target, labels(m)[0] AS target_type
        LIMIT $limit
    """, {"name": entity_name, "limit": depth_limit})
    return rows

def tog_llm_prune(question, candidate_paths, top_k=10):
    """LLM scores and prunes candidate relation paths (core ToG beam search step)."""
    if not candidate_paths:
        return []
    paths_str = "\n".join([f"{i}: {p['source']} -[{p['relation']}]-> {p['target']}" for i, p in enumerate(candidate_paths)])
    prompt = f"""Given the question and candidate knowledge graph paths, select the {top_k} MOST RELEVANT path indices.
Return ONLY a comma-separated list of indices, nothing else.

Question: {question}

Candidate paths:
{paths_str}

Most relevant indices (comma-separated):"""
    response = get_llm().invoke(prompt).content
    try:
        indices = [int(x.strip()) for x in re.findall(r"\d+", response)]
        return [candidate_paths[i] for i in indices if i < len(candidate_paths)][:top_k]
    except:
        return candidate_paths[:top_k]

def tog_answer(question, all_entities, max_depth=2, width=3):
    """Think-on-Graph: iterative beam search reasoning over the KG."""
    start_entities = tog_fuzzy_entities(question, all_entities)
    if not start_entities:
        prompt = f"Answer this agricultural question based on general knowledge: {question}\nAnswer:"
        return get_llm().invoke(prompt).content

    all_paths = []
    frontier = start_entities[:width]

    for depth in range(max_depth):
        candidates = []
        for entity in frontier:
            candidates.extend(tog_get_neighbors(entity))
        if not candidates:
            break
        pruned = tog_llm_prune(question, candidates, top_k=width * 3)
        all_paths.extend(pruned)
        frontier = list(set([p["target"] for p in pruned]))[:width]
        if not frontier:
            break

    if not all_paths:
        prompt = f"Answer this agricultural question based on general knowledge: {question}\nAnswer:"
        return get_llm().invoke(prompt).content

    context_str = "\n".join([f"{p['source']} -[{p['relation']}]-> {p['target']}" for p in all_paths])
    prompt = f"""Based on the following knowledge graph reasoning paths, answer the question.

Reasoning Paths:
{context_str}

Question: {question}

Answer:"""
    return get_llm().invoke(prompt).content



# ──────────────────────────────────────────────────────────────────────────
# SYSTEM 3: PROPOSED GRAPHRAG (fuzzy matching + intent + targeted Cypher + fallback)
# ──────────────────────────────────────────────────────────────────────────
def proposed_understand_query(question, all_entity_names):
    sample = ", ".join(all_entity_names[:100])
    prompt = f"""Analyze this agricultural question. Known entities sample: {sample}

Return ONLY JSON:
{{"intent":"pest_query|disease_query|pesticide_query|location_query|season_query|nutrient_query|general_query",
"crops":[], "pests":[], "diseases":[], "locations":[]}}

Question: {question}
JSON:"""
    response = get_llm().invoke(prompt).content
    try:
        clean = re.sub(r"```json|```", "", response).strip()
        return json.loads(clean)
    except:
        return {"intent": "general_query", "crops": [], "pests": [], "diseases": [], "locations": []}

def proposed_resolve(parsed, all_entities):
    entity_names = [e[0] for e in all_entities]
    type_map = {e[0]: e[1] for e in all_entities}
    resolved = {}
    for key, default_type in [("crops","Crop"),("pests","Pest"),("diseases","Disease"),("locations","Location")]:
        for name in parsed.get(key, []):
            m = process.extractOne(name, entity_names, scorer=fuzz.token_sort_ratio)
            if m and m[1] >= 78:
                resolved[m[0]] = type_map.get(m[0], default_type)
    return resolved

def proposed_query_graph(intent, resolved, parsed):
    results = []
    crops    = [n for n,t in resolved.items() if t=="Crop"]
    pests    = [n for n,t in resolved.items() if t=="Pest"]
    diseases = [n for n,t in resolved.items() if t=="Disease"]

    rel_map = {
        "pest_query": ("AFFECTED_BY", "Pest"),
        "disease_query": ("AFFECTED_BY", "Disease"),
        "location_query": ("GROWN_IN", None),
        "season_query": ("GROWN_DURING", "Season"),
        "nutrient_query": ("REQUIRES", None),
    }

    if intent == "pesticide_query":
        for crop in crops:
            results.extend(run_query("""
                MATCH (c:Crop)-[:CONTROLLED_BY]->(p) WHERE toLower(c.name) CONTAINS toLower($n)
                RETURN c.name AS source,'CONTROLLED_BY' AS relation,p.name AS target,labels(p)[0] AS target_type LIMIT 15
            """, {"n": crop}))
        for pest in pests:
            results.extend(run_query("""
                MATCH (c:Crop)-[:AFFECTED_BY]->(p:Pest) WHERE toLower(p.name) CONTAINS toLower($n)
                WITH c MATCH (c)-[:CONTROLLED_BY]->(x)
                RETURN c.name AS source,'CONTROLLED_BY' AS relation,x.name AS target,labels(x)[0] AS target_type LIMIT 15
            """, {"n": pest}))
        for disease in diseases:
            results.extend(run_query("""
                MATCH (c:Crop)-[:AFFECTED_BY]->(d:Disease) WHERE toLower(d.name) CONTAINS toLower($n)
                WITH c MATCH (c)-[:CONTROLLED_BY]->(x)
                RETURN c.name AS source,'CONTROLLED_BY' AS relation,x.name AS target,labels(x)[0] AS target_type LIMIT 15
            """, {"n": disease}))
    elif intent in rel_map:
        rel, ttype = rel_map[intent]
        for crop in crops:
            type_filter = f":{ttype}" if ttype else ""
            results.extend(run_query(f"""
                MATCH (c:Crop)-[:{rel}]->(x{type_filter}) WHERE toLower(c.name) CONTAINS toLower($n)
                RETURN c.name AS source,'{rel}' AS relation,x.name AS target,labels(x)[0] AS target_type LIMIT 20
            """, {"n": crop}))

    if not results:
        names = list(resolved.keys())
        if names:
            results = run_query("""
                MATCH (n)-[rel]->(m) WHERE n.name IN $names OR m.name IN $names
                RETURN n.name AS source, type(rel) AS relation, m.name AS target, labels(m)[0] AS target_type LIMIT 25
            """, {"names": names})
    return results

def proposed_graphrag_answer(question, all_entities, all_entity_names):
    parsed   = proposed_understand_query(question, all_entity_names)
    intent   = parsed.get("intent", "general_query")
    resolved = proposed_resolve(parsed, all_entities)
    graph_context = proposed_query_graph(intent, resolved, parsed)

    if graph_context:
        context_str = "\n".join([f"{r['source']} -[{r['relation']}]-> {r['target']}" for r in graph_context])
        prompt = f"""Based ONLY on this knowledge graph data, answer concisely:

{context_str}

Question: {question}
Answer:"""
        return get_llm().invoke(prompt).content
    else:
        prompt = f"Answer this agricultural question (no graph data found, use general knowledge): {question}\nAnswer:"
        return get_llm().invoke(prompt).content


# Lightweight LangChain GraphCypher wrapper used in evaluation
def langchain_cypher_answer(question):
    """Run a Cypher-based QA using LangChain's GraphCypherQAChain. Falls back to general LLM answer on error."""
    try:
        neo4j_graph = Neo4jGraph(neo4j_connection_string=NEO4J_URI,
                                 username=NEO4J_USERNAME,
                                 password=NEO4J_PASSWORD,
                                 database=NEO4J_DATABASE)
        chain = GraphCypherQAChain.from_llm(get_llm(), graph=neo4j_graph)
        # chain.run typically accepts the question string
        return chain.run(question)
    except Exception as e:
        # fallback to general LLM answer so evaluation can continue
        return get_llm().invoke(f"(GraphCypher failed: {e}) Answer this agricultural question: {question}\nAnswer:").content

# ──────────────────────────────────────────────────────────────────────────
# MAIN EVALUATION LOOP
# ──────────────────────────────────────────────────────────────────────────
def run_evaluation(n_questions=100, output_csv="evaluation_results.csv"):
    all_entities     = get_all_entities()
    all_entity_names = [e[0] for e in all_entities]

    results = []
    questions = TEST_QUESTIONS[:n_questions]

    for i, item in enumerate(questions):
        q, gt = item["q"], item["gt"]
        print(f"[{i+1}/{len(questions)}] {q}")

        try:
            ans_gemini = gemini_baseline_answer(q)
        except Exception as e:
            ans_gemini = f"Error: {e}"

        try:
            ans_tog = tog_answer(q, all_entities)
        except Exception as e:
            ans_tog = f"Error: {e}"

        try:
            ans_cypher = langchain_cypher_answer(q)
        except Exception as e:
            ans_cypher = f"Error: {e}"

        try:
            ans_proposed = proposed_graphrag_answer(q, all_entities, all_entity_names)
        except Exception as e:
            ans_proposed = f"Error: {e}"

        for model_name, ans in [
            ("Gemini Baseline Standalone Gemini 2.5 Flash", ans_gemini),
            ("Think-on-Graph (ToG)", ans_tog),
            ("LangChain GraphCypherQAChain", ans_cypher),
            ("Proposed GraphRAG (ours)", ans_proposed),
        ]:
            acc, prec, rec, f1 = evaluate_answer(ans, gt)
            results.append({
                "question": q, "model": model_name,
                "answer": ans[:200], "accuracy": acc,
                "precision": prec, "recall": rec, "f1": f1
            })

        time.sleep(0.5)  # rate limit safety

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved detailed results to {output_csv}")

    summary = df.groupby("model")[["accuracy","precision","recall","f1"]].mean().round(3)
    print("\n=== SUMMARY ===")
    print(summary)
    summary.to_csv("evaluation_summary.csv")

    return df, summary

def generate_latex_table(summary_df):
    order = ["Gemini Baseline Standalone Gemini 2.5 Flash",
         "Think-on-Graph (ToG)",
         "LangChain GraphCypherQAChain",
         "Proposed GraphRAG (ours)"]
    rows = []
    for model in order:
        if model in summary_df.index:
            row = summary_df.loc[model]
            bold = model == "Proposed GraphRAG (ours)"
            name = r"\textbf{Proposed Explainable GraphRAG (ours)}" if bold else model
            
            if bold:
                rows.append(f"{name} & \\textbf{{{row['accuracy']*100:.1f}\\%}} & \\textbf{{{row['precision']*100:.1f}\\%}} & \\textbf{{{row['recall']*100:.1f}\\%}} & \\textbf{{{row['f1']*100:.1f}\\%}} \\\\")
            else:
                rows.append(f"{name} & {row['accuracy']*100:.1f}\\% & {row['precision']*100:.1f}\\% & {row['recall']*100:.1f}\\% & {row['f1']*100:.1f}\\% \\\\")

    latex = r"""\begin{table}[h]
\centering
\caption{Performance Comparison with Real-World Baseline Systems}
\label{tab:performance_v2}
\begin{tabular}{|l|c|c|c|c|}
\hline
\textbf{Method} & \textbf{Accuracy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-score} \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
    with open("results_table.tex", "w") as f:
        f.write(latex)
    print("\n=== LATEX TABLE ===")
    print(latex)
    return latex

if __name__ == "__main__":
    df, summary = run_evaluation(n_questions=100)
    generate_latex_table(summary)