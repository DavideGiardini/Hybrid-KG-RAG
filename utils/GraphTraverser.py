import Levenshtein
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
from neo4j import GraphDatabase
import config.EnvLoader as el

URI = "neo4j://localhost"
AUTH = ("neo4j", el.NEO4J_PWD)

system_prompt = """You are going to be provided with a context consisting of multiple Wikipedia pages, along with all of their chapters and supchapters.
If existent, return the name of the page + the name of the chapter or subchapter in which the answer to the user's question can be found.
If the question is about something generic of the topic of the page, return the title of the page.
If the question is specific and none of the pages' chapters and subchapters fits it precisely, return 'ND'.
If the answer could be found in multiple pages or subchapters, or you are undecided between multiple options, respond with a list of ALL the identifiers of the relevant pages or subchapters, separated via a semicolon.
If the question needed multiple subchapters to be answered, respond with a list of ALL the identifiers of the subchapters used to answer the question, separated via a semicolon.
In any case, return the answer without providing any comments."""

def cos_sim(a,b):
    return np.dot(a, b)/(norm(a)*norm(b))

def subgraph_extractor(query_embed,n_starting_nodes:int=3,verbose:bool=False,max_nodes:int=5):
    """
    Ricerca all'interno del grafo un sub-graph rilevante rispetto all'embedding fornito in query_embed (q).
    Trova gli n (con n = n_starting_nodes) nodi pagina più rilevanti.
    Utilizza ciascuno di questi nodi (a) come nodo di partenza per la ricerca di un sub-graph.
    Costruisce il sottografo aggiungendo al nodo di partenza a, per ogni iterazione, un nodo b che sia il più rilevante all'embedding originale.
    La rilevanza di un nodo b viene definita come la cosine similarity tra q e un secondo embedding costruito come media tra a e gli embedding
    di tutte le relazioni attraversate per giungere al nodo b.
    Infine, calcola lo score di ogni sottografo come media di tutti gli score accumulati durante la sua costruzione, e restituisce il sottografo con score più alto.

    Restituisce in output:
    - Una lista degli uri dei nodi facenti parte del sottografo
    - Lo score di rilevanza del sottografo
    - Una lista di stringhe da utilizzare come contesto nelle chiamate ai LLM
    """
    # Search for starting nodes
    if verbose: print("Searching for starting nodes.")
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        starting_nodes, _, _ = driver.execute_query(
            """CALL db.index.vector.queryNodes('page_vector', 100, $embedding)
            YIELD node, score
            RETURN node.uri, node, score
            LIMIT $n_starting_nodes""",
            embedding=query_embed, n_starting_nodes=n_starting_nodes
        )
    if verbose: print("starting nodes: " + str([x["node"]["uri"] for x in starting_nodes]))
    subgraphs = {}
    for starting_node in starting_nodes:
        # Inizializziamo tutte le variabili necessarie inserendo le info di A
        prompt = []
        memory = {
            starting_node["node.uri"]: {
                "embeddings": [starting_node["node"]["embedding"]],
                "score": starting_node["score"]
                }
            }
        node_uri_list = [starting_node["node.uri"]]
        subgraphs_scores = [starting_node["score"]]
        # Cerchiamo i nuovi nodi partendo da A
        for i in range(max_nodes-1):
            with GraphDatabase.driver(URI, auth=AUTH) as driver:
                connected_relations, _, _ = driver.execute_query(
                    """WITH $node_list AS uri_list
                    MATCH (a:Page)-[r:vector_rel]->(b:Page)
                    WHERE a.uri IN uri_list and NOT b.uri IN uri_list
                    RETURN DISTINCT ID(a),a.uri,ID(r),r.embedding,r.text,ID(b),b.embedding,b.uri""",
                    node_list=node_uri_list
                )
            top_score = 0
            if not connected_relations:
                break
            # Per ciascuna relazione trovata, la inseriamo in memoria e cerchiamo il nodo B con score più alto
            for rel in connected_relations:
                if rel["b.uri"] not in memory.keys():
                    embedding_lst = memory[rel["a.uri"]]["embeddings"] + [rel["r.embedding"]]
                    memory[rel['b.uri']] = {
                        "embeddings": embedding_lst,
                        "score": cos_sim(np.mean(np.array(embedding_lst), axis=0), query_embed)
                    }
                score = memory[rel["b.uri"]]["score"]
                if score > top_score:
                    top_score = score
                    best_candidate = rel
            # Inseriamo il nodo B all'interno del sottografo
            node_uri_list.append(best_candidate["b.uri"])
            subgraphs_scores.append(top_score)
            prompt.append(best_candidate["a.uri"] + "#" + best_candidate["r.text"] + ":" + best_candidate["b.uri"])
        # Ora che abbiamo costruito il subgraph, salviamo le sue informazioni nel dizionario
        subgraphs[starting_node["node.uri"]] = {
            "nodes": node_uri_list,
            "score": sum(subgraphs_scores)/len(subgraphs_scores),
            "prompt": prompt
        }
    # Ordiniamo il dizionario dei subgraphs sulla base del loro score
    subgraphs = sorted(subgraphs.values(), key=lambda d: d['score'], reverse=True)
    if verbose: print(subgraphs)
    return subgraphs[0]["nodes"], subgraphs[0]["score"], subgraphs[0]["prompt"]

def subgraph_chunk_finder(chapter_lst:list)->list:
    """
    Chunk_finder per i nodi estratti da LLM costruiti sulla base di relazioni DBpedia.
    Richiede in input la lista dei sottocapitoli selezionati dal LLM.
    Partendo da tale lista, estrae i chunk che contengono le entità citate in tali sottocapitoli.
    - Nel caso esista un chunk che contiene tutte le entità, questo sarà il chunk restituito. Altrimenti:
    - Nel caso una pagina sia una OriginalPage, verrà cercato un chunk di tale original page nel quale vengono citate le altre entità. Altrimenti:
    - Se le pagine sono più di due (altrimenti l'operazione sarebbe analoga alla prima), cerchiamo tutti i chunk che citano almeno due di quelle entità.
      Se esistono chunk per cui tutte le entità vengono citate almeno una volta, questi vengono selezionati. Altrimenti:
    - Vengono selezionati tutti i chunk in cui almeno una di queste entità viene citata.

    Restituisce la lista degli ID dei chunk rilevanti.
    """
    # Il LLM avrà estratto un titolo del tipo: "NodoA#relazione:NodoB". Estraiamo da questo titolo il nome del NodoA e del NodoB.
    nodes = []
    for chapter in chapter_lst:
        nodes.append(chapter.split("#")[0])
        nodes.append(chapter.split(":")[-1])
    nodes = list(set(nodes))
    # Try 1: extract a chunk that cites all of the nodes
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        rel_chunks, _, _ = driver.execute_query(
            """MATCH (c:Chunk)
            WHERE ALL(page_uri IN $pageUris WHERE EXISTS((c)-[:cites]->(:Page {uri: page_uri})))
            RETURN ID(c)""",
            pageUris=nodes
        )
    if rel_chunks:
        return [x["ID(c)"] for x in rel_chunks]
    # Try 2: If one of the pages is an original Page, match the chunk of that page that cites the other pages
    # We will use the is_chunk_of relationship which is, differently from the has_chunk counterpart, transitive
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        OP_nodes, _, _ = driver.execute_query(
            """WITH $pageUris as list
            MATCH (a:OriginalPage)
            WHERE a.uri IN list
            RETURN a.uri""",
            pageUris=nodes
        )
    if OP_nodes:
        OP_nodes = [x["a.uri"] for x in OP_nodes]
        found_for = []
        for OP_node in OP_nodes:
            with GraphDatabase.driver(URI, auth=AUTH) as driver:
                rel_chunks, _, _ = driver.execute_query(
                    """WITH $pageUris as list
                    MATCH (a:OriginalPage {uri: $OP_node})<-[:is_chunk_of]-(c:Chunk)-[:cites]->(b:Page)
                    WHERE b.uri IN list
                    RETURN b.uri, ID(c)""",
                    OP_node=OP_node, pageUris=nodes
                )
        found_for = list(set([x["b.uri"] for x in rel_chunks]))
        if len(found_for) == len(nodes) - 1:
            return [x["ID(c)"] for x in rel_chunks]
    # Try 3: If the pages are more than 2, find the chunks that cites at least two of them
    if len(nodes) > 2:
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            rel_chunks, _, _ = driver.execute_query(
                """WITH $pageUris AS pageUris
                MATCH (c:Chunk)-[:cites]->(b:Page)
                WHERE b.uri IN pageUris
                AND size([page_uri IN pageUris WHERE EXISTS((c)-[:cites]->(:Page {uri: page_uri}))]) >= 2
                RETURN b.uri, ID(c)""",
                pageUris=nodes
            )
        found_for = list(set([x["b.uri"] for x in rel_chunks]))
        if len(found_for) == len(nodes):
            return [x["ID(c)"] for x in rel_chunks]
    # Try 4: All the chunks citing one of the pages
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        rel_chunks, _, _ = driver.execute_query(
            """WITH $pageUris AS pageUris
            MATCH (c:Chunk)-[:cites]->(b:Page)
            WHERE b.uri IN pageUris
            RETURN b.uri, ID(c)""",
            pageUris=nodes
        )
    return [chunk["ID(c)"] for chunk in rel_chunks]

def page_chunk_finder(selected_subchapter_lst:list)->list:
    """
    Chunk_finder per i nodi estratti da LLM che provengono da sottocapitoli di pagine wikipedia.
    Sulla base dei sottocapitoli selezionati dal LLM, ritorna tutti i chunk di tali sottocapitoli.
    """
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        selected_chunks, _, _ = driver.execute_query(
            """WITH $selected_subchapters AS list
            MATCH (a)-[:has_chunk]->(c:Chunk)
            WHERE a.uri IN list
            RETURN ID(c)""",
            selected_subchapters=selected_subchapter_lst
        )
    return [chunk["ID(c)"] for chunk in selected_chunks]

def return_best_candidate(subC_list:list, response:str)->str:
    """
    Data una lista e una stringa, questa funzione trova l'elemento della lista che più si avvicina alla stringa.
    È utilizzata per trovare, data una risposta del LLM, il nodo corrispondente, evitando un exact match che potrbbe essere intaccato da errori di spelling del LLM.
    """
    top_score = 0.5
    best_candidate = None
    for subC in subC_list:
        l_ratio = Levenshtein.ratio(response, subC)
        if l_ratio > top_score:
            top_score = l_ratio
            best_candidate = subC
    return best_candidate


def chapter_selector(text:str, query:str, system_prompt:str, llm, subC_list:list):
    """
    Given the system_prompt, the user's query, the context (all the relevant nodes) and a LLM,
    this functions asks to the LLM which nodes are more relevant to answer the user's query.
    Returns the list of best candidates and the cost of the API call.
    """
    messages = [
        ("system", system_prompt),
        ("human", "Context:\n" + text + "\n\n" + "Query:\n" + query),
    ]
    ai_msg = llm.invoke(messages)
    token_usage = ai_msg.response_metadata["token_usage"]
    # Compute the cost based on GPT4o costs
    cost = int(token_usage["prompt_tokens"])*0.00000250 + int(token_usage["completion_tokens"])*0.00001
    response = ai_msg.content
    #print(response)
    # The model is allowed to respond with "ND" if no node is relevant to the user's query
    if response == "ND":
        return [], cost, "ND"
    # The model is allowed to respond with multiple relevant nodes
    # For each of this node, we use the search for the best candidate between the available nodes, and insert them in the output list
    best_candidate_lst = []
    for el in response.split(";"):
        best_candidate = return_best_candidate(subC_list, el)
        if best_candidate:
            best_candidate_lst.append(best_candidate)
    return best_candidate_lst, cost, response

def prompt_creator(retrieved_OP):
    """
    Prende in input una lista di Original Pages, selezionate come più rilevanti sulla base di una user_query.
    Sulla base di questa lista costruisce il contesto che sarà passato poi al LLM per permettergli di scegliere il sottocapitolo più rilevante
    per rispondere alla domanda.
    Il testo sarà formattato in questo modo:
    1. Titolo prima pagina
    #Sottocapitolo1
    #Sottocapitolo2
    #Sottocapitolo2#Sottocapitolo1
    2. Titolo seconda pagina
    ecc.

    Restituisce in output il testo in questione e la lista di tutti i sottocapitoli (e le OriginalPage) utilizzate alla creazione di tale testo.
    """
    text = ""
    subC_list = [page["node.uri"] for page in retrieved_OP]
    # Create the text prompt from which the LLM will have to select one subchapter
    for idx, OriginalPage in enumerate(retrieved_OP):
        # Write the page title
        text += str(idx + 1) + ". " + OriginalPage["node.uri"] + "\n"
        # Find all the subchapters of the OP
        with GraphDatabase.driver(URI, auth=AUTH) as driver:
            connected_SubC, _, _ = driver.execute_query(
                """MATCH (a:OriginalPage)-[:has_subchapter*]->(b:SubChapter)
                WHERE ID(a) = $id_a
                RETURN b.uri""",
                id_a = OriginalPage["ID(node)"]
            )
        subC_list += [x["b.uri"] for x in connected_SubC]
        # Write every subChapter title in the text
        # Remove the page's title to save tokens
        for SubC in connected_SubC:
            text += SubC["b.uri"].replace(OriginalPage["node.uri"], "") + "\n"
    return text, subC_list

def CypherSearch(query_embed, user_query:str, llm):
    """
    Richiede in input l'embedding della domanda dell'utente e la domanda originale, insieme a un LLM.
    Cerca tutti i sottocapitoli presenti all'interno delle 3 Original Pages più rilevanti alla domanda.
    Cerca tutte le entità presenti all'interno del sottografo più rilevante alla domanda.
    Richiede al LLM di scegliere, tra queste entità/sottocapitoli, quelle necessarie a rispondere alla domanda.
    Per tutti i sottocapitoli scelti, estrae tutti i chunk appartenenti a tale sottocapitolo.
    Per tutte le entità scelte, cerca i chunk più rilevanti sulla base delle tali.
    Restituisce in output la lista dei chunk rilevanti e il costo della chiamata API al LLM.
    """

    # Search for top 3 Original Pages
    with GraphDatabase.driver(URI, auth=AUTH) as driver:
        retrieved_OP, _, _ = driver.execute_query(
            """CALL db.index.vector.queryNodes("original_page_vector", 100, $embedding)
            YIELD node, score
            RETURN node.uri, ID(node), score
            LIMIT 3""",
            embedding=query_embed
        )
    # Get prompt for LLM
    text, subC_list = prompt_creator(retrieved_OP)
    # add to the context of the OPs the context of the subgraph_extractor
    _, _, context = subgraph_extractor(query_embed)
    text += "\n" + "\n".join(context)
    subC_list += context
    # Let the LLM select the subchapter
    selected_subchapter_lst, cost, response = chapter_selector(text, user_query, system_prompt, llm, subC_list)
    if not selected_subchapter_lst:
        return [], cost, response, subC_list
    # Find DBpedia-relevant chunks
    selected_chunks = []
    DBpedia_subc = [x for x in selected_subchapter_lst if x in context]
    if DBpedia_subc:
        selected_chunks += subgraph_chunk_finder(DBpedia_subc)
    # Find pages-relevant chunks
    page_subc = [x for x in selected_subchapter_lst if x not in context]
    if page_subc:
        selected_chunks += page_chunk_finder(selected_subchapter_lst)
    # Write answer in dataset
    return selected_chunks, cost, response, subC_list
