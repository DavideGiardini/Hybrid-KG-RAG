# Hybrid-KG-RAG
Unveiling Hidden Information in Unstructured Documents: Organization and Hybrid Retrieval with Knowledge Graphs.

<p>
							<h4>Goal</h4>
							We generalize the informational content of a document along three distinct dimensions:
						</p>
						<ol>
							<li><strong>Content</strong>: The textual content of the document.</li>
							<li><strong>Structure</strong>: The hierarchical organization of the document, including headings, paragraphs, and sections.</li>
							<li><strong>Entities</strong>: The entities mentioned within the documents and their semantic relations.</li>
						</ol>
						<p>
							Given this categorization, our objective is to design a retrieval system, ranging from document structuring to the retrieval model itself, that can determine the most relevant chunks by integrating and combining all these informational dimensions.
							Specifically, the main objectives of this thesis are the following:
						</p>
						<ol>
							<li>Document Structuring: Developing a Knowledge Graph (KG) representation that effectively models a collection of unstructured documents, focusing on retaining and making explicit all of their informational dimensions: both the textual content and the more implicit metadata, such as the organization of passages and the relationships between entities.</li>
							<li>
								Hybrid Retrieval: Designing a retrieval system capable of leveraging this enriched structure, and therefore taking full advantage of all three informational dimensions to retrieve the most relevant chunks comprehensively.<br>
								Investigating which retrieval methods are more critical in identifying the most relevant passages, and consequently design the system to balance the weights in favor of those methods.
							</li>
							<li>Comparison with traditional RAG: Understanding whether structuring and enriching documents can offer a valuable benefit in RAG systems.</li>
						</ol>
						<p>
							<h4>Methodology</h4> The work proceeds in two stages:
						</p>
						<ul>
							<li><strong>Knowledge-Graph Structuring.</strong> A custom parser converts 5 k+ Wikipedia pages into a Neo4j graph that simultaneously stores:
								<ul>
									<li><em>Chunk</em> nodes for textual <em>content</em></li>  
									<li><em>SubChapter</em> nodes that preserve the document hierarchy, and <em>structure</em></li>
									<li><em>Entity</em> nodes linked by semantic relations harvested from DBpedia to preserve <em>relations</em></li>
								</ul>
								This representation makes content, layout, and entity links explicitly query-able.</li>
							<li><strong>Hybrid Retrieval Engine.</strong> Six complementary retrievers, covering page structure, entity sub-graphs, and dense-vector similarity, score each chunk from different viewpoints.  
								A lightweight neural network learns to fuse these scores, weighting each retriever according to its real contribution to relevance rather than treating them equally.</li>
						</ul>
						<p>
							<h4>Results.</h4> On the held-out split of Google NQ and on a 100-question synthetic Multi-Hop set, the hybrid approach:
							<ul>
								<li>surpasses the best single retriever and a “naive” graph + vector baseline on <em>all</em> metrics.</li>
								<li>delivers the highest F-scores (F<sub>1-3</sub>) for both single-hop and multi-hop queries.</li>
								<li>yields noticeably better RAG answers when assessed with ROUGE, cosine similarity, and RAGAS LLM judgments, without retraining for multi-hop tasks.</li>
							</ul></p>
