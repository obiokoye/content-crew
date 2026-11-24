# Retrieval-Augmented Generation (RAG)

RAG answers questions from your own documents by retrieving relevant passages first and adding them to the prompt.

The usual pipeline is: split documents into chunks, embed each chunk as a vector, store the vectors, then at question time embed the question and fetch the closest chunks.

Retrieval quality matters more than model size. Bad chunks in means a confident but wrong answer out.

Good RAG systems return citations (file and page) so a reader can check every claim.

RAG keeps private data out of model training and lets you update knowledge by re-indexing documents, with no fine-tuning.

Common failure modes are chunks that are too large, missing metadata, and questions that need several documents combined.
