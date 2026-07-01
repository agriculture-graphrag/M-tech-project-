import tiktoken


class SemanticChunker:
   

    def __init__(
        self,
        model="gpt-4o-mini",
        chunk_size=2500,
        overlap=300
    ):

        self.encoding = tiktoken.encoding_for_model(model)

        self.chunk_size = chunk_size

        self.overlap = overlap

    def count_tokens(self, text):

        return len(self.encoding.encode(text))

    def chunk(self, text):

        tokens = self.encoding.encode(text)

        chunks = []

        start = 0

        while start < len(tokens):

            end = start + self.chunk_size

            chunk_tokens = tokens[start:end]

            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(chunk_text)

            start += self.chunk_size - self.overlap

        return chunks

    def chunk_document(self, document):

        title = document.get("title", "")

        abstract = document.get("abstract", "")

        body = document.get("body", "")

        full_text = f"""
TITLE:
{title}

ABSTRACT:
{abstract}

BODY:
{body}
"""

        return self.chunk(full_text)


if __name__ == "__main__":

    sample = {
        "title": "Rice Blast Disease",
        "abstract": "Rice blast is caused by Magnaporthe oryzae.",
        "body": "Rice blast disease affects rice crop in humid climate. Tricyclazole is widely used."
    }

    chunker = SemanticChunker()

    chunks = chunker.chunk_document(sample)

    print(f"Total Chunks : {len(chunks)}")

    for i, c in enumerate(chunks):

        print("=" * 80)

        print(f"Chunk {i+1}")

        print(c[:500])