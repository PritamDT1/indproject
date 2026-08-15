from src.document_processor import process_documents


def main():
    print("=" * 60)
    print("MULTI-FORMAT DOCUMENT PROCESSOR TEST")
    print("=" * 60)

    file_paths = input(
        "\nEnter document paths separated by commas:\n"
    ).split(",")

    file_paths = [
        path.strip()
        for path in file_paths
        if path.strip()
    ]

    if not file_paths:
        print("No files provided.")
        return

    try:
        chunks = process_documents(file_paths)

        print("\n" + "=" * 60)
        print("DOCUMENT PROCESSING SUCCESSFUL")
        print("=" * 60)

        print(f"Total documents: {len(file_paths)}")
        print(f"Total chunks: {len(chunks)}")

        sources = sorted(
            set(chunk["source"] for chunk in chunks)
        )

        print("\nDocuments processed:")

        for source in sources:
            print(f"  - {source}")

        print("\nFirst 5 chunks:\n")

        for index, chunk in enumerate(chunks[:5], start=1):
            print("-" * 60)
            print(f"Chunk: {index}")
            print(f"Source: {chunk['source']}")
            print(f"Page: {chunk['page']}")
            print(f"Slide: {chunk['slide']}")
            print(f"Row: {chunk['row']}")

            print("\nText:")
            print(chunk["text"][:500])

        print("-" * 60)

    except Exception as error:
        print("\nERROR:")
        print(error)


if __name__ == "__main__":
    main()