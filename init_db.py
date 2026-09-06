import sqlite3
import os
import warnings
import chromadb

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


CHROMA_PATH = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

PDF_DIRECTORY = "data/raw_standards"
COLLECTION_NAME = "bis_knowledge"


def setup_sqlite():
    conn = sqlite3.connect("standards.db")
    cur = conn.cursor()

    # Drop existing tables to avoid column mismatch errors
    cur.execute("DROP TABLE IF EXISTS standards_meta")
    cur.execute("DROP TABLE IF EXISTS crs_licenses")

    cur.execute("""
    CREATE TABLE standards_meta (
        is_number TEXT PRIMARY KEY,
        title TEXT,
        status TEXT,
        last_updated TEXT,
        scheme TEXT,
        transition_deadline TEXT,
        contact_dept TEXT,
        contact_phone TEXT,
        contact_email TEXT
    )
    """)

    # License information
    cur.execute("""
    CREATE TABLE crs_licenses (
        r_number TEXT PRIMARY KEY,
        brand TEXT,
        manufacturer TEXT,
        product TEXT,
        is_number TEXT,
        status TEXT,
        valid_till TEXT
    )
    """)

    standards_data = [
        (
            'IS 16102 (Part 1): 2026',
            'Self-ballasted LED lamps for general lighting services - Safety requirements',
            'Active',
            '02 February 2026',
            'Compulsory Registration Scheme (CRS) - Scheme II',
            '02 February 2027 (Transition extension for manufacturers)',
            'CRS Registration Department, Room 407, Manakalya Building, BIS, New Delhi',
            '+91-11-23230856',
            'registration@bis.org.in'
        ),
        (
            'IS 16102 (Part 1): 2012',
            'Self-ballasted LED lamps for general lighting services - Safety requirements',
            'Withdrawn',
            '02 August 2026',
            'Compulsory Registration Scheme (CRS) - Scheme II',
            '02 February 2027',
            'CRS Registration Department',
            '+91-11-23230856',
            'registration@bis.org.in'
        ),
        (
            'IS 13252 (Part 1): 2010',
            'Information Technology Equipment - Safety - General Requirements',
            'Active (Reaffirmed Dec 2025)',
            'March 2015 (Amendment 2 in force)',
            'Compulsory Registration Scheme (CRS) - Scheme II',
            'Fully Enforced (Amendment 2 mandatory since 14 May 2017)',
            'LITD 07 / CRS Registration Department',
            '+91-11-23230856',
            'registration@bis.org.in'
        )
    ]

    cur.executemany("""
    INSERT INTO standards_meta (
        is_number, title, status, last_updated, scheme,
        transition_deadline, contact_dept, contact_phone, contact_email
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, standards_data)

    licenses_data = [
        (
            'R-41014095',
            'Philips / Signify',
            'Signify Innovations India Ltd',
            'Self-Ballasted LED Lamp',
            'IS 16102 (Part 1)',
            'Valid',
            '2027-11-30'
        ),
        (
            'R-41028392',
            'Havells',
            'Havells India Limited',
            'Self-Ballasted LED Lamp',
            'IS 16102 (Part 1)',
            'Valid',
            '2028-03-15'
        ),
        (
            'R-41013730',
            'Apple',
            'Hong Fu Jin Precision Electronics (Zhengzhou) Co., Ltd.',
            'Mobile Phone',
            'IS 13252(Part 1) & IS 16333(Part 3)',
            'Registered',
            '2029-04-15'
        ),
        (
            'R-41013870',
            'Apple',
            'Luxsan Precision Industry (Kunshan) Co., Ltd.',
            'Mobile Phone',
            'IS 13252(Part 1) & IS 16333(Part 3)',
            'Registered',
            '2029-04-20'
        ),
        (
            'R-41013706',
            'Micromax / YU',
            'Shenzhen Sprocomm Communication Equipment Co., Ltd.',
            'Mobile Phone',
            'IS 13252(Part 1)',
            'Expired',
            '2023-04-15'
        ),
        (
            'R-41012416',
            'Lenovo',
            'Lenovo Mobile Communication Technology Ltd.',
            'Mobile Phone',
            'IS 13252(Part 1)',
            'Cancelled',
            '2019-02-24'
        ),
        (
            'R-99012345',
            'Counterfeit Brand',
            'Unregistered Workshop',
            'LED Lamp',
            'IS 16102 (Part 1)',
            'Cancelled / Fake',
            '2024-01-01'
        )
    ]

    cur.executemany("""
    INSERT INTO crs_licenses (
        r_number, brand, manufacturer, product, is_number, status, valid_till
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, licenses_data)

    conn.commit()
    conn.close()

    print("✓ SQLite database 'standards.db' successfully seeded.")


def setup_chroma_with_nomic():
    pdf_dir = PDF_DIRECTORY

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)

    print(f"1. Loading PDFs from {pdf_dir} using LangChain...")

    loader = PyPDFDirectoryLoader(pdf_dir)
    docs = loader.load()

    if not docs:
        print(
            "Warning: No PDFs found in data/raw_standards! "
            "Please verify PDF placement."
        )
        return

    # Split documents into smaller overlapping chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )

    split_docs = text_splitter.split_documents(docs)

    print(f"✓ Created {len(split_docs)} semantic chunks.")

    # Connect to Chroma and remove old collection if it exists
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
        print("✓ Old Chroma collection removed.")
    except Exception:
        pass

    print(
        f"2. Initializing Ollama local embeddings "
        f"({EMBEDDING_MODEL})..."
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    print("3. Indexing vectors in local Chroma store...")

    Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH
    )

    print(
        f"✓ Success: Vector collection embedded with "
        f"{EMBEDDING_MODEL} and saved to {CHROMA_PATH}."
    )


if __name__ == "__main__":
    setup_sqlite()
    setup_chroma_with_nomic()
