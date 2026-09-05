import sqlite3
import os
import chromadb
import pdfplumber

def setup_sqlite():
    conn = sqlite3.connect("standards.db")
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS standards_meta (
        is_number TEXT PRIMARY KEY,
        title TEXT,
        status TEXT,
        last_updated TEXT,
        scheme TEXT,
        withdrawal_date TEXT,
        transition_deadline TEXT,
        contact_dept TEXT,
        contact_phone TEXT,
        contact_email TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS crs_licenses (
        r_number TEXT PRIMARY KEY,
        brand TEXT,
        manufacturer TEXT,
        product TEXT,
        is_number TEXT,
        status TEXT,
        valid_till TEXT
    )
    """)

    cur.execute("DELETE FROM standards_meta")
    cur.execute("""
    INSERT INTO standards_meta VALUES 
    (
        'IS 16102 (Part 1): 2026',
        'Self-ballasted LED lamps for general lighting services - Part 1: Safety requirements',
        'Active',
        '02 February 2026',
        'Compulsory Registration Scheme (CRS) - Scheme II',
        'N/A',
        '02 February 2027',
        'CRS Registration Department, Room 407, Manakalya Building, BIS, New Delhi',
        '+91-11-23230856',
        'registration@bis.org.in'
    ),
    (
        'IS 16102 (Part 1): 2012',
        'Self-ballasted LED lamps for general lighting services - Part 1: Safety requirements',
        'Withdrawn',
        '02 August 2026',
        'Compulsory Registration Scheme (CRS) - Scheme II',
        '02 August 2026',
        '02 February 2027 (Extension for registered manufacturers)',
        'CRS Registration Department',
        '+91-11-23230856',
        'registration@bis.org.in'
    )
    """)

    cur.execute("DELETE FROM crs_licenses")
    cur.execute("""
    INSERT INTO crs_licenses VALUES 
    ('R-41014095', 'Philips / Signify', 'Signify Innovations India Ltd', 'Self-Ballasted LED Lamp', 'IS 16102 (Part 1)', 'Valid', '2027-11-30'),
    ('R-41028392', 'Havells', 'Havells India Limited', 'Self-Ballasted LED Lamp', 'IS 16102 (Part 1)', 'Valid', '2028-03-15'),
    ('R-99012345', 'Counterfeit Brand', 'Unregistered Workshop', 'LED Lamp', 'IS 16102 (Part 1)', 'Cancelled / Expired', '2024-01-01')
    """)

    conn.commit()
    conn.close()
    print("✓ SQLite database 'standards.db' successfully populated.")

def extract_pdf_chunks(pdf_path):
    chunks = []
    if os.path.exists(pdf_path):
        print(f"Ingesting PDF: {pdf_path}")
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            paragraphs = text.split("\n\n")
            current_chunk = ""
            for p in paragraphs:
                clean_p = p.strip()
                if not clean_p:
                    continue
                if len(current_chunk) + len(clean_p) < 1100:
                    current_chunk += clean_p + "\n"
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = clean_p + "\n"
            if current_chunk:
                chunks.append(current_chunk.strip())
    return chunks

def setup_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("bis_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="bis_knowledge",
        metadata={"hnsw:space": "cosine"}
    )

    pdf_dir = "data/raw_standards"
    all_chunks = []
    if os.path.exists(pdf_dir):
        for file in sorted(os.listdir(pdf_dir)):
            if file.endswith(".pdf"):
                extracted = extract_pdf_chunks(os.path.join(pdf_dir, file))
                all_chunks.extend(extracted)

    # Core grounded fallbacks derived from your dataset
    if len(all_chunks) < 5:
        all_chunks.extend([
            "IS 16102 (Part 1): 2026 specifies safety and interchangeability requirements for self-ballasted LED lamps up to 60W, covering voltages 50V to 250V AC and up to 1000V DC. Cap types include E14, E27, B22d, and B15d. The 2026 edition explicitly brings rechargeable battery LED lamps, multi-function smart lamps, and photobiological safety under mandatory scope.",
            "Electronics and IT Goods (Compulsory Registration) Order 2012 issued under Section 10(1)(p) of BIS Act 1986 mandates that no person shall manufacture, store, import, or sell notified goods without conforming to specified Indian Standards and obtaining BIS registration. Defective and non-conforming goods must be deformed beyond use and scrapped.",
            "Gazette Notification HQ-PUB013/1/2020-PUB-BIS established IS 16102 (Part 1):2026 on 02 February 2026. The previous standard IS 16102 (Part 1):2012 had concurrent validity until 02 August 2026, after which it is formally Withdrawn. A BIS circular grants manufacturers an implementation transition window until 02 February 2027 to align existing registrations.",
            "Standard Mark specifications require registered products under CRS to display either the official interlocking S-monogram or the words 'Self Declaration - Conforming to IS 16102 (Part 1)' along with the assigned 8-digit Registration Number in the exact format R-XXXXXXXX. Minimum text font size is Arial 6.",
            "CRS Certification Process: 1. Sample testing at a BIS-recognized laboratory (test reports must not be older than 90 days). 2. Online application on the BIS portal with Form VI. 3. Foreign applicants must nominate an Authorized Indian Representative (AIR). 4. Hard copies submitted within 15 days of online filing. Registration is valid for 2 years.",
            "Sectional Committee ETD 23 (Lamps and Related Equipments) governs standard formulation, aligned with IEC 62560. Chaired by HOD Asit Kumar Maharana and Member Secretary Jainendra Kumar with 31 member organizations including Bajaj, Havells, Signify, CPRI, and consumer groups."
        ])

    ids = [f"chunk_{i+1}" for i in range(len(all_chunks))]
    metadatas = [{"source": "bis_gazette_corpus", "index": i} for i in range(len(all_chunks))]

    collection.add(
        documents=all_chunks,
        metadatas=metadatas,
        ids=ids
    )
    print(f"✓ ChromaDB initialized with {len(all_chunks)} vector knowledge chunks.")

if __name__ == "__main__":
    setup_sqlite()
    setup_chroma()