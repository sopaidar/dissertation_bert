"""
Remove Duplicates from Zotero CSV Export
Finds and removes duplicate papers based on DOI, Title, or both

Usage:
    python remove_duplicates.py
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher

# ============================================
# CONFIGURATION
# ============================================

INPUT_FILE = "bd_g.csv"  # Your Zotero CSV file
OUTPUT_FILE = "zotero_cleaned.csv"  # Output file without duplicates
DUPLICATES_FILE = "duplicates_found.csv"  # List of removed duplicates

# Similarity threshold for title matching (0.0 to 1.0)
TITLE_SIMILARITY_THRESHOLD = 0.90  # 90% similar = duplicate

# ============================================
# LOAD DATA
# ============================================

print("=" * 60)
print("DUPLICATE REMOVAL FOR ZOTERO CSV")
print("=" * 60)

print(f"\n📁 Loading data from: {INPUT_FILE}")

try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    print(f"✅ Loaded {len(df)} records")
except FileNotFoundError:
    print(f"❌ Error: Could not find '{INPUT_FILE}'")
    print("Make sure the file is in the same directory as this script")
    exit()

# ============================================
# HELPER FUNCTIONS
# ============================================

def clean_title(title):
    """Clean and normalize title for comparison"""
    if pd.isna(title):
        return ""
    title = str(title).lower()
    # Remove extra spaces
    title = ' '.join(title.split())
    # Remove common punctuation
    for char in [':', ',', '.', ';', '!', '?', '"', "'"]:
        title = title.replace(char, '')
    return title

def title_similarity(title1, title2):
    """Calculate similarity between two titles (0.0 to 1.0)"""
    if pd.isna(title1) or pd.isna(title2):
        return 0.0
    clean1 = clean_title(title1)
    clean2 = clean_title(title2)
    if not clean1 or not clean2:
        return 0.0
    return SequenceMatcher(None, clean1, clean2).ratio()

def normalize_doi(doi):
    """Normalize DOI for comparison"""
    if pd.isna(doi):
        return None
    doi = str(doi).lower().strip()
    # Remove common prefixes
    doi = doi.replace('https://doi.org/', '')
    doi = doi.replace('http://doi.org/', '')
    doi = doi.replace('doi:', '')
    doi = doi.strip()
    return doi if doi else None

# ============================================
# STEP 1: FIND DUPLICATES BY DOI
# ============================================

print("\n" + "=" * 60)
print("STEP 1: Finding duplicates by DOI")
print("=" * 60)

# Normalize DOIs
if 'DOI' in df.columns:
    df['DOI_normalized'] = df['DOI'].apply(normalize_doi)
    
    # Find DOI duplicates (excluding empty DOIs)
    doi_duplicates = df[df['DOI_normalized'].notna() & df['DOI_normalized'].duplicated(keep=False)]
    
    if len(doi_duplicates) > 0:
        print(f"\n📊 Found {len(doi_duplicates)} papers with duplicate DOIs")
        print(f"📊 {doi_duplicates['DOI_normalized'].nunique()} unique DOIs are duplicated")
        
        # Show examples
        print("\nExample DOI duplicates:")
        for doi in doi_duplicates['DOI_normalized'].value_counts().head(3).index:
            count = doi_duplicates[doi_duplicates['DOI_normalized'] == doi].shape[0]
            print(f"  - DOI: {doi} ({count} copies)")
    else:
        print("✅ No DOI duplicates found")
else:
    print("⚠️  No DOI column found in CSV")
    df['DOI_normalized'] = None

# ============================================
# STEP 2: FIND DUPLICATES BY TITLE
# ============================================

print("\n" + "=" * 60)
print("STEP 2: Finding duplicates by Title Similarity")
print("=" * 60)

title_duplicates_indices = []

if 'Title' in df.columns:
    titles = df['Title'].tolist()
    print(f"\n🔍 Comparing {len(titles)} titles...")
    print(f"   (Similarity threshold: {TITLE_SIMILARITY_THRESHOLD*100:.0f}%)")
    
    # Compare all titles (this can be slow for large datasets)
    compared = set()
    
    for i in range(len(titles)):
        if i % 100 == 0:
            print(f"   Progress: {i}/{len(titles)} papers checked", end='\r')
        
        if i in title_duplicates_indices:
            continue
            
        for j in range(i + 1, len(titles)):
            if j in title_duplicates_indices:
                continue
            
            # Skip if already compared
            pair = tuple(sorted([i, j]))
            if pair in compared:
                continue
            compared.add(pair)
            
            # Calculate similarity
            similarity = title_similarity(titles[i], titles[j])
            
            if similarity >= TITLE_SIMILARITY_THRESHOLD:
                title_duplicates_indices.append(j)
                print(f"\n   Found duplicate: {similarity*100:.1f}% similar")
                print(f"      Original [{i}]: {titles[i][:80]}...")
                print(f"      Duplicate [{j}]: {titles[j][:80]}...")
    
    print(f"\n\n📊 Found {len(title_duplicates_indices)} title-based duplicates")
else:
    print("⚠️  No Title column found in CSV")

# ============================================
# STEP 3: COMBINE AND REMOVE DUPLICATES
# ============================================

print("\n" + "=" * 60)
print("STEP 3: Removing Duplicates")
print("=" * 60)

# Mark duplicates
df['is_duplicate'] = False
df['duplicate_reason'] = ''

# Mark DOI duplicates (keep first occurrence)
if 'DOI_normalized' in df.columns:
    doi_dups_mask = df['DOI_normalized'].notna() & df['DOI_normalized'].duplicated(keep='first')
    df.loc[doi_dups_mask, 'is_duplicate'] = True
    df.loc[doi_dups_mask, 'duplicate_reason'] = 'Duplicate DOI'

# Mark title duplicates
df.loc[title_duplicates_indices, 'is_duplicate'] = True
df.loc[df.index.isin(title_duplicates_indices) & (df['duplicate_reason'] == ''), 'duplicate_reason'] = 'Similar Title'

# Both DOI and Title
both_mask = doi_dups_mask & df.index.isin(title_duplicates_indices)
df.loc[both_mask, 'duplicate_reason'] = 'Duplicate DOI & Similar Title'

# Count duplicates
total_duplicates = df['is_duplicate'].sum()
doi_only = (df['duplicate_reason'] == 'Duplicate DOI').sum()
title_only = (df['duplicate_reason'] == 'Similar Title').sum()
both = (df['duplicate_reason'] == 'Duplicate DOI & Similar Title').sum()

print(f"\n📊 Summary:")
print(f"   Total duplicates found: {total_duplicates}")
print(f"   - Duplicate DOI only: {doi_only}")
print(f"   - Similar title only: {title_only}")
print(f"   - Both DOI & title: {both}")
print(f"\n   Original papers: {len(df)}")
print(f"   After deduplication: {len(df) - total_duplicates}")
print(f"   Reduction: {total_duplicates/len(df)*100:.1f}%")

# ============================================
# STEP 4: SAVE RESULTS
# ============================================

print("\n" + "=" * 60)
print("STEP 4: Saving Results")
print("=" * 60)

# Save cleaned dataset (without duplicates)
df_cleaned = df[df['is_duplicate'] == False].copy()

# Drop helper columns
cols_to_drop = ['DOI_normalized', 'is_duplicate', 'duplicate_reason']
for col in cols_to_drop:
    if col in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=[col])

df_cleaned.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ Saved cleaned dataset: {OUTPUT_FILE}")
print(f"   ({len(df_cleaned)} papers)")

# Save list of duplicates for review
if total_duplicates > 0:
    df_duplicates = df[df['is_duplicate'] == True].copy()
    
    # Keep only useful columns
    cols_to_keep = ['Title', 'Publication Year', 'Author', 'DOI', 'Publication Title', 'duplicate_reason']
    cols_available = [col for col in cols_to_keep if col in df_duplicates.columns]
    
    df_duplicates[cols_available].to_csv(DUPLICATES_FILE, index=False)
    print(f"✅ Saved duplicate list: {DUPLICATES_FILE}")
    print(f"   ({total_duplicates} duplicates removed)")
else:
    print("ℹ️  No duplicates file created (no duplicates found)")

# ============================================
# STEP 5: DETAILED STATISTICS
# ============================================

print("\n" + "=" * 60)
print("DETAILED STATISTICS")
print("=" * 60)

# Year distribution comparison
if 'Publication Year' in df.columns:
    print("\n📅 Papers by Year (Before → After):")
    years_before = df['Publication Year'].value_counts().sort_index()
    years_after = df_cleaned['Publication Year'].value_counts().sort_index()
    
    all_years = sorted(set(years_before.index) | set(years_after.index))
    for year in all_years[-10:]:  # Show last 10 years
        before = years_before.get(year, 0)
        after = years_after.get(year, 0)
        removed = before - after
        if removed > 0:
            print(f"   {int(year)}: {before} → {after} (removed {removed})")
        else:
            print(f"   {int(year)}: {before} (no duplicates)")

# Show some duplicate examples
if total_duplicates > 0 and 'Title' in df.columns:
    print("\n📋 Example Duplicates Removed:")
    duplicates_sample = df[df['is_duplicate'] == True].head(5)
    for idx, row in duplicates_sample.iterrows():
        print(f"\n   [{row['duplicate_reason']}]")
        print(f"   Title: {row['Title'][:80]}...")
        if 'Publication Year' in row and pd.notna(row['Publication Year']):
            print(f"   Year: {int(row['Publication Year'])}")
        if 'DOI' in row and pd.notna(row['DOI']):
            print(f"   DOI: {row['DOI']}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("✅ DEDUPLICATION COMPLETE!")
print("=" * 60)
print(f"\n📄 Generated files:")
print(f"  1. {OUTPUT_FILE} - Cleaned dataset ({len(df_cleaned)} papers)")
if total_duplicates > 0:
    print(f"  2. {DUPLICATES_FILE} - Removed duplicates ({total_duplicates} papers)")

print("\n💡 Next Steps:")
print("  1. Review the cleaned dataset")
if total_duplicates > 0:
    print("  2. Check duplicates_found.csv to verify removals")
print("  3. Use cleaned dataset for your analysis")
print("  4. Import cleaned CSV back to Zotero if needed")

print("\n" + "=" * 60)
