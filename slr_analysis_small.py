"""
BERTopic Analysis for Zotero CSV Export
Includes Temporal Analysis (Topics Over Time)

Requirements:
pip install bertopic pandas matplotlib plotly scikit-learn sentence-transformers umap-learn hdbscan openpyxl
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================

# File paths
INPUT_FILE = "big_data_value_chain.csv"  # Your Zotero CSV export
OUTPUT_FOLDER = "bertopic_big_data_value_chain/"

# Column names from Zotero export (check your CSV and adjust if needed)
ABSTRACT_COLUMN = "Abstract Note"  # Zotero usually uses "Abstract Note"
TITLE_COLUMN = "Title"
YEAR_COLUMN = "Publication Year"
AUTHOR_COLUMN = "Author"
DOI_COLUMN = "DOI"
PUBLICATION_COLUMN = "Publication Title"

# Alternative column names (Zotero sometimes varies)
# If above don't work, try these:
# ABSTRACT_COLUMN = "Abstract"
# YEAR_COLUMN = "Year" or "Date"

# BERTopic parameters
NUM_TOPICS = "auto"
MIN_TOPIC_SIZE = 30
LANGUAGE = "english"

# ============================================
# STEP 1: LOAD ZOTERO CSV
# ============================================

print("=" * 60)
print("BERTOPIC ANALYSIS - ZOTERO CSV EXPORT")
print("=" * 60)
print(f"\n📁 Loading data from: {INPUT_FILE}")

try:
    # Load CSV
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    print(f"✅ Successfully loaded {len(df)} records from Zotero")
    
    # Show available columns
    print(f"\n📋 Available columns in your CSV:")
    for i, col in enumerate(df.columns[:15], 1):  # Show first 15 columns
        print(f"   {i}. {col}")
    if len(df.columns) > 15:
        print(f"   ... and {len(df.columns) - 15} more columns")
    
except FileNotFoundError:
    print(f"\n❌ Error: Could not find '{INPUT_FILE}'")
    print("\nSteps to fix:")
    print("1. Export from Zotero: Right-click items → Export → CSV")
    print("2. Save file in same folder as this script")
    print("3. Update INPUT_FILE variable with correct filename")
    exit()
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# ============================================
# STEP 2: CHECK AND VALIDATE COLUMNS
# ============================================

print("\n🔍 Checking required columns...")

# Check if abstract column exists
if ABSTRACT_COLUMN not in df.columns:
    print(f"\n⚠️  Column '{ABSTRACT_COLUMN}' not found!")
    print("\nTrying alternative names...")
    
    # Try alternatives
    possible_abstract_cols = ["Abstract", "Abstract Note", "abstractNote", "AB"]
    for col in possible_abstract_cols:
        if col in df.columns:
            ABSTRACT_COLUMN = col
            print(f"✅ Found abstract column: '{col}'")
            break
    else:
        print(f"\n❌ Could not find abstract column!")
        print(f"Available columns: {list(df.columns)}")
        print("\nPlease update ABSTRACT_COLUMN in the script")
        exit()

# Check if year column exists
if YEAR_COLUMN not in df.columns:
    print(f"\n⚠️  Column '{YEAR_COLUMN}' not found!")
    
    # Try alternatives
    possible_year_cols = ["Year", "Publication Year", "Date", "publicationYear"]
    for col in possible_year_cols:
        if col in df.columns:
            YEAR_COLUMN = col
            print(f"✅ Found year column: '{col}'")
            break
    else:
        print(f"\n⚠️  No year column found - temporal analysis will be skipped")
        YEAR_COLUMN = None

print(f"\n✅ Configuration:")
print(f"   Abstract column: {ABSTRACT_COLUMN}")
print(f"   Year column: {YEAR_COLUMN}")
if TITLE_COLUMN in df.columns:
    print(f"   Title column: {TITLE_COLUMN}")

# ============================================
# STEP 3: PREPROCESS DATA
# ============================================

print("\n🔧 Preprocessing data...")

# Filter only papers with abstracts
original_count = len(df)
df = df[df[ABSTRACT_COLUMN].notna()].copy()
df = df[df[ABSTRACT_COLUMN].astype(str).str.len() > 50].copy()

print(f"✅ Removed {original_count - len(df)} records without valid abstracts")
print(f"✅ Working with {len(df)} papers")

# Extract abstracts
abstracts = df[ABSTRACT_COLUMN].astype(str).tolist()

# Extract years if available
if YEAR_COLUMN:
    try:
        # Clean year column (remove any non-numeric characters) - FIXED WITH r''
        df[YEAR_COLUMN] = df[YEAR_COLUMN].astype(str).str.extract(r'(\d{4})', expand=False)
        df[YEAR_COLUMN] = pd.to_numeric(df[YEAR_COLUMN], errors='coerce')
        
        # Remove papers without valid years
        df = df[df[YEAR_COLUMN].notna()].copy()
        years = df[YEAR_COLUMN].astype(int).tolist()
        
        print(f"\n📅 Year range: {min(years)} - {max(years)}")
        print(f"📅 Papers per year:")
        year_counts = df[YEAR_COLUMN].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"   {int(year)}: {count} papers")
    except:
        print(f"\n⚠️  Could not parse year column - temporal analysis will be skipped")
        YEAR_COLUMN = None
        years = None
else:
    years = None

# Update abstracts after year filtering
abstracts = df[ABSTRACT_COLUMN].astype(str).tolist()

# Show sample
print(f"\n📄 Sample abstract (first 200 chars):")
print(f"   {abstracts[0][:200]}...")

# ============================================
# STEP 4: INITIALIZE BERTOPIC
# ============================================

print("\n🤖 Initializing BERTopic model...")

# Comprehensive stopwords
custom_stopwords = [
    'the', 'and', 'of', 'to', 'in', 'is', 'for', 'on', 'with', 'as', 'at', 'by',
    'from', 'this', 'that', 'these', 'those', 'be', 'are', 'was', 'were', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'can', 'may', 'might', 'must', 'an', 'a', 'or', 'but', 'if', 'then', 'so',
    'than', 'such', 'no', 'not', 'only', 'own', 'same', 'too', 'very',
    'we', 'us', 'our', 'it', 'its',
    'paper', 'study', 'research', 'article', 'work', 'propose', 'proposed',
    'present', 'presented', 'approach', 'method', 'result', 'results',
    'show', 'shows', 'demonstrate', 'demonstrates', 'analysis', 'analyzed',
    'based', 'using', 'used', 'use', 'also', 'however', 'therefore', 'thus'
]

# FIXED: Removed min_df parameter that was causing the error
vectorizer_model = CountVectorizer(
    max_features=1000,
    stop_words=custom_stopwords,
    ngram_range=(1, 3)
)

topic_model = BERTopic(
    language=LANGUAGE,
    min_topic_size=MIN_TOPIC_SIZE,
    nr_topics=NUM_TOPICS,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)

# ============================================
# STEP 5: FIT MODEL
# ============================================

print("\n⏳ Training BERTopic model...")
print("   This may take 10-30 minutes... ☕")

topics, probs = topic_model.fit_transform(abstracts)

print("\n✅ Model training complete!")

# ============================================
# STEP 6: ANALYZE RESULTS
# ============================================

topic_info = topic_model.get_topic_info()
num_topics_found = len(topic_info) - 1
outlier_count = sum([1 for t in topics if t == -1])

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\n📊 Discovered {num_topics_found} topics")
print(f"📊 Outliers: {outlier_count} ({outlier_count/len(df)*100:.1f}%)")

# Display topics
print("\n" + "=" * 60)
print("TOPIC DESCRIPTIONS")
print("=" * 60)

for idx, row in topic_info.iterrows():
    if row['Topic'] == -1:
        continue
    
    topic_id = row['Topic']
    count = row['Count']
    top_words = topic_model.get_topic(topic_id)
    
    if top_words:
        keywords = ", ".join([word for word, score in top_words[:10]])
        print(f"\n📌 Topic {topic_id} ({count} papers):")
        print(f"   {keywords}")

# ============================================
# STEP 7: ADD TOPICS TO DATAFRAME
# ============================================

print("\n📝 Adding topic assignments...")

df['Topic'] = topics
df['Topic_Probability'] = [max(prob) if len(prob) > 0 else 0 for prob in probs]

# Topic names
topic_names = {}
for topic_id in topic_info['Topic']:
    if topic_id == -1:
        topic_names[-1] = "Outliers"
    else:
        top_words = topic_model.get_topic(topic_id)
        if top_words:
            name = "_".join([word for word, score in top_words[:3]])
            topic_names[topic_id] = f"Topic_{topic_id}_{name}"

df['Topic_Name'] = df['Topic'].map(topic_names)

# ============================================
# STEP 8: SAVE RESULTS
# ============================================

print("\n💾 Saving results...")

import os
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Save with all metadata
output_csv = f"{OUTPUT_FOLDER}papers_with_topics.csv"
columns_to_save = ['Topic', 'Topic_Name', 'Topic_Probability', ABSTRACT_COLUMN]

if TITLE_COLUMN in df.columns:
    columns_to_save.insert(0, TITLE_COLUMN)
if YEAR_COLUMN:
    columns_to_save.insert(1, YEAR_COLUMN)
if DOI_COLUMN in df.columns:
    columns_to_save.append(DOI_COLUMN)
if AUTHOR_COLUMN in df.columns:
    columns_to_save.append(AUTHOR_COLUMN)

df[columns_to_save].to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")

# Save topic summary
topic_summary = f"{OUTPUT_FOLDER}topic_summary.csv"
topic_info.to_csv(topic_summary, index=False)
print(f"✅ Saved: {topic_summary}")

# Save keywords
with open(f"{OUTPUT_FOLDER}topic_keywords.txt", 'w', encoding='utf-8') as f:
    f.write("TOPIC MODELING RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    for topic_id in sorted([t for t in topic_info['Topic'] if t != -1]):
        topic_count = len(df[df['Topic'] == topic_id])
        f.write(f"Topic {topic_id} ({topic_count} papers)\n")
        f.write("-" * 80 + "\n")
        
        top_words = topic_model.get_topic(topic_id)
        if top_words:
            for word, score in top_words[:20]:
                f.write(f"  {word}: {score:.4f}\n")
        f.write("\n")

print(f"✅ Saved: {OUTPUT_FOLDER}topic_keywords.txt")

# ============================================
# STEP 9: CREATE VISUALIZATIONS
# ============================================

print("\n📊 Creating visualizations...")

# 1. Topic bar chart
fig1 = topic_model.visualize_barchart(top_n_topics=num_topics_found)
fig1.write_html(f"{OUTPUT_FOLDER}topic_barchart.html")
print(f"✅ Saved: {OUTPUT_FOLDER}topic_barchart.html")

# 2. Intertopic distance map
fig2 = topic_model.visualize_topics()
fig2.write_html(f"{OUTPUT_FOLDER}topic_map.html")
print(f"✅ Saved: {OUTPUT_FOLDER}topic_map.html")

# 3. Topic hierarchy
fig3 = topic_model.visualize_hierarchy()
fig3.write_html(f"{OUTPUT_FOLDER}topic_hierarchy.html")
print(f"✅ Saved: {OUTPUT_FOLDER}topic_hierarchy.html")

# 4. Heatmap
fig4 = topic_model.visualize_heatmap()
fig4.write_html(f"{OUTPUT_FOLDER}topic_heatmap.html")
print(f"✅ Saved: {OUTPUT_FOLDER}topic_heatmap.html")

# 5. Static distribution chart
plt.figure(figsize=(12, 8))
topic_counts = df[df['Topic'] != -1]['Topic'].value_counts().sort_index()
plt.bar(topic_counts.index, topic_counts.values)
plt.xlabel('Topic ID', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.title(f'Topic Distribution ({len(df)} Papers)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}topic_distribution.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_FOLDER}topic_distribution.png")
plt.close()

# 6. TOPICS OVER TIME (if year available)
if YEAR_COLUMN and years:
    print("\n📈 Creating temporal analysis (Topics Over Time)...")
    try:
        # Prepare data for topics over time
        timestamps = df[YEAR_COLUMN].astype(int).tolist()
        
        # Create topics over time
        topics_over_time = topic_model.topics_over_time(
            abstracts, 
            timestamps,
            nr_bins=len(set(timestamps))  # One bin per year
        )
        
        # Visualize
        fig5 = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=min(10, num_topics_found)  # Show top 10 topics
        )
        fig5.write_html(f"{OUTPUT_FOLDER}topics_over_time.html")
        print(f"✅ Saved: {OUTPUT_FOLDER}topics_over_time.html")
        
        # Save temporal data
        topics_over_time.to_csv(f"{OUTPUT_FOLDER}topics_over_time.csv", index=False)
        print(f"✅ Saved: {OUTPUT_FOLDER}topics_over_time.csv")
        
    except Exception as e:
        print(f"⚠️  Could not create temporal analysis: {e}")
else:
    print("\nℹ️  Temporal analysis skipped (no year data)")

# ============================================
# STEP 10: STATISTICS
# ============================================

print("\n" + "=" * 60)
print("STATISTICS FOR YOUR SLR")
print("=" * 60)

stats_file = f"{OUTPUT_FOLDER}statistics_for_paper.txt"
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("STATISTICS FOR SYSTEMATIC LITERATURE REVIEW\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Total Papers Analyzed: {len(df)}\n")
    f.write(f"Topics Discovered: {num_topics_found}\n")
    f.write(f"Outliers: {outlier_count} ({outlier_count/len(df)*100:.1f}%)\n\n")
    
    if YEAR_COLUMN and years:
        f.write("YEAR DISTRIBUTION:\n")
        f.write("-" * 80 + "\n")
        year_dist = df[YEAR_COLUMN].value_counts().sort_index()
        for year, count in year_dist.items():
            f.write(f"{int(year)}: {count} papers\n")
        f.write("\n")
    
    f.write("TOPIC DISTRIBUTION:\n")
    f.write("-" * 80 + "\n")
    for topic_id in sorted([t for t in topic_info['Topic'] if t != -1]):
        topic_count = len(df[df['Topic'] == topic_id])
        percentage = (topic_count / len(df)) * 100
        f.write(f"Topic {topic_id}: {topic_count} papers ({percentage:.1f}%)\n")

print(f"✅ Saved: {stats_file}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\n📁 Results saved in: {OUTPUT_FOLDER}")
print("\n📄 Generated files:")
print("  1. papers_with_topics.csv - Papers with topic assignments")
print("  2. topic_summary.csv - Topic overview")
print("  3. topic_keywords.txt - Detailed keywords")
print("  4. topic_barchart.html - Interactive bar chart")
print("  5. topic_map.html - Topic distance map")
print("  6. topic_hierarchy.html - Topic relationships")
print("  7. topic_heatmap.html - Topic similarity")
print("  8. topic_distribution.png - Static chart")
if YEAR_COLUMN and years:
    print("  9. topics_over_time.html - TEMPORAL ANALYSIS ⭐")
    print("  10. topics_over_time.csv - Temporal data")
    print("  11. statistics_for_paper.txt - Stats for SLR")
else:
    print("  9. statistics_for_paper.txt - Stats for SLR")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Open topics_over_time.html to see topic trends ⭐")
print("2. Review topic_keywords.txt")
print("3. Check papers_with_topics.csv for individual assignments")
print("4. Use visualizations in your paper")
print("\n🎉 Happy analyzing!")
print("=" * 60)
