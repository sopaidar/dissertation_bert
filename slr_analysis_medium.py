"""
BERTopic Analysis for Blockchain + Government SLR
Optimized for ~1800 papers with temporal analysis

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
# CONFIGURATION - OPTIMIZED FOR 1800 PAPERS
# ============================================

# File paths
INPUT_FILE = "bc_g.csv"  # Your Zotero CSV export
OUTPUT_FOLDER = "bertopic_blockchain_gov/"

# Column names from Zotero export
ABSTRACT_COLUMN = "Abstract Note"
TITLE_COLUMN = "Title"
YEAR_COLUMN = "Publication Year"
AUTHOR_COLUMN = "Author"
DOI_COLUMN = "DOI"
PUBLICATION_COLUMN = "Publication Title"

# BERTopic parameters - OPTIMIZED FOR 1800 PAPERS
NUM_TOPICS = "auto"  # Let BERTopic decide optimal number
MIN_TOPIC_SIZE = 15  # Good balance for 1800 papers (expect 15-25 topics)
LANGUAGE = "english"

# ============================================
# STEP 1: LOAD ZOTERO CSV
# ============================================

print("=" * 70)
print("BERTOPIC ANALYSIS - BLOCKCHAIN & GOVERNMENT")
print("Optimized for ~1800 papers")
print("=" * 70)
print(f"\n📁 Loading data from: {INPUT_FILE}")

try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8')
    print(f"✅ Successfully loaded {len(df)} records from Zotero")
    
    print(f"\n📋 Available columns (first 15):")
    for i, col in enumerate(df.columns[:15], 1):
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

# Check abstract column
if ABSTRACT_COLUMN not in df.columns:
    possible_abstract_cols = ["Abstract", "Abstract Note", "abstractNote", "AB"]
    for col in possible_abstract_cols:
        if col in df.columns:
            ABSTRACT_COLUMN = col
            print(f"✅ Found abstract column: '{col}'")
            break
    else:
        print(f"\n❌ Could not find abstract column!")
        print(f"Available columns: {list(df.columns)}")
        exit()

# Check year column
if YEAR_COLUMN not in df.columns:
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
print(f"   Min topic size: {MIN_TOPIC_SIZE} papers")

# ============================================
# STEP 3: PREPROCESS DATA
# ============================================

print("\n🔧 Preprocessing data...")

# Filter papers with abstracts
original_count = len(df)
df = df[df[ABSTRACT_COLUMN].notna()].copy()
df = df[df[ABSTRACT_COLUMN].astype(str).str.len() > 50].copy()

print(f"✅ Removed {original_count - len(df)} records without valid abstracts")
print(f"✅ Working with {len(df)} papers")

# Extract abstracts
abstracts = df[ABSTRACT_COLUMN].astype(str).tolist()

# Extract years
if YEAR_COLUMN:
    try:
        df[YEAR_COLUMN] = df[YEAR_COLUMN].astype(str).str.extract(r'(\d{4})', expand=False)
        df[YEAR_COLUMN] = pd.to_numeric(df[YEAR_COLUMN], errors='coerce')
        df = df[df[YEAR_COLUMN].notna()].copy()
        years = df[YEAR_COLUMN].astype(int).tolist()
        
        print(f"\n📅 Year range: {min(years)} - {max(years)}")
        print(f"📅 Papers per year:")
        year_counts = df[YEAR_COLUMN].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"   {int(year)}: {count} papers")
    except:
        print(f"\n⚠️  Could not parse year column")
        YEAR_COLUMN = None
        years = None
else:
    years = None

# Update abstracts after filtering
abstracts = df[ABSTRACT_COLUMN].astype(str).tolist()

print(f"\n📄 Sample abstract (first 200 chars):")
print(f"   {abstracts[0][:200]}...")

# ============================================
# STEP 4: INITIALIZE BERTOPIC
# ============================================

print("\n🤖 Initializing BERTopic model...")
print(f"   Expected topics: 15-25 (with min_topic_size={MIN_TOPIC_SIZE})")

# Domain-specific stopwords for blockchain + government
custom_stopwords = [
    # Generic stopwords
    'the', 'and', 'of', 'to', 'in', 'is', 'for', 'on', 'with', 'as', 'at', 'by',
    'from', 'this', 'that', 'these', 'those', 'be', 'are', 'was', 'were', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'can', 'may', 'might', 'must', 'an', 'a', 'or', 'but', 'if', 'then', 'so',
    'than', 'such', 'no', 'not', 'only', 'own', 'same', 'too', 'very',
    'we', 'us', 'our', 'it', 'its', 'which',
    # Academic stopwords
    'paper', 'study', 'research', 'article', 'work', 'propose', 'proposed',
    'present', 'presented', 'approach', 'method', 'result', 'results',
    'show', 'shows', 'demonstrate', 'demonstrates', 'analysis', 'analyzed',
    'based', 'using', 'used', 'use', 'also', 'however', 'therefore', 'thus',
    'new', 'provide', 'provides', 'discuss', 'discusses'
]

# Optimized vectorizer for 1800 papers
vectorizer_model = CountVectorizer(
    max_features=2000,  # Increased for more papers
    stop_words=custom_stopwords,
    ngram_range=(1, 3),  # Capture phrases like "e-voting", "smart contract"
    min_df=3,  # Word must appear in at least 3 documents
    max_df=0.85  # Ignore words in >85% of documents
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
# STEP 5: FIT MODEL (15-45 minutes for 1800 papers)
# ============================================

print("\n⏳ Training BERTopic model...")
print("   This will take 15-45 minutes for ~1800 papers... ☕☕")
print("   Progress will be shown below:")

topics, probs = topic_model.fit_transform(abstracts)

print("\n✅ Model training complete!")

# ============================================
# STEP 6: ANALYZE RESULTS
# ============================================

topic_info = topic_model.get_topic_info()
num_topics_found = len(topic_info) - 1
outlier_count = sum([1 for t in topics if t == -1])

print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"\n📊 Discovered {num_topics_found} topics")
print(f"📊 Outliers: {outlier_count} ({outlier_count/len(df)*100:.1f}%)")
print(f"📊 Average papers per topic: {len(df)/num_topics_found:.0f}")

# Display all topics
print("\n" + "=" * 70)
print("TOPIC DESCRIPTIONS")
print("=" * 70)

for idx, row in topic_info.iterrows():
    if row['Topic'] == -1:
        continue
    
    topic_id = row['Topic']
    count = row['Count']
    percentage = (count / len(df)) * 100
    top_words = topic_model.get_topic(topic_id)
    
    if top_words:
        keywords = ", ".join([word for word, score in top_words[:8]])
        print(f"\n📌 Topic {topic_id} ({count} papers, {percentage:.1f}%):")
        print(f"   {keywords}")

# ============================================
# STEP 7: ADD TOPICS TO DATAFRAME
# ============================================

print("\n📝 Adding topic assignments...")

df['Topic'] = topics
df['Topic_Probability'] = [max(prob) if len(prob) > 0 else 0 for prob in probs]

# Create meaningful topic names
topic_names = {}
for topic_id in topic_info['Topic']:
    if topic_id == -1:
        topic_names[-1] = "Outliers"
    else:
        top_words = topic_model.get_topic(topic_id)
        if top_words:
            # Use top 3 keywords for name
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

# Save topic summary with percentages
topic_summary_df = topic_info.copy()
topic_summary_df['Percentage'] = (topic_summary_df['Count'] / len(df) * 100).round(2)
topic_summary_df.to_csv(f"{OUTPUT_FOLDER}topic_summary.csv", index=False)
print(f"✅ Saved: {OUTPUT_FOLDER}topic_summary.csv")

# Save detailed keywords
with open(f"{OUTPUT_FOLDER}topic_keywords.txt", 'w', encoding='utf-8') as f:
    f.write("BLOCKCHAIN & GOVERNMENT - TOPIC MODELING RESULTS\n")
    f.write("=" * 80 + "\n\n")
    
    for topic_id in sorted([t for t in topic_info['Topic'] if t != -1]):
        topic_count = len(df[df['Topic'] == topic_id])
        percentage = (topic_count / len(df)) * 100
        f.write(f"Topic {topic_id} ({topic_count} papers, {percentage:.1f}%)\n")
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
try:
    fig1 = topic_model.visualize_barchart(top_n_topics=min(15, num_topics_found))
    fig1.write_html(f"{OUTPUT_FOLDER}topic_barchart.html")
    print(f"✅ Saved: {OUTPUT_FOLDER}topic_barchart.html")
except Exception as e:
    print(f"⚠️  Could not create bar chart: {e}")

# 2. Intertopic distance map
try:
    fig2 = topic_model.visualize_topics()
    fig2.write_html(f"{OUTPUT_FOLDER}topic_map.html")
    print(f"✅ Saved: {OUTPUT_FOLDER}topic_map.html")
except Exception as e:
    print(f"⚠️  Could not create topic map: {e}")

# 3. Topic hierarchy
try:
    fig3 = topic_model.visualize_hierarchy()
    fig3.write_html(f"{OUTPUT_FOLDER}topic_hierarchy.html")
    print(f"✅ Saved: {OUTPUT_FOLDER}topic_hierarchy.html")
except Exception as e:
    print(f"⚠️  Could not create hierarchy: {e}")

# 4. Heatmap
try:
    fig4 = topic_model.visualize_heatmap()
    fig4.write_html(f"{OUTPUT_FOLDER}topic_heatmap.html")
    print(f"✅ Saved: {OUTPUT_FOLDER}topic_heatmap.html")
except Exception as e:
    print(f"⚠️  Could not create heatmap: {e}")

# 5. Static distribution chart
plt.figure(figsize=(14, 8))
topic_counts = df[df['Topic'] != -1]['Topic'].value_counts().sort_index()
bars = plt.bar(topic_counts.index, topic_counts.values, color='steelblue', alpha=0.8)
plt.xlabel('Topic ID', fontsize=13, fontweight='bold')
plt.ylabel('Number of Papers', fontsize=13, fontweight='bold')
plt.title(f'Blockchain & Government Topics Distribution ({len(df)} Papers)', 
          fontsize=15, fontweight='bold')
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}topic_distribution.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_FOLDER}topic_distribution.png")
plt.close()

# 6. TOPICS OVER TIME (CRITICAL FOR SLR)
if YEAR_COLUMN and years:
    print("\n📈 Creating temporal analysis (Topics Over Time)...")
    try:
        timestamps = df[YEAR_COLUMN].astype(int).tolist()
        
        topics_over_time = topic_model.topics_over_time(
            abstracts, 
            timestamps,
            nr_bins=len(set(timestamps))
        )
        
        # Visualize top topics over time
        fig5 = topic_model.visualize_topics_over_time(
            topics_over_time,
            top_n_topics=min(12, num_topics_found)
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
# STEP 10: STATISTICS FOR PAPER
# ============================================

print("\n" + "=" * 70)
print("STATISTICS FOR YOUR SLR PAPER")
print("=" * 70)

stats_file = f"{OUTPUT_FOLDER}statistics_for_paper.txt"
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("BLOCKCHAIN & GOVERNMENT - SYSTEMATIC LITERATURE REVIEW STATISTICS\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Total Papers Analyzed: {len(df)}\n")
    f.write(f"Topics Discovered: {num_topics_found}\n")
    f.write(f"Outliers: {outlier_count} ({outlier_count/len(df)*100:.1f}%)\n")
    f.write(f"Average Papers per Topic: {len(df)/num_topics_found:.1f}\n\n")
    
    if YEAR_COLUMN and years:
        f.write("TEMPORAL DISTRIBUTION:\n")
        f.write("-" * 80 + "\n")
        year_dist = df[YEAR_COLUMN].value_counts().sort_index()
        for year, count in year_dist.items():
            percentage = (count / len(df)) * 100
            f.write(f"{int(year)}: {count} papers ({percentage:.1f}%)\n")
        f.write("\n")
    
    f.write("TOPIC DISTRIBUTION:\n")
    f.write("-" * 80 + "\n")
    for topic_id in sorted([t for t in topic_info['Topic'] if t != -1]):
        topic_count = len(df[df['Topic'] == topic_id])
        percentage = (topic_count / len(df)) * 100
        
        # Get top 3 keywords for description
        top_words = topic_model.get_topic(topic_id)
        if top_words:
            keywords = ", ".join([word for word, score in top_words[:3]])
            f.write(f"Topic {topic_id}: {topic_count} papers ({percentage:.1f}%) - {keywords}\n")

print(f"✅ Saved: {stats_file}")

# Sample papers per topic
print("\n📋 Generating sample papers per topic...")
sample_file = f"{OUTPUT_FOLDER}sample_papers_per_topic.txt"
with open(sample_file, 'w', encoding='utf-8') as f:
    f.write("REPRESENTATIVE PAPERS PER TOPIC\n")
    f.write("=" * 80 + "\n\n")
    
    for topic_id in sorted([t for t in topic_info['Topic'] if t != -1]):
        topic_papers = df[df['Topic'] == topic_id]
        
        # Get top keywords
        top_words = topic_model.get_topic(topic_id)
        keywords = ", ".join([word for word, score in top_words[:5]]) if top_words else ""
        
        f.write(f"\nTopic {topic_id} - {keywords}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total papers: {len(topic_papers)}\n\n")
        
        # Show top 3 papers with highest probability
        top_papers = topic_papers.nlargest(3, 'Topic_Probability')
        
        for idx, row in top_papers.iterrows():
            if TITLE_COLUMN in df.columns and pd.notna(row[TITLE_COLUMN]):
                f.write(f"• {row[TITLE_COLUMN]}\n")
            if YEAR_COLUMN and pd.notna(row[YEAR_COLUMN]):
                f.write(f"  Year: {int(row[YEAR_COLUMN])}\n")
            f.write(f"  Confidence: {row['Topic_Probability']:.3f}\n")
            f.write(f"  Abstract: {row[ABSTRACT_COLUMN][:150]}...\n\n")

print(f"✅ Saved: {sample_file}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\n📁 All results saved in: {OUTPUT_FOLDER}")
print(f"\n📊 Key Findings:")
print(f"   • {num_topics_found} distinct research topics identified")
print(f"   • {len(df)} papers analyzed")
print(f"   • {outlier_count} outliers ({outlier_count/len(df)*100:.1f}%)")
if YEAR_COLUMN and years:
    print(f"   • Year range: {min(years)}-{max(years)}")

print("\n📄 Generated files:")
print("  1. papers_with_topics.csv - Papers with topic assignments")
print("  2. topic_summary.csv - Topic overview with percentages")
print("  3. topic_keywords.txt - Detailed keywords (top 20 per topic)")
print("  4. topic_barchart.html - Interactive bar chart")
print("  5. topic_map.html - Topic distance visualization")
print("  6. topic_hierarchy.html - Topic relationships")
print("  7. topic_heatmap.html - Topic similarity matrix")
print("  8. topic_distribution.png - Publication-ready chart")
if YEAR_COLUMN and years:
    print("  9. topics_over_time.html - TEMPORAL ANALYSIS ⭐")
    print("  10. topics_over_time.csv - Temporal data for analysis")
print(f"  {11 if YEAR_COLUMN and years else 9}. statistics_for_paper.txt - Stats for your SLR")
print(f"  {12 if YEAR_COLUMN and years else 10}. sample_papers_per_topic.txt - Representative papers")

print("\n" + "=" * 70)
print("NEXT STEPS FOR YOUR SLR:")
print("=" * 70)
print("1. 📈 Open topics_over_time.html → Analyze topic evolution")
print("2. 📋 Review topic_keywords.txt → Manually label/rename topics")
print("3. 📊 Check sample_papers_per_topic.txt → Validate topic assignments")
print("4. 📝 Use statistics_for_paper.txt → Report in methodology section")
print("5. 🎨 Use PNG/HTML visualizations → Include in your paper")
print("\n💡 Tip: Topics with <20 papers might need manual review")
print("🎉 Happy analyzing!")
print("=" * 70)
