"""
BERTopic Analysis for Systematic Literature Review
Topic Modeling on 3500 Big Data & Blockchain Abstracts

Requirements:
pip install bertopic pandas matplotlib plotly scikit-learn sentence-transformers umap-learn hdbscan
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
INPUT_FILE = "abstracts.txt"  # Change to your file name
OUTPUT_FOLDER = "bertopic_results/"  # Where to save results

# Since you have a TXT file with one abstract per line:
# No column names needed!
ABSTRACT_COLUMN = "Abstract"  # Internal use only
TITLE_COLUMN = None  # Not available
YEAR_COLUMN = None  # Not available

# BERTopic parameters
NUM_TOPICS = 15  # Number of topics to extract (will auto-adjust if needed)
MIN_TOPIC_SIZE = 30  # Minimum papers per topic
LANGUAGE = "english"  # Language of your abstracts

# ============================================
# STEP 1: LOAD DATA
# ============================================

print("=" * 60)
print("BERTOPIC ANALYSIS FOR SYSTEMATIC LITERATURE REVIEW")
print("=" * 60)
print(f"\n📁 Loading data from: {INPUT_FILE}")

# Load TXT file - one abstract per line
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        abstracts = f.readlines()
    
    # Clean up abstracts (remove newlines, strip whitespace)
    abstracts = [abs.strip() for abs in abstracts if abs.strip()]
    
    # Create dataframe for tracking
    df = pd.DataFrame({
        'Paper_ID': range(1, len(abstracts) + 1),
        'Abstract': abstracts
    })
    
    print(f"✅ Successfully loaded {len(abstracts)} abstracts")
    
except FileNotFoundError:
    print(f"❌ Error: Could not find file '{INPUT_FILE}'")
    print("\nMake sure:")
    print("1. The file is in the same folder as this script")
    print("2. The filename is correct (including .txt extension)")
    print("3. Or provide full path: 'C:/Users/YourName/Documents/abstracts.txt'")
    exit()
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# ============================================
# STEP 2: PREPROCESS DATA
# ============================================

print("\n🔧 Preprocessing data...")

# Extract abstracts (already loaded)
abstracts = df['Abstract'].tolist()

# Remove very short abstracts (less than 50 characters)
original_count = len(abstracts)
valid_indices = [i for i, abs in enumerate(abstracts) if len(abs.strip()) > 50]
abstracts = [abstracts[i] for i in valid_indices]
df = df.iloc[valid_indices].reset_index(drop=True)

removed = original_count - len(abstracts)
if removed > 0:
    print(f"⚠️  Removed {removed} abstracts that were too short (< 50 chars)")
print(f"✅ Final dataset: {len(abstracts)} valid abstracts")

# Show sample abstract
print(f"\n📄 Sample abstract (first 200 chars):")
print(f"   {abstracts[0][:200]}...")


# ============================================
# STEP 3: INITIALIZE BERTOPIC
# ============================================

print("\n🤖 Initializing BERTopic model...")
print(f"   - Target topics: {NUM_TOPICS}")
print(f"   - Minimum topic size: {MIN_TOPIC_SIZE}")
print(f"   - Language: {LANGUAGE}")

# Custom stopwords for your domain
custom_stopwords = [
    # Generic stopwords
    'the', 'and', 'of', 'to', 'in', 'is', 'for', 'on', 'with', 'as', 'at', 'by', 
    'from', 'this', 'that', 'these', 'those', 'be', 'are', 'was', 'were', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'can', 'may', 'might', 'must', 'an', 'a', 'or', 'but', 'if', 'then', 'so',
    'than', 'such', 'no', 'not', 'only', 'own', 'same', 'than', 'too', 'very',
    'we', 'us', 'our', 'it', 'its',
    # Academic paper stopwords
    'paper', 'study', 'research', 'article', 'work', 'propose', 'proposed',
    'present', 'presented', 'approach', 'method', 'result', 'results',
    'show', 'shows', 'demonstrate', 'demonstrates', 'analysis', 'analyzed',
    'based', 'using', 'used', 'use', 'also', 'however', 'therefore', 'thus',
    'furthermore', 'moreover', 'additionally', 'in this', 'of the', 'in the',
    'to the', 'for the', 'on the', 'with the', 'from the'
]

# Vectorizer for better keyword extraction
vectorizer_model = CountVectorizer(
    max_features=1000,
    stop_words=custom_stopwords,
    ngram_range=(1, 3),  # Allows 1-3 word phrases
    min_df=5  # Word must appear in at least 5 documents
)

# Initialize BERTopic
topic_model = BERTopic(
    language=LANGUAGE,
    min_topic_size=MIN_TOPIC_SIZE,
    nr_topics=NUM_TOPICS,
    vectorizer_model=vectorizer_model,
    calculate_probabilities=True,
    verbose=True
)

# ============================================
# STEP 4: FIT MODEL (This may take 10-30 minutes)
# ============================================

print("\n⏳ Training BERTopic model...")
print("   This may take 10-30 minutes depending on your hardware...")
print("   Be patient! ☕")

topics, probs = topic_model.fit_transform(abstracts)

print("\n✅ Model training complete!")

# ============================================
# STEP 5: ANALYZE RESULTS
# ============================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# Get topic info
topic_info = topic_model.get_topic_info()
num_topics_found = len(topic_info) - 1  # -1 excludes outlier topic (-1)

# Count outliers safely
outlier_count = sum([1 for t in topics if t == -1])

print(f"\n📊 Discovered {num_topics_found} topics")
print(f"📊 Outliers (ungrouped papers): {outlier_count}")

# Display topics
print("\n" + "=" * 60)
print("TOPIC DESCRIPTIONS")
print("=" * 60)

for idx, row in topic_info.iterrows():
    if row['Topic'] == -1:
        continue  # Skip outlier topic
    
    topic_id = row['Topic']
    count = row['Count']
    
    # Get top words for this topic
    top_words = topic_model.get_topic(topic_id)
    if top_words:
        keywords = ", ".join([word for word, score in top_words[:10]])
        print(f"\n📌 Topic {topic_id} ({count} papers):")
        print(f"   Keywords: {keywords}")

# ============================================
# STEP 6: ADD TOPICS TO DATAFRAME
# ============================================

print("\n📝 Adding topic assignments to dataframe...")

df['Topic'] = topics
df['Topic_Probability'] = [max(prob) if len(prob) > 0 else 0 for prob in probs]

# Add topic names (you can customize these later)
topic_names = {}
for topic_id in topic_info['Topic']:
    if topic_id == -1:
        topic_names[-1] = "Outliers"
    else:
        # Use top 3 keywords as topic name
        top_words = topic_model.get_topic(topic_id)
        if top_words:
            name = "_".join([word for word, score in top_words[:3]])
            topic_names[topic_id] = f"Topic_{topic_id}_{name}"
        else:
            topic_names[topic_id] = f"Topic_{topic_id}"

df['Topic_Name'] = df['Topic'].map(topic_names)

# ============================================
# STEP 7: SAVE RESULTS
# ============================================

print("\n💾 Saving results...")

import os
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Save dataframe with topics
output_csv = f"{OUTPUT_FOLDER}papers_with_topics.csv"
df.to_csv(output_csv, index=False)
print(f"✅ Saved: {output_csv}")

# Save topic information
topic_summary = f"{OUTPUT_FOLDER}topic_summary.csv"
topic_info.to_csv(topic_summary, index=False)
print(f"✅ Saved: {topic_summary}")

# Save detailed topic keywords
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
# STEP 8: CREATE VISUALIZATIONS
# ============================================

print("\n📊 Creating visualizations...")

# 1. Topic distribution bar chart
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

# 5. Topic distribution (static image)
plt.figure(figsize=(12, 8))
topic_counts = df[df['Topic'] != -1]['Topic'].value_counts().sort_index()
plt.bar(topic_counts.index, topic_counts.values)
plt.xlabel('Topic ID', fontsize=12)
plt.ylabel('Number of Papers', fontsize=12)
plt.title('Topic Distribution (3500 Papers)', fontsize=14, fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{OUTPUT_FOLDER}topic_distribution.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_FOLDER}topic_distribution.png")
plt.close()

# 6. Topics over time (if year column exists)
# Not available for TXT file format
print("ℹ️  Topics over time not available (no year data in TXT file)")

# ============================================
# STEP 9: GENERATE STATISTICS FOR PAPER
# ============================================

print("\n" + "=" * 60)
print("STATISTICS FOR YOUR SLR PAPER")
print("=" * 60)

stats_file = f"{OUTPUT_FOLDER}statistics_for_paper.txt"
with open(stats_file, 'w', encoding='utf-8') as f:
    f.write("STATISTICS FOR SYSTEMATIC LITERATURE REVIEW\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Total Papers Analyzed: {len(df)}\n")
    f.write(f"Topics Discovered: {num_topics_found}\n")
    f.write(f"Outliers: {outlier_count} ({outlier_count/len(df)*100:.1f}%)\n\n")
    
    f.write("TOPIC DISTRIBUTION:\n")
    f.write("-" * 80 + "\n")
    for topic_id in sorted([t for t in topic_info['Topic'] if t != -1]):
        topic_count = len(df[df['Topic'] == topic_id])
        percentage = (topic_count / len(df)) * 100
        f.write(f"Topic {topic_id}: {topic_count} papers ({percentage:.1f}%)\n")

print(f"✅ Saved: {stats_file}")

# ============================================
# STEP 10: SAMPLE PAPERS PER TOPIC
# ============================================

print("\n📋 Generating sample papers per topic...")

sample_file = f"{OUTPUT_FOLDER}sample_papers_per_topic.txt"
with open(sample_file, 'w', encoding='utf-8') as f:
    f.write("REPRESENTATIVE PAPERS PER TOPIC\n")
    f.write("=" * 80 + "\n\n")
    
    for topic_id in sorted([t for t in topic_info['Topic'] if t != -1]):
        topic_papers = df[df['Topic'] == topic_id]
        
        f.write(f"Topic {topic_id}\n")
        f.write("-" * 80 + "\n")
        
        # Show top 5 papers with highest probability
        top_papers = topic_papers.nlargest(5, 'Topic_Probability')
        
        for idx, row in top_papers.iterrows():
            f.write(f"\nPaper ID: {row['Paper_ID']}\n")
            f.write(f"Probability: {row['Topic_Probability']:.3f}\n")
            f.write(f"Abstract: {row['Abstract'][:300]}...\n")
        
        f.write("\n\n")

print(f"✅ Saved: {sample_file}")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "=" * 60)
print("✅ ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\n📁 All results saved in: {OUTPUT_FOLDER}")
print("\nGenerated files:")
print("  1. papers_with_topics.csv - Your data with topic assignments")
print("  2. topic_summary.csv - Overview of all topics")
print("  3. topic_keywords.txt - Detailed keywords per topic")
print("  4. topic_barchart.html - Interactive bar chart")
print("  5. topic_map.html - Interactive topic map")
print("  6. topic_hierarchy.html - Topic relationships")
print("  7. topic_heatmap.html - Topic similarity heatmap")
print("  8. topic_distribution.png - Static chart for paper")
print("  9. statistics_for_paper.txt - Stats to report in SLR")
print("  10. sample_papers_per_topic.txt - Representative papers")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Open the HTML files in your browser to explore topics")
print("2. Review topic_keywords.txt to understand each topic")
print("3. Rename topics in papers_with_topics.csv based on your interpretation")
print("4. Use statistics_for_paper.txt in your SLR methodology section")
print("5. Include visualizations in your paper")
print("\nHappy analyzing! 🎉")
print("=" * 60)
