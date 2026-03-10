import streamlit as st
import pandas as pd
from fuzzywuzzy import process, fuzz
import requests
from textblob import TextBlob
from symspellpy import SymSpell, Verbosity

# Set page config
st.set_page_config(page_title="Word Corrector - 4 Methods", layout="wide")

st.title("🔤 Word Corrector - 4 Methods")
st.write("Enter a misclassified word and get correction suggestions from four different methods")

# Load dictionary
@st.cache_data
def load_dictionary():
    url = "https://inventwithpython.com/dictionary.txt"
    dictionary = pd.read_csv(url, header=None, names=["word"])
    dictionary['word'] = dictionary['word'].astype(str).str.lower().str.strip()
    return dictionary

# Initialize SymSpell
@st.cache_resource
def init_symspell():
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dictionary = load_dictionary()
    for word in dictionary['word'].tolist():
        if isinstance(word, str) and len(word) > 0:
            sym_spell.create_dictionary_entry(word, 1)
    return sym_spell

dictionary = load_dictionary()
dictionary_words = [w for w in dictionary["word"].tolist() if isinstance(w, str)]
sym_spell = init_symspell()

# Fuzzy matching function
def get_fuzzy_matches(word, dictionary_words, n=5):
    matches = process.extract(word.lower(), dictionary_words, scorer=fuzz.WRatio, limit=n)
    return [m[0] for m in matches]

# Fixed length-based fuzzy matching function (+- 2 chars)
def get_length_matched_fuzzy_suggestions(word, dictionary_words, n=5):
    word_len = len(str(word))
    # Filter words within 2 characters length difference
    filtered_words = [w for w in dictionary_words if abs(len(str(w)) - word_len) <= 2]
    
    # If no words match the length criteria, use all dictionary words
    if not filtered_words:
        filtered_words = dictionary_words
    
    # Get fuzzy matches from the filtered list
    matches = process.extract(word.lower(), filtered_words, scorer=fuzz.WRatio, limit=n)
    return [m[0] for m in matches]

# SymSpell correction
def get_symspell_correction(word):
    try:
        suggestions = sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            top_suggestions = suggestions[:5]
            best_guess = top_suggestions[0].term if top_suggestions else None
            other_suggestions = [s.term for s in top_suggestions[1:]] if len(top_suggestions) > 1 else []
            return best_guess, other_suggestions
        return None, []
    except Exception as e:
        return None, []

# TextBlob correction
def get_textblob_correction(word):
    try:
        blob = TextBlob(word)
        corrected = str(blob.correct())
        return corrected
    except Exception as e:
        return f"Error: {str(e)}"

# Main layout - Graph toggle at the top (before any results)
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    user_word = st.text_input("Enter misclassified word:", placeholder="e.g., helo, wrld, pythn, energatik")
with col2:
    st.write("")  # Spacing
    st.write("")  # Spacing
    show_graphs = st.checkbox("📊 Show Graphs", value=True, help="Show similarity graphs with results")
with col3:
    st.write("")  # Spacing
    st.write("")  # Spacing
    get_corrections = st.button("🔍 Get Corrections", type="primary", use_container_width=True)

if get_corrections and user_word:
    # Get results from all methods
    std_suggestions = get_fuzzy_matches(user_word, dictionary_words, n=5)
    len_suggestions = get_length_matched_fuzzy_suggestions(user_word, dictionary_words, n=5)
    symspell_best, symspell_others = get_symspell_correction(user_word)
    textblob_result = get_textblob_correction(user_word)
    
    # Display results in four columns
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.subheader("📊 Standard Fuzzy")
        for i, suggestion in enumerate(std_suggestions, 1):
            st.write(f"{i}. {suggestion}")
    
    with col2:
        st.subheader("📏 Length-Matched")
        st.caption(f"(±2 chars from '{user_word}' length {len(user_word)})")
        for i, suggestion in enumerate(len_suggestions, 1):
            st.write(f"{i}. {suggestion}")
    
    with col3:
        st.subheader("🔍 SymSpell")
        if symspell_best:
            st.write(f"**Best guess:** {symspell_best}")
            if symspell_others:
                st.write("**Other suggestions:**")
                for i, suggestion in enumerate(symspell_others, 1):
                    st.write(f"{i+1}. {suggestion}")
        else:
            st.write("No suggestions found")
    
    with col4:
        st.subheader("📝 TextBlob")
        st.write(f"**Correction:** {textblob_result}")
    
    # Show graphs immediately after results if toggle is on
    if show_graphs:
        st.markdown("---")
        st.subheader("📈 Similarity Scores Comparison")
        
        # Create two columns for the graphs
        graph_col1, graph_col2 = st.columns(2)
        
        with graph_col1:
            st.write("**Standard Fuzzy Matching**")
            if std_suggestions:
                std_scores = process.extract(
                    user_word.lower(), 
                    std_suggestions, 
                    scorer=fuzz.WRatio
                )
                std_df = pd.DataFrame(std_scores, columns=["Word", "Score"])
                # Sort by score for better visualization
                std_df = std_df.sort_values('Score', ascending=True)
                st.bar_chart(std_df.set_index("Word"), height=400)
                
                # Show scores as numbers
                with st.expander("View scores"):
                    for word, score in std_scores:
                        st.write(f"- {word}: {score}")
            else:
                st.write("No suggestions to display")
        
        with graph_col2:
            st.write("**Length-Matched Fuzzy**")
            if len_suggestions:
                len_scores = process.extract(
                    user_word.lower(), 
                    len_suggestions, 
                    scorer=fuzz.WRatio
                )
                len_df = pd.DataFrame(len_scores, columns=["Word", "Score"])
                # Sort by score for better visualization
                len_df = len_df.sort_values('Score', ascending=True)
                st.bar_chart(len_df.set_index("Word"), height=400)
                
                # Show scores as numbers
                with st.expander("View scores"):
                    for word, score in len_scores:
                        st.write(f"- {word}: {score}")
            else:
                st.write("No suggestions to display")

# Dictionary info in sidebar
st.sidebar.info(f"📚 Dictionary loaded with {len(dictionary):,} words")

# Show examples
with st.sidebar.expander("Try these examples"):
    st.write("""
    - **helo** (length 4) → hello (length 5)
    - **wrld** (length 4) → world (length 5)
    - **pythn** (length 5) → python (length 6)
    - **energatik** (length 9) → energetic (length 8)
    - **comparsion** (length 10) → comparison (length 10)
    - **recieve** (length 7) → receive (length 7)
    """)

# Add length filter info
with st.sidebar.expander("🔧 How Length-Matched works"):
    st.write("""
    The Length-Matched Fuzzy filter only considers words that are within:
    - **±2 characters** of your input word's length
    
    For example, if you type 'helo' (length 4), it will only consider dictionary words of length:
    - 2, 3, 4, 5, or 6 characters
    """)