import streamlit as st
from songwriter import write_songs
from streamlit_support import *

st.set_page_config(page_title="SwiftAI", layout="wide", page_icon="favicon.ico")
st.title("Taylor Swift Song Generator: SwiftAI")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.markdown("""Fan of Taylor? Want her to release some new music? Write a song on a specific topic? Well 
            look no further than **SwiftAI**! I built a neural network to write brand new Taylor Swift songs
            from YOUR input. Just put in a seed phrase, then look at the *five* generated
             bangers and pick the one you like best!""")

with st.sidebar:
    st.markdown("""# Wait Time Information""")
    st.markdown("""Writing new songs is a *creative* process. The time to generate songs will **depend on your desired song
        length**. The longer it is, the longer it will take, and vice versa. Typically, writing some 200-word
        songs will take *a few minutes*. The first time you load this site, it may take some extra
        time to pre-load data.""")

taylor_swift = get_SwiftAI(get_cache_time())
pred_len = taylor_swift.pred_length

song_length = st.slider("Desired song length (words): ", 50, 200)

original_seed = st.text_input(f'Enter your {pred_len}-word seed phrase here. For example: \"I knew you were '
                              f'trouble\" (separated by spaces). Press enter/return to generate: ').lower().strip()

if len(original_seed.split(' ')) != pred_len:
    st.text(f'Please input exactly {pred_len} words separated by spaces.')
else:
    with st.spinner("Generating songs... do not edit anything... check sidebar for wait information..."):
        songs = write_songs(taylor_swift, original_seed, song_length)
    for num in [i+1 for i in range(len(songs))]:
        st.markdown(f"""*Song #{num}*:""")
        st.text(songs[num - 1])
        st.text("\n")
