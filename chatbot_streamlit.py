import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import base64

st.set_page_config(
    page_title="Bulipe chatbot",
    page_icon="images/logo.png",
    layout="wide" 
)

def set_custom_style(background_image_path, sidebar_image_path):
    with open(background_image_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    with open(sidebar_image_path, "rb") as sidebar_img:
        sidebar_encoded = base64.b64encode(sidebar_img.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    [data-testid=stSidebar] {{
        background-image: url("data:image/png;base64,{sidebar_encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .stSidebar .sidebar-content {{
        background-color: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(5px);
        padding: 1rem;
        border-radius: 10px;
    }}

    .stSidebar [data-testid="stSidebarNav"] {{
        color: black !important;
    }}

    .sidebar-logo-container {{
        display: flex;
        justify-content: center;
        margin-bottom: 1rem;
    }}

    .sidebar-logo {{
        width: 120px;
        height: 120px;
        border-radius: 50%;
        border: 2px solid #00000033;
        object-fit: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Sidebar content using HTML and CSS styles
    with st.sidebar:
        st.markdown('<div class="sidebar-logo-container">', unsafe_allow_html=True)
        logo_encoded = base64.b64encode(open("images/r.png", "rb").read()).decode()
        st.markdown(f'<img src="data:image/png;base64,{logo_encoded}" class="sidebar-logo"/>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<h2 style="color: black;">Filters</h2>', unsafe_allow_html=True)

        st.markdown("### 💬 Bulipe Chatbot")
        st.markdown("Welcome to Bulipe Tech Services Chatbot. Get help with our digital skills programs.")


        st.markdown("---")
        st.markdown("#### 📞 Contact")
        st.markdown("📧 [info@bulipetech.com](mailto:info@bulipetech.com)")
        st.markdown("🌐 [www.bulipetech.com](https://www.bulipetech.com)")




# Load data
@st.cache_data
def load_data():
    return pd.read_excel("chatbot_data_main.xlsx")[["Input", "Response"]].dropna()

df = load_data()

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Embed all questions once
@st.cache_resource
def embed_questions(questions):
    return model.encode(questions, convert_to_tensor=True)

question_embeddings = embed_questions(df["Input"].tolist())

st.title("Bulipe Tech Services based Chatbot")
set_custom_style("images/B.jpg", "images/sidebar.jpg")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Assalamu Alaikum 🌿! I'm your Bulipe Tech  chatbot. Ask me anything about our digital skills programs."
    }]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Search function
def get_best_response(query):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, question_embeddings)[0]
    best_idx = scores.argmax()
    return df.iloc[best_idx]["Response"]

# Chat input
if user_input := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    answer = get_best_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
