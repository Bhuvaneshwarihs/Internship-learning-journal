
import streamlit as st
import chromadb
import open_clip
import torch
from PIL import Image
import ollama

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Lost & Found Reunion",
    page_icon="🔎",
    layout="wide"
)
# ---------------------------------------------------
# SIDEBAR DASHBOARD
# ---------------------------------------------------

with st.sidebar:

    st.title("🔎 Lost & Found AI")

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "🔎 Text Search",
            "📷 Image Search",
            "📊 Dataset Analytics"
        ]
    )

    st.markdown("---")

    st.write("System Info")

    st.metric("Items Indexed","500")
    st.metric("Model","CLIP")
    st.metric("Database","ChromaDB")
    # ---------------------------------------------------
# SEARCH HISTORY
# ---------------------------------------------------

if "search_history" not in st.session_state:
    st.session_state.search_history = []

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------
st.markdown("""
<style>

/* ---------- MAIN BACKGROUND ---------- */

[data-testid="stAppViewContainer"]{
background:
radial-gradient(circle at 20% 30%, rgba(120,119,198,0.4), transparent 40%),
radial-gradient(circle at 80% 70%, rgba(255,0,200,0.25), transparent 40%),
linear-gradient(135deg,#0f172a,#1e1b4b,#4c1d95);
min-height:100vh;
color:white;
font-family: 'Segoe UI', sans-serif;
}

/* ---------- HERO TITLE ---------- */

.hero{
text-align:center;
padding-top:40px;
padding-bottom:20px;
}

.hero h1{
font-size:60px;
font-weight:800;
letter-spacing:1px;
}

.hero p{
font-size:20px;
opacity:0.8;
}

/* ---------- GLASS METRIC CARDS ---------- */


[data-testid="stMetric"]{
background: rgba(255,255,255,0.08);
padding:25px;
border-radius:16px;
backdrop-filter: blur(12px);
border:1px solid rgba(255,255,255,0.18);
box-shadow:0 10px 30px rgba(0,0,0,0.45);
}

/* ---------- SEARCH PANEL ---------- */

.search-card{
margin-top:40px;
}

.search-card{
background: rgba(255,255,255,0.08);
padding:40px;
border-radius:20px;
backdrop-filter: blur(14px);
border:1px solid rgba(255,255,255,0.2);
box-shadow:0 10px 40px rgba(0,0,0,0.4);
}

/* ---------- INPUT BOX ---------- */

input{
border-radius:10px !important;
padding:12px !important;
border:none !important;
}

/* ---------- BUTTON ---------- */

.stButton>button{
background: linear-gradient(90deg,#6366f1,#a855f7);
border:none;
color:white;
padding:12px 35px;
border-radius:12px;
font-size:16px;
font-weight:600;
box-shadow:0 10px 30px rgba(0,0,0,0.4);
transition:0.3s;
}

.stButton>button:hover{
transform:scale(1.05);
box-shadow:0 15px 40px rgba(0,0,0,0.6);
}

/* ---------- PRODUCT CARDS ---------- */

.product-card{
background: rgba(255,255,255,0.08);
border-radius:18px;
padding:15px;
backdrop-filter: blur(10px);
border:1px solid rgba(255,255,255,0.15);
transition:0.3s;
}

.product-card:hover{
transform:translateY(-10px) scale(1.03);
box-shadow:0 20px 50px rgba(0,0,0,0.6);
}

/* ---------- MATCH BADGE ---------- */

.confidence-medium{
background:#22c55e;
color:white;
padding:6px 12px;
border-radius:8px;
font-weight:bold;
display:inline-block;
margin-top:5px;
}

/* ---------- CENTER PAGE WIDTH ---------- */

.block-container{
max-width:1500px;
margin:auto;
padding-top:2rem;
}

/* ---------- FOOTER ---------- */

footer{
text-align:center;
padding:30px;
opacity:0.7;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* CENTERED WEBSITE LAYOUT */

.block-container {
    max-width: 1500px;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
    margin: auto;
}

/* TOP SPACING */

.block-container {
    padding-top: 2rem;
}

/* SMALL COLUMN SPACING */

[data-testid="column"]{
    padding:8px;
}


/* NOT FOUND ALERT ANIMATION */

/* CENTER WRAPPER */

.notfound-wrapper{
display:flex;
justify-content:center;
align-items:center;
margin-top:40px;
}

/* CARD */

.notfound-card{
width:300px;
height:170px;
position:relative;
}

/* LID */

.notfound-lid{
width:100%;
height:45px;
background: linear-gradient(90deg,#6366f1,#a855f7);
border-radius:12px 12px 0 0;
position:absolute;
top:-45px;
left:0;
transform-origin:bottom;
animation: lidDrop 0.6s ease forwards;
box-shadow:0 6px 20px rgba(0,0,0,0.4);
}

/* CARD BODY */

.notfound-body{
width:100%;
height:100%;
background: rgba(255,255,255,0.08);
border-radius:12px;
display:flex;
flex-direction:column;
justify-content:center;
align-items:center;
backdrop-filter: blur(12px);
border:1px solid rgba(255,255,255,0.2);
box-shadow:0 15px 40px rgba(0,0,0,0.4);
}

/* TEXT */

.notfound-text{
color:#ffb4b4;
font-size:18px;
font-weight:600;
margin-top:6px;
}

/* ICON */

.notfound-icon{
font-size:24px;
}

/* LID DROP ANIMATION */

@keyframes lidDrop{

0%{
transform: rotateX(-120deg);
opacity:0;
}

100%{
transform: rotateX(0deg);
opacity:1;
}

}


</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* metric text color fix */

[data-testid="stMetricValue"]{
color:white !important;
font-size:36px;
font-weight:700;
}

[data-testid="stMetricLabel"]{
color:#cbd5f5 !important;
font-size:14px;
}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

.ai-box{
background: rgba(255,255,255,0.08);
border-radius:16px;
padding:20px;
margin-top:20px;
backdrop-filter: blur(12px);
border:1px solid rgba(255,255,255,0.15);
box-shadow:0 10px 30px rgba(0,0,0,0.4);
color:white;
font-size:16px;
line-height:1.6;
}

</style>
""", unsafe_allow_html=True)






# ---------------------------------------------------
# HERO HEADER
# ---------------------------------------------------

st.markdown("""
<div class="hero">
<h1>🔎 Lost & Found Reunion</h1>
<p>AI Powered Smart Campus Lost Item Finder</p>
</div>
""", unsafe_allow_html=True)



# ---------------------------------------------------
# METRICS DASHBOARD
# ---------------------------------------------------

col1,col2,col3 = st.columns(3)

col1.metric("Items Indexed","500")
col2.metric("Search Model","CLIP")
col3.metric("Search Type","Text + Image")

st.divider()

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer, preprocess

model, tokenizer, preprocess = load_model()

device = "cpu"
model = model.to(device)

# ---------------------------------------------------
# VECTOR DATABASE
# ---------------------------------------------------

client = chromadb.PersistentClient(path="vector_db")
collection = client.get_collection(name="lost_items")

# ---------------------------------------------------
# AI EXPLANATION
# ---------------------------------------------------

def explain_match(query, metadata):

    try:

        prompt = f"""
A student lost an item described as: "{query}"

The AI search system retrieved this item:
Product Name: {metadata['product_name']}
Category: {metadata['category']}
Description: {metadata['description']}

Explain clearly why this item is likely a match.

Write a helpful explanation in 4–5 sentences that describes:
• similarity of object type
• similarity of features
• why the AI model considered this a match

Use simple language suitable for students.
"""

        response = ollama.chat(
            model="mistral",
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]

    except:
        return "AI explanation unavailable."

# ---------------------------------------------------
# SEARCH CARD
# ---------------------------------------------------

col1,col2,col3 = st.columns([1,2,1])

with col2:

    st.markdown('<div class="search-card">', unsafe_allow_html=True)

    st.subheader("🔎 Search Lost Item")

    query = st.text_input(
        "Describe your lost item",
        placeholder="Example: black wireless headphones"
    )

    search_btn = st.button("Search")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------
# TEXT SEARCH
# ---------------------------------------------------
if search_btn:

    if query:

        tokens = tokenizer([f"a photo of {query}"]).to(device)

        with torch.no_grad():
            text_features = model.encode_text(tokens)

        text_features /= text_features.norm(dim=-1, keepdim=True)

        query_embedding = text_features[0].cpu().numpy().tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4,
            include=["metadatas","distances"]
        )

        valid_results=[]

        for i in range(len(results["ids"][0])):

            metadata=results["metadatas"][0][i]
            distance=results["distances"][0][i]

            confidence=round((1/(1+abs(distance)))*100,2)

            if confidence>39:
                valid_results.append((metadata,confidence))

        st.markdown("### 🔍 Search Results")

        if len(valid_results)==0:

            st.markdown("""
            <div class="notfound-wrapper">
                <div class="notfound-card">
                    <div class="notfound-lid"></div>
                    <div class="notfound-body">
                        <div class="notfound-icon">❌</div>
                        <div class="notfound-text">Item not found</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:

            cols=st.columns(4)
            ai_explanation = None

            for i,(metadata,confidence) in enumerate(valid_results):

                img=Image.open(metadata["image_path"]).resize((250,250))

                with cols[i]:

                    st.markdown('<div class="product-card">',unsafe_allow_html=True)

                    st.image(img,use_container_width=True)

                    st.markdown(
                        f'<div class="product-title">{metadata["product_name"]}</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="product-category">{metadata["category"]}</div>',
                        unsafe_allow_html=True
                    )

                    st.write(metadata["description"])

                    st.markdown(
                        f'<div class="confidence-medium">Match: {confidence}%</div>',
                        unsafe_allow_html=True
                    )

                    if i == 0:
                        ai_explanation = explain_match(query, metadata)

                    st.markdown('</div>',unsafe_allow_html=True)

            # AFTER LOOP SHOW AI EXPLANATION

            if ai_explanation:

                st.markdown("### 🤖 AI Match Explanation")

                st.markdown(f"""
                <div class="ai-box">
                {ai_explanation}
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------
# IMAGE SEARCH
# ---------------------------------------------------

st.divider()

st.markdown("### 📷 Search by Image")

uploaded_file = st.file_uploader(
    "Upload photo of lost item",
    type=["jpg","png","jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image,width=250)

    if st.button("Search by Image"):

        img = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        query_embedding=image_features[0].cpu().numpy().tolist()

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=4,
            include=["metadatas","distances"]
        )

        valid_results=[]

        for i in range(len(results["ids"][0])):

            metadata=results["metadatas"][0][i]
            distance=results["distances"][0][i]

            confidence=round((1/(1+abs(distance)))*100,2)

            if confidence>50:
                valid_results.append((metadata,confidence))

        st.markdown("### 🔍 Similar Items")

        if len(valid_results)==0:

            st.error("❌ Item not found in campus database")

        else:

            cols=st.columns(4)
            ai_explanation = None

            for i,(metadata,confidence) in enumerate(valid_results):

                img=Image.open(metadata["image_path"]).resize((250,200))

                with cols[i]:

                    st.markdown('<div class="product-card">',unsafe_allow_html=True)

                    st.image(img,use_container_width=True)

                    st.markdown(
                        f'<div class="product-title">{metadata["product_name"]}</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        f'<div class="product-category">{metadata["category"]}</div>',
                        unsafe_allow_html=True
                    )

                    st.write(metadata["description"])

                    st.markdown(
                        f'<div class="confidence-medium">Match: {confidence}%</div>',
                        unsafe_allow_html=True
                    )

                    if i==0:
                        ai_explanation = explain_match("uploaded image",metadata)

                    st.markdown('</div>',unsafe_allow_html=True)

            # AFTER LOOP SHOW AI EXPLANATION

            if ai_explanation:

                st.markdown("### 🤖 AI Match Explanation")

                st.markdown(f"""
                <div class="ai-box">
                {ai_explanation}
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------
# DATASET ANALYTICS
# ---------------------------------------------------

import pandas as pd

st.divider()

st.markdown("### 📊 Dataset Analytics")

data = {
    "Category":[
        "Headphones",
        "Earbuds",
        "Phones",
        "Laptops",
        "Bottles"
    ],
    "Items":[120,100,90,80,110]
}

df = pd.DataFrame(data)

st.bar_chart(df.set_index("Category"))

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("""
<footer>
Built with ❤️ using AI Vision Models <br>
<b>Developer:Bhavya</b>  | Lost & Found AI System
</footer>
""",unsafe_allow_html=True)

