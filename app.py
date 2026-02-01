import streamlit as st
import openai
import numpy as np
from typing import List
import time
import uuid

# Small helper functions and cached embeddings
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
	parts = []
	start = 0
	while start < len(text):
		end = start + chunk_size
		parts.append(text[start:end])
		start = max(end - overlap, end)
	return parts

@st.cache_data(show_spinner=False)
def get_embeddings(texts: List[str], api_key: str, model: str = "text-embedding-3-small"):
	# Returns numpy array of embeddings (len(texts), dim)
	openai.api_key = api_key
	# Be robust to empty requests
	if not texts:
		return np.zeros((0, 1))
	resp = openai.Embedding.create(model=model, input=texts)
	vecs = [item["embedding"] for item in resp["data"]]
	return np.array(vecs, dtype=np.float32)

def cosine_sim(a: np.ndarray, b: np.ndarray):
	# a: (n, d) b: (m, d) -> (n, m)
	a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
	b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
	return np.dot(a_norm, b_norm.T)

def build_prompt(chunks: List[str], question: str, top_indices: List[int]):
	selected = "\n\n---\n\n".join(chunks[i] for i in top_indices)
	prompt = (
		"You are a helpful assistant. Use only the provided document excerpts to answer the user's question. "
		"If the answer is not contained in the excerpts, say you don't know.\n\n"
		f"Context excerpts:\n{selected}\n\nQuestion: {question}\n\nAnswer concisely:"
	)
	return prompt

def query_openai_chat(api_key: str, system: str, user_prompt: str, temperature: float = 0.2, model: str = "gpt-3.5-turbo"):
	openai.api_key = api_key
	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": user_prompt}
	]
	resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=temperature)
	return resp["choices"][0]["message"]["content"].strip()

# Initialize session state
if "history" not in st.session_state:
	st.session_state.history = []  # list of (role, text)
if "snapshots" not in st.session_state:
	st.session_state.snapshots = {}  # name -> {"history": ..., "timestamp": ...}

# Sidebar: API key and options
st.set_page_config(page_title="Doc Chat", layout="wide")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
chat_model = st.sidebar.selectbox(
	"Chat model",
	options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
	index=0,
	help="Select chat/completion model. Default: gpt-3.5-turbo"
)
embedding_model = st.sidebar.selectbox(
	"Embedding model",
	options=["text-embedding-3-small", "text-embedding-3-large"],
	index=0,
	help="Select embedding model. Default: text-embedding-3-small"
)
top_k = st.sidebar.slider("Top document chunks to use", min_value=1, max_value=10, value=3)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2)
chunk_size = st.sidebar.number_input("Chunk size (chars)", value=1000, step=100)
overlap = st.sidebar.number_input("Chunk overlap (chars)", value=200, step=50)

st.sidebar.markdown("Quick actions:")
if st.sidebar.button("Clear chat history"):
	st.session_state.history = []

if st.sidebar.button("Clear snapshots"):
	st.session_state.snapshots = {}

st.sidebar.markdown("Environment:")
st.sidebar.caption("Recommended: create a .venv and pip install the packages below.")

# <-- added: attribution in sidebar -->
st.sidebar.markdown("Created by Mr Hezbullah Shah — [www.MrHezbu.com](http://www.MrHezbu.com)")

# Main UI
st.title("Chat with your TXT Document")
st.markdown("Upload a .txt file and ask questions about its content. Embeddings are cached for speed.")

uploaded = st.file_uploader("Upload TXT file", type=["txt"])
document_text = ""
if uploaded is not None:
	bytes_data = uploaded.read()
	try:
		document_text = bytes_data.decode("utf-8")
	except:
		document_text = bytes_data.decode("latin-1")

if not document_text:
	st.info("Upload a TXT file to begin.")
	# still show small help
	st.markdown("- Use your OpenAI API key in the sidebar.\n- Adjust Top-K and chunk size for performance/accuracy.")
	# End early if no doc
else:
	st.subheader("Document preview")
	st.text_area("Preview (first 20k chars)", value=document_text[:20000], height=200)

	# Chunk and embed (cached)
	chunks = chunk_text(document_text, chunk_size=chunk_size, overlap=overlap)
	with st.spinner("Computing/Loading embeddings (cached)..."):
		embs = get_embeddings(chunks, api_key or "no_key", model=embedding_model)
	# Quick stats
	st.write(f"Document split into {len(chunks)} chunks. Embedding dim: {embs.shape[1] if embs.size else 0}")

	# Chat area
	st.subheader("Chat")
	question = st.text_input("Your question", key="question_input")
	send = st.button("Ask")

	# Snapshot controls
	st.write("Memory snapshots:")
	col1, col2 = st.columns([2, 1])
	with col1:
		snap_name = st.text_input("Snapshot name", key="snap_name")
	with col2:
		if st.button("Save snapshot"):
			name = snap_name.strip() or f"snapshot-{time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:6]}"
			st.session_state.snapshots[name] = {"history": list(st.session_state.history), "timestamp": time.time()}
			st.success(f"Saved snapshot '{name}'")
	if st.session_state.snapshots:
		choice = st.selectbox("Restore snapshot", options=["(none)"] + list(st.session_state.snapshots.keys()))
		if choice and choice != "(none)" and st.button("Restore"):
			st.session_state.history = list(st.session_state.snapshots[choice]["history"])
			st.success(f"Restored snapshot '{choice}'")

	# On ask: compute query embedding, find top_k chunks, call OpenAI
	if send:
		if not api_key:
			st.error("Provide OpenAI API key in the sidebar.")
		elif not document_text:
			st.error("Upload a document first.")
		else:
			with st.spinner("Generating answer..."):
				# embed query
				q_emb = get_embeddings([question], api_key, model=embedding_model)[0:1]  # shape (1, dim)
				sims = cosine_sim(embs, q_emb).flatten()  # (n_chunks,)
				top_idx = list(np.argsort(-sims)[:top_k])
				# Build prompt
				system = "You are a helpful assistant answering from provided document excerpts."
				user_prompt = build_prompt(chunks, question, top_idx)
				# Optionally include last few chat turns in system instruction or appended—keep concise
				if st.session_state.history:
					# append last user and assistant turns to prompt (limit to last 6 entries)
					recent = "\n\n".join([f"{r}: {t}" for r, t in st.session_state.history[-6:]])
					user_prompt = f"{recent}\n\n{user_prompt}"
				# Query OpenAI chat
				try:
					answer = query_openai_chat(api_key, system, user_prompt, temperature=temperature, model=chat_model)
				except Exception as e:
					st.error(f"OpenAI API error: {e}")
					answer = ""
				# Save to history
				st.session_state.history.append(("user", question))
				st.session_state.history.append(("assistant", answer))
				# Display streaming style
				st.markdown("**Answer:**")
				st.write(answer)

	# Show full conversation
	if st.session_state.history:
		st.subheader("Conversation")
		for role, text in st.session_state.history:
			if role == "user":
				st.markdown(f"**You:** {text}")
			else:
				st.markdown(f"**Assistant:** {text}")

st.markdown("---")
st.markdown("Created by Mr Hezbullah Shah — [www.MrHezbu.com](http://www.MrHezbu.com)")
st.caption("Required packages: streamlit, openai, numpy. Example: python -m venv .venv && .venv\\Scripts\\pip install -U pip && .venv\\Scripts\\pip install streamlit openai numpy")
