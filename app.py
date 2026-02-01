import streamlit as st
import openai
import numpy as np
from typing import List
import time
import uuid  # import uuid for unique snapshot names

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

# Sidebar: API key and essential options (kept minimal to avoid scrolling)
st.set_page_config(page_title="Doc Chat", layout="wide")  # set page title and layout
st.sidebar.title("Settings")  # sidebar title
api_key = st.sidebar.text_input("OpenAI API Key", type="password")  # API key input (visible)
chat_model = st.sidebar.selectbox(  # select chat model (visible)
	"Chat model",  # label
	options=["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],  # options list
	index=0,  # default selection index
	help="Select chat/completion model. Default: gpt-3.5-turbo"  # help text
)

# Advanced settings moved into a collapsed expander to keep sidebar compact
with st.sidebar.expander("Advanced (click to expand)", expanded=False):  # collapsed by default
	embedding_model = st.selectbox(  # embedding model selection inside expander
		"Embedding model",
		options=["text-embedding-3-small", "text-embedding-3-large"],
		index=0,
		help="Select embedding model. Default: text-embedding-3-small"
	)
	top_k = st.slider("Top document chunks to use", min_value=1, max_value=10, value=3)  # number of chunks to use
	temperature = st.slider("Temperature", 0.0, 1.0, 0.2)  # sampling temperature
	chunk_size = st.number_input("Chunk size (chars)", value=1000, step=100)  # chunk size in chars
	overlap = st.number_input("Chunk overlap (chars)", value=200, step=50)  # overlap between chunks
	st.markdown("Quick actions:")  # quick actions heading
	if st.button("Clear chat history"):  # button to clear history placed in expander
		st.session_state.history = []  # clear conversation history
	if st.button("Clear snapshots"):  # button to clear snapshots placed in expander
		st.session_state.snapshots = {}  # clear snapshot store
	st.markdown("Environment:")  # environment note
	st.caption("Recommended: create a .venv and pip install the packages below.")  # short note about venv

# Add concise attribution with prefix
st.sidebar.markdown("Created by: [Mr Hezbullah Shah](http://www.MrHezbu.com)")  # author attribution as linked name

# Main UI
st.title("Chat with your TXT Document")  # main title
st.markdown("Upload a .txt file and ask questions about its content. Embeddings are cached for speed.")  # short instructions

uploaded = st.file_uploader("Upload TXT file", type=["txt"])  # file uploader widget
document_text = ""  # initialize document text variable
if uploaded is not None:  # if a file is uploaded
	bytes_data = uploaded.read()  # read file bytes
	try:
		document_text = bytes_data.decode("utf-8")  # try decoding as utf-8
	except:
		document_text = bytes_data.decode("latin-1")  # fallback to latin-1 if utf-8 fails

if not document_text:  # if no document text available
	st.info("Upload a TXT file to begin.")  # show info prompt
	# still show small help
	st.markdown("- Use your OpenAI API key in the sidebar.\n- Adjust Top-K and chunk size for performance/accuracy.")  # tips
	# End early if no doc
else:  # when a document has been uploaded and decoded
	st.subheader("Document preview")  # preview header
	st.text_area("Preview (first 20k chars)", value=document_text[:20000], height=200)  # show preview of first 20k chars

	# Chunk and embed (cached)
	chunks = chunk_text(document_text, chunk_size=chunk_size, overlap=overlap)  # split document into chunks
	with st.spinner("Computing/Loading embeddings (cached)..."):  # spinner while embeddings are computed/loaded
		embs = get_embeddings(chunks, api_key or "no_key", model=embedding_model)  # get cached embeddings
	# Quick stats
	st.write(f"Document split into {len(chunks)} chunks. Embedding dim: {embs.shape[1] if embs.size else 0}")  # show simple stats

	# Chat area
	st.subheader("Chat")  # chat header
	question = st.text_input("Your question", key="question_input")  # question input
	send = st.button("Ask")  # ask button

	# Snapshot controls (moved to main area to avoid sidebar overflow)
	st.write("Memory snapshots:")  # snapshots header
	col1, col2 = st.columns([2, 1])  # two-column layout for snapshot name + save button
	with col1:
		snap_name = st.text_input("Snapshot name", key="snap_name")  # input for snapshot name
	with col2:
		if st.button("Save snapshot"):  # button to save snapshot
			name = snap_name.strip() or f"snapshot-{time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:6]}"  # generate name if empty
			st.session_state.snapshots[name] = {"history": list(st.session_state.history), "timestamp": time.time()}  # store snapshot
			st.success(f"Saved snapshot '{name}'")  # show success
	if st.session_state.snapshots:  # if any snapshots exist
		choice = st.selectbox("Restore snapshot", options=["(none)"] + list(st.session_state.snapshots.keys()))  # choose snapshot
		if choice and choice != "(none)" and st.button("Restore"):  # restore button
			st.session_state.history = list(st.session_state.snapshots[choice]["history"])  # restore history
			st.success(f"Restored snapshot '{choice}'")  # show success

	# On ask: compute query embedding, find top_k chunks, call OpenAI
	if send:  # if Ask was clicked
		if not api_key:  # require API key
			st.error("Provide OpenAI API key in the sidebar.")  # error message
		elif not document_text:  # require document
			st.error("Upload a document first.")  # error message
		else:
			with st.spinner("Generating answer..."):  # spinner while generating
				# embed query
				q_emb = get_embeddings([question], api_key, model=embedding_model)[0:1]  # shape (1, dim)
				sims = cosine_sim(embs, q_emb).flatten()  # (n_chunks,) similarity scores
				top_idx = list(np.argsort(-sims)[:top_k])  # pick top-k chunk indices
				# Build prompt
				system = "You are a helpful assistant answering from provided document excerpts."  # system message
				user_prompt = build_prompt(chunks, question, top_idx)  # build user prompt with context
				# Optionally include last few chat turns in system instruction or appended—keep concise
				if st.session_state.history:  # if history exists
					# append last user and assistant turns to prompt (limit to last 6 entries)
					recent = "\n\n".join([f"{r}: {t}" for r, t in st.session_state.history[-6:]])  # recent turns
					user_prompt = f"{recent}\n\n{user_prompt}"  # prepend recent turns to prompt
				# Query OpenAI chat
				try:
					answer = query_openai_chat(api_key, system, user_prompt, temperature=temperature, model=chat_model)  # call chat API with selected model
				except Exception as e:
					st.error(f"OpenAI API error: {e}")  # display API error
					answer = ""  # fallback empty answer
				# Save to history
				st.session_state.history.append(("user", question))  # record user question
				st.session_state.history.append(("assistant", answer))  # record assistant answer
				# Display streaming style
				st.markdown("**Answer:**")  # answer label
				st.write(answer)  # show answer text

	# Show full conversation
	if st.session_state.history:  # if history is non-empty
		st.subheader("Conversation")  # conversation header
		for role, text in st.session_state.history:  # iterate through history tuples
			if role == "user":  # user message
				st.markdown(f"**You:** {text}")  # format user message
			else:  # assistant message
				st.markdown(f"**Assistant:** {text}")  # format assistant message

st.markdown("---")  # divider
st.markdown("Created by: [Mr Hezbullah Shah](http://www.MrHezbu.com)")  # footer attribution with prefix
st.caption("Required packages: streamlit, openai, numpy. Example: python -m venv .venv && .venv\\Scripts\\pip install -U pip && .venv\\Scripts\\pip install streamlit openai numpy")  # caption with install hint
