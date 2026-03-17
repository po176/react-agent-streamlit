
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import os

st.set_page_config(page_title="Groq Agent", page_icon="Robot", layout="wide")
st.title("ReAct Agent")
st.caption("Powered by Groq (LLaMA 3) + LangGraph")

with st.sidebar:
    st.header("Configuration")

    groq_api_key = st.text_input("Groq API Key", type="password",
                                  value=os.environ.get("GROQ_API_KEY", ""))
    tavily_key   = st.text_input("Tavily API Key (for search)", type="password",
                                  value=os.environ.get("TAVILY_API_KEY", ""))

    st.divider()
    st.header("Tool Settings")
    use_search = st.toggle("Enable Web Search", value=True)

    if use_search:
        st.success("Web Search: ON")
    else:
        st.warning("Web Search: OFF - chat only")

    st.divider()
    st.header("Model Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens  = st.slider("Max Tokens", 256, 4096, 1024, 128)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()
    if use_search and not tavily_key:
        st.error("Please enter your Tavily API key, or toggle Web Search OFF.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    web_tool = TavilySearchResults(max_results=3, tavily_api_key=tavily_key)

    if use_search:
        active_tools = [web_tool]
    else:
        active_tools = []

    agent = create_react_agent(llm, active_tools)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                history = [
                    HumanMessage(content=m["content"]) if m["role"] == "user"
                    else {"role": "assistant", "content": m["content"]}
                    for m in st.session_state.messages
                ]
                response = agent.invoke({"messages": history})
                answer   = response["messages"][-1].content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {e}")
