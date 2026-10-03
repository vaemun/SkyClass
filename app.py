import streamlit as st
from db import (
    add_message,
    add_source,
    create_chat,
    delete_chat,
    delete_source,
    get_chat,
    get_chats,
    get_messages,
    get_sources,
    init_db,
    update_chat_title,
)
from vector_functions import (
    delete_chat_vectors,
    delete_source_vectors,
    get_relevant_docs,
    process_document,
    process_image,
    process_url,
    stream_response,
    IMAGE_EXTS,
)

st.set_page_config(page_title="Study Notes Assistant", layout="wide")
init_db()

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

with st.sidebar:
    st.title("Study Notes")

    if st.button("+ New Chat", use_container_width=True, type="primary"):
        chat_id = create_chat()
        st.session_state.current_chat_id = chat_id
        st.rerun()

    st.divider()

    chats = get_chats()
    for chat in chats:
        c1, c2 = st.columns([4, 1])
        is_active = chat["id"] == st.session_state.current_chat_id
        with c1:
            label = chat["title"][:28] + (".." if len(chat["title"]) > 28 else "")
            if st.button(
                label,
                key=f"chat_{chat['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.current_chat_id = chat["id"]
                st.rerun()
        with c2:
            if st.button("🗑", key=f"del_{chat['id']}"):
                delete_chat_vectors(chat["id"])
                delete_chat(chat["id"])
                if st.session_state.current_chat_id == chat["id"]:
                    st.session_state.current_chat_id = None
                st.rerun()

    if st.session_state.current_chat_id:
        st.divider()
        st.write("**Sources**")
        sources = get_sources(st.session_state.current_chat_id)
        if not sources:
            st.caption("_No sources_")
        for src in sources:
            c1, c2 = st.columns([4, 1])
            icons = {"document": ":page_facing_up:", "image": ":framed_picture:", "link": ":link:"}
            icon = icons.get(src["type"], ":page_facing_up:")
            label = src["name"][:22] + (".." if len(src["name"]) > 22 else "")
            with c1:
                st.write(f"{icon} {label}")
            with c2:
                if st.button("✕", key=f"src_del_{src['id']}"):
                    delete_source_vectors(
                        st.session_state.current_chat_id, src["name"]
                    )
                    delete_source(src["id"])
                    st.rerun()

        if st.button("← Back to Chats", use_container_width=True):
            st.session_state.current_chat_id = None
            st.rerun()

if st.session_state.current_chat_id is None:
    st.title("Study Notes Assistant")
    st.markdown(
        "Upload PDFs and images of your notes, then ask questions. "
        "Answers are grounded in your uploaded material with source citations."
    )
else:
    chat_id = st.session_state.current_chat_id
    chat_info = get_chat(chat_id)
    if chat_info:
        st.title(chat_info["title"])

    messages = get_messages(chat_id)
    for msg in messages:
        with st.chat_message(msg["sender"]):
            st.write(msg["content"])

    with st.expander("Upload study material (PDF, image, or text file)", expanded=False):
        with st.form(key=f"source_form_{chat_id}"):
            uploaded_file = st.file_uploader(
                "Upload file",
                type=[
                    "pdf",
                    "txt",
                    "csv",
                    "docx",
                    "md",
                    "png",
                    "jpg",
                    "jpeg",
                    "gif",
                    "bmp",
                    "webp",
                ],
            )
            url_input = st.text_input("Or enter a web URL")
            submitted = st.form_submit_button("Add Source", use_container_width=True)

            if submitted:
                if uploaded_file:
                    try:
                        fname = uploaded_file.name
                        ext = (
                            fname.rsplit(".", 1)[-1].lower()
                            if "." in fname
                            else ""
                        )
                        src_type = "image" if ext in IMAGE_EXTS else "document"

                        if ext in IMAGE_EXTS:
                            chunk_count = process_image(
                                uploaded_file, chat_id, fname
                            )
                        else:
                            chunk_count = process_document(
                                uploaded_file, chat_id, fname
                            )

                        add_source(chat_id, fname, src_type)
                        st.success(f"Added {fname} ({chunk_count} chunks)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
                elif url_input:
                    try:
                        chunk_count = process_url(url_input, chat_id)
                        add_source(chat_id, url_input, "link")
                        st.success(f"Added URL ({chunk_count} chunks)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing URL: {e}")

    prompt = st.chat_input("Ask a question about your study notes...")
    if prompt:
        add_message(chat_id, "user", prompt)

        chat = get_chat(chat_id)
        if chat and chat["title"] == "New Chat":
            new_title = prompt[:45] + ("..." if len(prompt) > 45 else "")
            update_chat_title(chat_id, new_title)

        with st.chat_message("user"):
            st.write(prompt)

        docs = get_relevant_docs(chat_id, prompt)

        with st.chat_message("assistant"):
            stream = stream_response(chat_id, prompt, docs=docs)
            response = st.write_stream(stream)

            if docs:
                st.divider()
                st.caption("**Sources:**")
                seen = set()
                for doc in docs:
                    sname = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page", "?")
                    key = (sname, page)
                    if key not in seen:
                        seen.add(key)
                        st.caption(f"- {sname}, page {page}")

        full_response = response if response else ""
        if docs:
            seen = set()
            source_lines = []
            for doc in docs:
                sname = doc.metadata.get("source_name", doc.metadata.get("source", "unknown"))
                page = doc.metadata.get("page", "?")
                key = (sname, page)
                if key not in seen:
                    seen.add(key)
                    source_lines.append(f"{sname}, page {page}")
            if source_lines:
                full_response += "\n\n**Sources:** " + "; ".join(source_lines)

        add_message(chat_id, "ai", full_response)
        st.rerun()
