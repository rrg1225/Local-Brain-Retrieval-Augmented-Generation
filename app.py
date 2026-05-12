"""
Streamlit 界面：流式对话、来源溯源、多项目空间隔离、本地 GPU Embedding/Rerank、
基于 session_state 的多轮上下文。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
from llama_index.core.llms import ChatMessage, MessageRole

from rag_engine import (
    _load_index_state,
    _save_index_state,
    bootstrap_index,
    build_chat_engine,
    build_memory_for_completed_turns,
    delete_project_space,
    delete_workspace_nodes,
    format_sources_md,
    get_llm_instance,
    incremental_upsert_file,
    initial_scan,
    list_existing_collections,
)
from config import get_config, save_persistent_workspace, remove_persistent_workspace
from watcher import DebouncedIndexHandler, start_observer

USER_AVATAR = "🧑‍💻"
ASSISTANT_AVATAR = "🧠"

# (展示名, 模型 ID，写入 st.session_state.selected_llm)
LLM_CHOICES: list[tuple[str, str]] = [
    ("Gemini 2.5 Flash (Google)", "gemini-2.5-flash"),
    ("Qwen-Plus (Aliyun)", "qwen-plus"),
    ("Qwen-Max (Aliyun)", "qwen-max"),
]

QUICK_PROMPTS: list[tuple[str, str]] = [
    ("整体架构", "帮我梳理当前工作区项目的整体架构与技术栈。"),
    ("核心代码", "解释最近可能被修改过的核心业务代码在做什么，并指出关键类/方法。"),
    ("依赖关系", "根据本地文档与代码，总结数据库或模块之间的依赖关系。"),
]


def _save_workspace_to_json(space_name: str, path: Path) -> None:
    """将新工作区路径持久化到当前空间下。"""
    save_persistent_workspace(space_name, path)


def _sync_memory_from_ui_messages(memory, ui_messages: list[dict]) -> None:
    """用当前界面已确认的多轮消息重建记忆，避免与 CondensePlusContext 内部状态不一致。"""
    memory.reset()
    for row in ui_messages:
        role = MessageRole.USER if row["role"] == "user" else MessageRole.ASSISTANT
        memory.put(ChatMessage(role=role, content=row["content"]))


@st.cache_resource
def _bootstrap_rag(space_name: str):
    cfg, _embed_model, index = bootstrap_index(collection_name=space_name)
    handler = DebouncedIndexHandler(index, cfg, collection_name=space_name)
    observer = start_observer(handler, cfg.workspaces)
    return cfg, index, observer, handler


def _format_messages_as_markdown(messages: list[dict]) -> str:
    lines: list[str] = [
        "# 第二大脑 · 对话导出",
        "",
        f"_导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "---",
        "",
    ]
    for i, m in enumerate(messages, start=1):
        role_label = "用户" if m["role"] == "user" else "助手"
        lines.append(f"## 轮次 {i} · {role_label}")
        lines.append("")
        lines.append(m.get("content", "").strip())
        lines.append("")
        src = m.get("sources")
        if src and not str(src).startswith("_（"):
            lines.append("### 来源溯源")
            lines.append("")
            lines.append(str(src).strip())
            lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _chat_history_path(space_name: str) -> Path:
    return Path(f".chat_history_{space_name}.json")


def _load_chat_history(space_name: str) -> list[dict]:
    """从 .chat_history_{space_name}.json 加载持久化对话，容错损坏/缺失。"""
    path = _chat_history_path(space_name)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return []


def _save_chat_history(space_name: str, messages: list[dict]) -> None:
    """将当前对话持久化到空间专属 JSON 文件（原子写入）。"""
    path = _chat_history_path(space_name)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
    except OSError:
        pass


def _render_sources_expander(sources_md: str) -> None:
    """溯源区：以 Markdown 原生渲染，支持代码块预览引用片段。"""
    with st.expander("🔍 展开查看参考的源码片段", expanded=False):
        if not sources_md or str(sources_md).strip().startswith("_（"):
            st.caption("本轮未命中可展示路径的检索节点。")
            return
        st.caption("以下为检索到的本地参考片段，包含文件路径与内容预览。")
        st.divider()
        st.markdown(sources_md)


def _render_sidebar_dashboard(cfg, index, observer, handler,
                               messages: list[dict], space: str) -> None:
    """渲染侧边栏（标题和空间选择器由 run_app 提前渲染）。"""
    st.sidebar.markdown("##### 对话模型 (LLM)")
    labels = [x[0] for x in LLM_CHOICES]
    id_by_label = {x[0]: x[1] for x in LLM_CHOICES}
    ids = [x[1] for x in LLM_CHOICES]
    if "selected_llm" not in st.session_state:
        st.session_state.selected_llm = ids[0]
    try:
        select_index = ids.index(st.session_state.selected_llm)
    except ValueError:
        select_index = 0
        st.session_state.selected_llm = ids[0]

    chosen_label = st.sidebar.selectbox(
        "选择对话大模型",
        labels,
        index=select_index,
        help="切换后下一轮起使用新模型。Embedding / Rerank 始终由本地 GPU 完成。",
    )
    new_id = id_by_label[chosen_label]
    if new_id != st.session_state.selected_llm:
        st.session_state.selected_llm = new_id
        st.rerun()

    st.sidebar.caption(f"当前 LLM：`{st.session_state.selected_llm}`")
    st.sidebar.caption(f"Embedding：`{cfg.local_embed_model}`")
    st.sidebar.caption(f"Rerank：`{cfg.local_rerank_model}`")
    st.sidebar.caption(f"活跃空间：`{space}`")

    st.sidebar.divider()
    st.sidebar.markdown("##### 运行配置")
    c1, c2 = st.sidebar.columns(2)
    with c1:
        st.metric("防抖 (s)", f"{cfg.debounce:g}")
    with c2:
        cfg.top_k = st.slider(
            "Top-K (Rerank)", min_value=2, max_value=15, value=cfg.top_k,
            help="Rerank 后最终保留的参考节点数",
        )

    api_label = cfg.proxy_base.strip() if cfg.proxy_base else "默认 Google API"
    st.sidebar.text_area("API 基址 (Worker)", api_label, height=68, disabled=True)

    tunnel = cfg.http_proxy or "未设置"
    st.sidebar.text_input("本地 HTTP 隧道", tunnel, disabled=True)

    st.sidebar.caption("Chat 模式：`condense_plus_context` + Local Rerank")

    st.sidebar.divider()

    # --- 知识库管理 ---
    with st.sidebar.expander("📁 知识库管理", expanded=False):
        dynamic_dir = (Path("./dynamic_workspace").resolve() / space)

        st.markdown("##### 批量上传")
        uploaded_files = st.file_uploader(
            "拖拽或选择文件（支持 .zip 批量解压）",
            key=f"kb_file_uploader_{space}",
            label_visibility="collapsed",
            accept_multiple_files=True,
        )
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name.lower().endswith(".zip"):
                    zip_path = dynamic_dir / uploaded_file.name
                    extract_dir = dynamic_dir / uploaded_file.name.rsplit(".", 1)[0]
                    try:
                        dynamic_dir.mkdir(parents=True, exist_ok=True)
                        with open(zip_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        extract_dir.mkdir(parents=True, exist_ok=True)
                        with zipfile.ZipFile(zip_path, "r") as zf:
                            for zip_info in zf.infolist():
                                try:
                                    zip_info.filename = zip_info.filename.encode("cp437").decode("gbk")
                                except Exception:
                                    zip_info.filename = zip_info.filename.encode("utf-8", "ignore").decode("utf-8", "ignore")
                                zf.extract(zip_info, extract_dir)
                        zip_path.unlink()
                    except (OSError, zipfile.BadZipFile) as e:
                        st.error(f"解压 {uploaded_file.name} 失败：{e}")
                        continue

                    if extract_dir not in cfg.workspaces:
                        cfg.workspaces.append(extract_dir)
                        _save_workspace_to_json(space, extract_dir)
                    with st.spinner(f"正在并发扫描 {uploaded_file.name} 解压内容..."):
                        progress_bar = st.progress(0, text="准备扫描...")
                        def _zip_progress(current: int, total: int):
                            progress_bar.progress(
                                current / total,
                                text=f"已入库 {current}/{total} 批次",
                            )
                        n_zip = initial_scan(index, cfg, collection_name=space,
                                            progress_callback=_zip_progress)
                        progress_bar.empty()
                    observer.schedule(handler, str(extract_dir), recursive=True)
                    st.success(f"入库 {n_zip} 个节点 → 空间 [{space}]")
                    st.toast(f"压缩包 {uploaded_file.name} 解压并全量入库成功！")
                else:
                    saved_path = dynamic_dir / uploaded_file.name
                    try:
                        dynamic_dir.mkdir(parents=True, exist_ok=True)
                        with open(saved_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        with st.spinner(f"正在入库 {uploaded_file.name}..."):
                            incremental_upsert_file(index, cfg, saved_path,
                                                    collection_name=space)
                        st.toast(f"文件 {uploaded_file.name} 入库成功！")
                    except OSError as e:
                        st.error(f"文件保存或入库失败：{e}")

        st.divider()
        st.markdown("##### 动态添加本地文件夹")
        new_dir_str = st.text_input(
            "外部文件夹绝对路径",
            key=f"kb_dir_input_{space}",
            placeholder="C:\\Users\\... 或 /home/...",
        )
        if st.button("扫描并监听此目录", key=f"kb_dir_btn_{space}", use_container_width=True):
            if not new_dir_str.strip():
                st.error("请输入路径")
            else:
                new_path = Path(new_dir_str.strip()).resolve()
                if not new_path.exists() or not new_path.is_dir():
                    st.error(f"路径不存在或不是目录：{new_path}")
                else:
                    if new_path not in cfg.workspaces:
                        cfg.workspaces.append(new_path)
                    _save_workspace_to_json(space, new_path)
                    with st.spinner("正在并发扫描入库..."):
                        progress_bar = st.progress(0, text="准备扫描...")
                        def _dir_progress(current: int, total: int):
                            progress_bar.progress(
                                current / total,
                                text=f"已入库 {current}/{total} 批次",
                            )
                        n_dir = initial_scan(index, cfg, collection_name=space,
                                            progress_callback=_dir_progress)
                        progress_bar.empty()
                    observer.schedule(handler, str(new_path), recursive=True)
                    st.success(f"入库 {n_dir} 个节点 → 空间 [{space}]")

        st.divider()
        st.markdown("##### 已加载工作区")
        non_dynamic = [w for w in cfg.workspaces if w.name != "dynamic_workspace"]
        if not non_dynamic:
            st.caption("暂无额外加载的工作区")
        else:
            for i, w in enumerate(non_dynamic):
                w_left, w_right = st.columns([3, 1])
                display_name = w.name if len(w.name) <= 36 else w.name[:33] + "..."
                w_left.markdown(f"📁 {display_name}")
                w_left.caption(str(w))
                path_hash = hashlib.md5(str(w).encode()).hexdigest()[:8]
                if w_right.button("✕", key=f"rm_ws_{path_hash}_{space}_{i}",
                                  help=f"移出工作区 {w.name}"):
                    with st.spinner(f"正在清理 {w.name} 的索引节点..."):
                        deleted = delete_workspace_nodes(index, space, w)
                    if w in cfg.workspaces:
                        cfg.workspaces.remove(w)
                    remove_persistent_workspace(space, w)
                    if w.is_dir() and "dynamic_workspace" in w.parts:
                        shutil.rmtree(w, ignore_errors=True)
                    st.toast(f"已清理 {deleted} 个索引节点")
                    st.rerun()

        st.divider()
        st.markdown("##### 已上传文件管理")
        dynamic_dir.mkdir(parents=True, exist_ok=True)
        dynamic_files = sorted(
            [f for f in dynamic_dir.iterdir() if f.is_file()],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not dynamic_files:
            st.caption("暂无动态上传的文件")
        else:
            for f in dynamic_files:
                left, right = st.columns([3, 1])
                display_name = f.name if len(f.name) <= 36 else f.name[:33] + "..."
                left.markdown(f"📄 {display_name}")
                if right.button("🗑️", key=f"del_{f.name}_{space}",
                                help=f"删除 {f.name}"):
                    doc_id = str(f.resolve())
                    try:
                        index.delete_ref_doc(doc_id, delete_from_docstore=True)
                    except Exception:
                        pass
                    state = _load_index_state(space)
                    if doc_id in state:
                        del state[doc_id]
                        _save_index_state(state, space)
                    try:
                        f.unlink()
                    except OSError:
                        pass
                    st.rerun()

    st.sidebar.divider()
    st.sidebar.markdown("##### 导出对话")
    export_md = _format_messages_as_markdown(messages)
    st.sidebar.download_button(
        label="下载为 Markdown",
        data=export_md.encode("utf-8"),
        file_name=f"第二大脑对话_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown; charset=utf-8",
        disabled=len(messages) == 0,
        use_container_width=True,
        help="将当前会话导出为 .md 文件",
    )

    st.sidebar.divider()
    st.sidebar.markdown("##### 会话管理")
    st.sidebar.warning("清空后无法恢复当前窗口内的对话记录。", icon="⚠️")
    confirm = st.sidebar.checkbox("我确认要清空全部对话", value=False)
    if st.sidebar.button(
        "清空对话记录",
        type="primary",
        disabled=not confirm,
        use_container_width=True,
        help="需先勾选确认",
    ):
        st.session_state.messages = []
        _save_chat_history(st.session_state.current_space, [])
        st.session_state.pop("quick_prompt_submit", None)
        st.rerun()


def _render_welcome_panel() -> None:
    _left, center, _right = st.columns([1, 2.2, 1])
    with center:
        st.markdown(
            "<h2 style='text-align:center;margin-bottom:0.25rem;'>第二大脑</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:#666;font-size:1.05rem;'>"
            "连接你的工作区代码、文档与架构图，用对话快速理解与追溯。"
            "</p>",
            unsafe_allow_html=True,
        )
        st.info(
            "在下方输入问题，或点击一条「灵感提示」快速开始。支持多轮追问，例如："
            "「基于上一条，给出重构建议」。"
        )
        st.markdown("###### 灵感提示")
        q1, q2, q3 = st.columns(3)
        for col, (label, full_text), idx in zip((q1, q2, q3), QUICK_PROMPTS, range(3)):
            with col:
                if st.button(label, key=f"qp_{idx}", use_container_width=True):
                    st.session_state.quick_prompt_submit = full_text
                    st.rerun()
                st.caption(full_text)


def run_app() -> None:
    st.set_page_config(page_title="第二大脑", layout="wide", initial_sidebar_state="expanded")

    # --- 空间初始化 ---
    if "current_space" not in st.session_state:
        st.session_state.current_space = "default_space"

    # --- 侧边栏顶部：标题 + 空间管理器 ---
    st.sidebar.markdown("### 第二大脑")
    st.sidebar.caption("本地 RAG 个人助理控制台 · GPU Embedding / Rerank")
    st.sidebar.divider()

    st.sidebar.markdown("##### 项目空间")
    existing = list_existing_collections(get_config())
    current = st.session_state.current_space
    if current not in existing:
        existing = [current] + existing

    col_a, col_b = st.sidebar.columns([4, 1])
    with col_a:
        space_idx = existing.index(current) if current in existing else 0
        chosen = st.selectbox(
            "选择空间", existing, index=space_idx,
            key="space_selector", label_visibility="collapsed",
        )
    with col_b:
        if st.button("＋", key="new_space_btn", help="新建项目空间",
                     use_container_width=True):
            st.session_state.show_new_space_input = True

    if st.session_state.get("show_new_space_input"):
        new_name = st.sidebar.text_input(
            "新空间名称", key="new_space_name_input",
            placeholder="例如：shop_space",
        )
        cc1, cc2 = st.sidebar.columns(2)
        with cc1:
            if st.button("创建", key="confirm_create_space", use_container_width=True):
                name = new_name.strip()
                if name:
                    st.session_state.current_space = name
                    st.session_state.messages = []
                    st.session_state.show_new_space_input = False
                    st.rerun()
                else:
                    st.error("名称不能为空")
        with cc2:
            if st.button("取消", key="cancel_new_space", use_container_width=True):
                st.session_state.show_new_space_input = False
                st.rerun()
    elif chosen != current:
        st.session_state.current_space = chosen
        st.session_state.messages = _load_chat_history(chosen)
        st.rerun()

    if current != "default_space":
        destroy_col1, destroy_col2 = st.sidebar.columns([3, 2])
        with destroy_col1:
            st.caption(f"空间 `{current}`")
        with destroy_col2:
            if st.button("🗑️ 销毁", key="destroy_space_btn",
                         help="永久删除此空间及其全部索引", use_container_width=True):
                st.session_state.confirm_destroy = True

    if st.session_state.get("confirm_destroy") and current != "default_space":
        st.sidebar.warning(f"确定要销毁空间 `{current}` 吗？此操作不可撤销！")
        dc1, dc2 = st.sidebar.columns(2)
        with dc1:
            if st.button("确认销毁", key="confirm_destroy_yes",
                         use_container_width=True, type="primary"):
                delete_project_space(get_config(), current)
                try:
                    _chat_history_path(current).unlink(missing_ok=True)
                except OSError:
                    pass
                st.session_state.current_space = "default_space"
                st.session_state.messages = _load_chat_history("default_space")
                st.session_state.confirm_destroy = False
                st.sidebar.success(f"空间 `{current}` 已销毁")
                st.rerun()
        with dc2:
            if st.button("取消", key="confirm_destroy_no", use_container_width=True):
                st.session_state.confirm_destroy = False
                st.rerun()

    st.sidebar.divider()

    space = st.session_state.current_space

    # --- 按空间启动 ---
    cfg, index, observer, handler = _bootstrap_rag(space)

    if "messages" not in st.session_state:
        st.session_state.messages = _load_chat_history(space)

    messages: list[dict] = st.session_state.messages
    _render_sidebar_dashboard(cfg, index, observer, handler, messages, space)

    # --- 聊天区域 ---
    chat_value = st.chat_input("基于本地代码与文档提问（支持多轮指代）…")
    prompt = chat_value
    if not prompt and st.session_state.get("quick_prompt_submit"):
        prompt = st.session_state.pop("quick_prompt_submit")

    if not messages and not prompt:
        _render_welcome_panel()
    elif messages:
        for m in messages:
            av = USER_AVATAR if m["role"] == "user" else ASSISTANT_AVATAR
            with st.chat_message(m["role"], avatar=av):
                st.markdown(m["content"])
                if m.get("sources"):
                    _render_sources_expander(m["sources"])

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    memory = build_memory_for_completed_turns(cfg)
    recent = st.session_state.messages[:-1][-10:]
    _sync_memory_from_ui_messages(memory, recent)

    llm, llm_fallback_note = get_llm_instance(st.session_state.selected_llm, cfg)
    if llm_fallback_note:
        st.warning(llm_fallback_note)

    chat_engine = build_chat_engine(index, llm, cfg, memory, collection_name=space)

    response_text = ""
    sources_md = ""

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            stream_resp = chat_engine.stream_chat(prompt)
            response_text = st.write_stream(stream_resp.response_gen)
            sources_md = format_sources_md(stream_resp.source_nodes)
            _render_sources_expander(sources_md)
        except Exception as e:
            error_msg = str(e)
            is_quota_error = "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg

            if is_quota_error:
                current_is_gemini = "gemini" in (st.session_state.selected_llm or "")
                fallback_model = "qwen-plus" if current_is_gemini else "gemini-2.5-flash"
                fallback_label = "千问" if current_is_gemini else "Gemini"
                st.toast(f"⚡ 主模型额度已满，正在无缝切换至 {fallback_label} 接力…", icon="🔄")
                try:
                    fallback_llm, _ = get_llm_instance(fallback_model, cfg)
                    retry_engine = build_chat_engine(index, fallback_llm, cfg, memory, collection_name=space)
                    stream_resp = retry_engine.stream_chat(prompt)
                    response_text = st.write_stream(stream_resp.response_gen)
                    sources_md = format_sources_md(stream_resp.source_nodes)
                    _render_sources_expander(sources_md)
                except Exception as fallback_err:
                    st.error(f"😭 抱歉，所有大模型均暂时无法响应。请稍后再试。（{fallback_err!s}）")
            else:
                err_msg = f"对话模型调用失败：{e!s}。请检查网络、密钥或模型可用性。"
                print(f"[第二大脑] {err_msg}")
                st.error(err_msg)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response_text,
            "sources": sources_md,
        }
    )
    _save_chat_history(space, st.session_state.messages)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    run_app()
