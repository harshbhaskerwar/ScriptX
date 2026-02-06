import streamlit as st
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import logging
import sys
import time

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bound Script Generator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv(override=True)

# Custom CSS for Premium UI, Navbar, and Tables
st.markdown("""
<style>
    /* Hide default sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* General App Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Navbar Container */
    .nav-container {
        display: flex;
        justify_content: center;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff;
        font-weight: 700;
        text-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    
    /* Gradient Text */
    .gradient-text {
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Comparison Table Styling */
    table {
        width: 100%;
        border-collapse: collapse;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        overflow: hidden;
        color: white;
    }
    th {
        background: rgba(255, 255, 255, 0.1);
        padding: 15px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    td {
        padding: 12px 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    tr:hover {
        background: rgba(255, 255, 255, 0.1);
    }
    
    /* Start Button Styling in Home */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: black;
        font-weight: bold;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 30px;
        transition: transform 0.2s;
    }
    div.stButton > button:first-child:hover {
        transform: scale(1.05);
    }

    /* Tab Buttons in Navbar */
    .nav-button {
        width: 100%;
    }

    /* ChatBot Sidebar Styling */
    .chat-sidebar {
        background: rgba(15, 15, 15, 0.8) !important;
        backdrop-filter: blur(20px);
        border-left: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        height: 100vh;
        overflow-y: auto;
    }

    .chat-message {
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 10px;
        max-width: 85%;
    }

    .user-message {
        background: rgba(0, 201, 255, 0.2);
        color: white;
        align-self: flex-end;
        margin-left: auto;
        border: 1px solid rgba(0, 201, 255, 0.3);
    }

    .ai-message {
        background: rgba(255, 255, 255, 0.05);
        color: white;
        align-self: flex-start;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .script-viewer {
        background: rgba(0, 0, 0, 0.3);
        padding: 40px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #e0e0e0;
        font-family: 'Courier New', Courier, monospace;
        line-height: 1.6;
        white-space: pre-wrap;
        box-shadow: inset 0 0 20px rgba(0,0,0,0.5);
        min-height: 600px;
    }

    .diff-added {
        background-color: rgba(40, 167, 69, 0.2) !important;
        border-left: 4px solid #28a745;
        display: block;
        width: 100%;
    }

    .diff-removed {
        background-color: rgba(220, 53, 69, 0.2) !important;
        border-left: 4px solid #dc3545;
        text-decoration: line-through;
        display: block;
        width: 100%;
    }

    .typing-animation::after {
        content: '|';
        animation: blink 1s infinite;
    }

    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0; }
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SHARED SERVICES & LOGGING
# -----------------------------------------------------------------------------
# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version=os.getenv("AZURE_LLM_API_VERSION")
)

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def stream_azure_response_generic(messages, deployment_name, max_tokens=4000, token_tracker=None):
    """Generic stream wrapper for all methods"""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            stream=True,
            temperature=0.8,
            max_tokens=max_tokens,
            stream_options={"include_usage": True}
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
            
            # Capture token usage if available (usually in the last chunk)
            if token_tracker is not None and hasattr(chunk, 'usage') and chunk.usage:
                token_tracker['prompt'] += chunk.usage.prompt_tokens
                token_tracker['completion'] += chunk.usage.completion_tokens
                token_tracker['total'] += chunk.usage.total_tokens
                
    except Exception as e:
        logger.error(f"❌ LLM Error: {str(e)}")
        st.error(f"Generation failed: {str(e)}")
        raise e

def display_execution_time(start_time, token_tracker=None):
    end_time = time.time()
    execution_time = end_time - start_time
    
    tokens_html = ""
    if token_tracker:
        # Use a single line or stripped string to avoid Markdown code block indentation issues
        tokens_html = f"""
        <div style="margin-top: 15px; font-size: 0.9rem; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px;">
            <div style="display: flex; justify-content: space-around; width: 100%;">
                <span>📥 Input: {token_tracker['prompt']}</span>
                <span>📤 Output: {token_tracker['completion']}</span>
                <span>∑  Total: {token_tracker['total']}</span>
            </div>
        </div>
        """.strip()
        
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        padding: 20px;
        border-radius: 10px;
        color: black;
        text-align: center;
        width: 100%;
        margin-top: 30px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        <div style="font-weight: bold; font-size: 1.5rem;">✅ Generation Complete!</div>
        <div style="font-size: 1rem; margin-top: 5px;">Time Taken: {execution_time:.2f} seconds</div>
        {tokens_html}
    </div>
    """, unsafe_allow_html=True)

import difflib
import re

# -----------------------------------------------------------------------------
# 2.5 EDIT & CHAT FUNCTIONALITY
# -----------------------------------------------------------------------------

def get_diff_html(old_text, new_text):
    """Generates an HTML diff between two strings"""
    if not old_text:
        return f'<div class="diff-added">{new_text}</div>'
    
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    
    diff = difflib.ndiff(old_lines, new_lines)
    html_output = []
    
    for line in diff:
        if line.startswith('+ '):
            html_output.append(f'<div class="diff-added">{line[2:]}</div>')
        elif line.startswith('- '):
            html_output.append(f'<div class="diff-removed">{line[2:]}</div>')
        elif line.startswith('  '):
            html_output.append(f'<div>{line[2:]}</div>')
            
    return "".join(html_output)

def handle_ai_interaction(prompt, current_script, deployment_name):
    """Handles both chat and script editing with high-precision instructions"""
    messages = [
        {"role": "system", "content": """You are ScriptX, a professional Screenplay Architect.
        
        TASKS:
        1. **Chat**: Answer questions and analyze the script.
        2. **Edit**: Modify the script based on EXACT user instructions.
        
        CRITICAL EDITING RULES:
        - If the user asks to change a specific value (e.g., "Change 2027 to 2026"), you MUST ensure that EVERY instance is updated. 
        - Always return the FULL script, never snippets.
        - Wrap the COMPLETE updated script inside <SCRIPT> and </SCRIPT> markers.
        - If the user's request is a question, do NOT use the markers.
        
        FORMATTING:
        - Use standard screenplay format.
        - Start with TITLE: and end with FADE OUT or THE END."""},
        {"role": "user", "content": f"CURRENT SCRIPT:\n\n{current_script}\n\nUSER REQUEST: {prompt}\n\nIf you are editing, please double-check that you applied the changes correctly before responding."}
    ]
    
    return stream_azure_response_generic(messages, deployment_name)

def render_script_editor(script_key, chat_key):
    """Renders the split-screen editor UI with robust script detection and streaming chat"""
    deployment = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
    
    diff_view_key = f"{script_key}_diff_view"
    if diff_view_key not in st.session_state:
        st.session_state[diff_view_key] = None

    col_left, col_right = st.columns([0.65, 0.35])
    
    with col_left:
        st.markdown(f"### 📄 Script Preview")
        script_area = st.empty()
        
        if st.session_state[diff_view_key]:
            display_content = st.session_state[diff_view_key]
        else:
            display_content = f'<div class="script-viewer">{st.session_state[script_key]}</div>'
        
        script_area.markdown(display_content, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state[diff_view_key]:
                if st.button("✅ Accept Changes", key=f"clear_{script_key}", use_container_width=True):
                    st.session_state[diff_view_key] = None
                    st.rerun()
        with c2:
            st.download_button(
                label="📥 Download Script",
                data=st.session_state[script_key],
                file_name=f"script_{script_key}.txt",
                mime="text/plain",
                use_container_width=True
            )

    with col_right:
        st.markdown("### 🤖 ScriptX Assistant")
        
        # Unified Chat Container
        with st.container(height=600, border=True):
            # Messages area
            for msg in st.session_state[chat_key]:
                role_class = "user-message" if msg["role"] == "user" else "ai-message"
                st.markdown(f'<div class="chat-message {role_class}">{msg["content"]}</div>', unsafe_allow_html=True)
            
            # AI Processing Placeholder (appears inside the box)
            is_processing = st.session_state[chat_key] and st.session_state[chat_key][-1]["role"] == "user"
            
            if is_processing:
                last_prompt = st.session_state[chat_key][-1]["content"]
                chat_msg_placeholder = st.empty()
                with st.spinner("ScriptX is refining..."):
                    response_gen = handle_ai_interaction(last_prompt, st.session_state[script_key], deployment)
                    
                    full_response = ""
                    for chunk in response_gen:
                        full_response += chunk
                        visible_chat = re.sub(r'<SCRIPT>.*', '\n\n*(Updating script...)*', full_response, flags=re.DOTALL | re.IGNORECASE)
                        visible_chat = re.sub(r'```.*', '\n\n*(Processing code...)*', visible_chat, flags=re.DOTALL | re.IGNORECASE)
                        chat_msg_placeholder.markdown(f'<div class="chat-message ai-message">{visible_chat}</div>', unsafe_allow_html=True)
                    
                    new_script = None
                    script_match = re.search(r'<SCRIPT>(.*?)</SCRIPT>', full_response, re.DOTALL | re.IGNORECASE)
                    if script_match:
                        new_script = script_match.group(1).strip()
                        final_chat = full_response.replace(script_match.group(0), "").strip()
                    else:
                        code_match = re.search(r'```(?:python|markdown|text)?\n(.*?)\n```', full_response, re.DOTALL | re.IGNORECASE)
                        if code_match:
                            new_script = code_match.group(1).strip()
                            final_chat = full_response.replace(code_match.group(0), "").strip()
                        else:
                            if "TITLE:" in full_response and len(full_response) > 500:
                                parts = re.split(r'TITLE:', full_response, maxsplit=1, flags=re.IGNORECASE)
                                if len(parts) > 1:
                                    final_chat = parts[0].strip()
                                    new_script = "TITLE:" + parts[1].strip()
                            else:
                                final_chat = full_response.strip()
                    
                    if new_script:
                        diff_raw = get_diff_html(st.session_state[script_key], new_script)
                        st.session_state[diff_view_key] = f'<div class="script-viewer">{diff_raw}</div>'
                        st.session_state[script_key] = new_script
                    
                    st.session_state[chat_key].append({"role": "assistant", "content": final_chat})
                    st.rerun()

            # Chat Input inside the same container
            if prompt := st.chat_input("Ask a question or request an edit...", key=f"input_{script_key}"):
                st.session_state[chat_key].append({"role": "user", "content": prompt})
                st.rerun()

# -----------------------------------------------------------------------------
# 3. PAGE LOGIC: HOME
# -----------------------------------------------------------------------------
def render_home():
    st.markdown("<h1 style='text-align: center; font-size: 4rem;'>🎬 Script<span class='gradient-text'>X</span></h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #aaa;'>The Ultimate Multi-Modal Screenplay Generation Suite</p>", unsafe_allow_html=True)
    
    st.write("") # Spacer
    
    # Centered Start Button - Moved Up
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        if st.button("🚀 START CREATING NOW", use_container_width=True):
            st.session_state.page = 'method1'
            st.rerun()

    st.divider()

    st.markdown("<h3 style='text-align: center;'>Choose Your Generation Paradigm</h3>", unsafe_allow_html=True)
    
    # Detailed Explanation Section
    st.markdown("""
    ### **1. Hierarchical + Sequential Chain (The "Blueprint" Method)**
    *Best for: Speed, Structural Rigidity, First Drafts*
    
    This approach mimics the traditional outlining process. 
    1. **Master Outline**: It first generates a rigid, page-by-page outline for the entire script.
    2. **Sequential Execution**: It then locks the structure and fills in the content scene-by-scene.
    3. **Result**: A highly structured script that rarely deviates from the plan, though it may lack some "organic" flow in dialogue between distant scenes.
    
    ### **2. Iterative Expansion (The "Organic" Method)**
    *Best for: Character-Driven Stories, Dialogue Continuity, Final Polish*
    
    This method writes like a human novelist, thinking about the immediate history.
    1. **Rolling Context**: As it writes Scene 10, it "reads" Scene 7, 8, and 9 to ensure continuity.
    2. **Dynamic Evolution**: The story can evolve more naturally as potential plot holes are smoothed over in real-time.
    3. **Result**: A smoother, more coherent narrative flow, but significantly slower and more token-intensive.
    
    ### **3. Chunk-Based Acts (The "Architect" Method)**
    *Best for: Long-Form Scripts, Complex Narratives, 3-Act Structure*
    
    This balances structure and detail by breaking the script into major movements (Acts).
    1. **Act Division**: It respects the classic 3-Act or 5-Act structure.
    2. **Chunking**: It generates content in 10-15 page chunks, ensuring the pacing fits the specific act's requirements (e.g., the "Rising Action" of Act 2 vs the "Climax" of Act 3).
    3. **Result**: Professional-grade pacing and structural integrity for full feature-length works.
    """)
    
    st.markdown("### 📊 Technical Comparison Matrix")
    st.markdown("""
    | Feature | Method 1 (Sequential) | Method 2 (Iterative) | Method 3 (Chunk-Based) |
    | :--- | :--- | :--- | :--- |
    | **Context** | Outline vs Current Scene | Last 3 Scenes (Rolling) | Previous Chunk Summary |
    | **Continuity** | Structural (High) | Micro-Detail (Excellent) | Narrative Arc (Very Good) |
    | **Generation Speed** | ⚡ Fastest | 🐢 Slowest | ⚡ Fast-Medium |
    | **Cost / Tokens** | 💲 Low | 💲💲💲 High | 💲💲 Medium |
    | **Ideal Use Case** | Fast Prototyping | Heavy Dialogue / Character | Full Feature Film |
    """)

# -----------------------------------------------------------------------------
# 4. PAGE LOGIC: METHOD 1 (Sequential)
# -----------------------------------------------------------------------------
def render_method1():
    st.title("📝 Method 1: Hierarchical + Sequential")
    st.markdown("**Overview**: Generates a master outline, then executes scene-by-scene based on that blueprint.")

    # Check if script already exists
    if st.session_state.m1_script:
        if st.button("🔄 Generate New Script", key="m1_reset"):
            st.session_state.m1_script = ""
            st.session_state.m1_chat_history = []
            st.rerun()
        
        render_script_editor("m1_script", "m1_chat_history")
        return

    user_input = st.text_area("Script Description", placeholder="Enter your story concept...", height=100, key="m1_input")
    
    deployment = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
    
    num_pages = st.number_input("Target Pages", 10, 200, 140, key="m1_pages")

    if st.button("Generate Script (Sequential)", key="m1_btn"):
        if not user_input:
            st.error("Please enter a description.")
            return

        # Start Timer
        start_time = time.time()
        token_tracker = {'prompt': 0, 'completion': 0, 'total': 0}

        # 1. Outline
        st.subheader("Phase 1: Master Outline")
        outline_ph = st.empty()
        
        def gen_outline():
            return stream_azure_response_generic([
                {"role": "system", "content": "You are an expert screenwriter."},
                {"role": "user", "content": f"Create a {num_pages}-page script outline for: {user_input}. Format as numbered list of scenes with summaries."}
            ], deployment, token_tracker=token_tracker)
            
        outline_text = ""
        with outline_ph.container():
            outline_text = st.write_stream(gen_outline())
        
        # 2. Scenes
        st.subheader("Phase 2: Scene Execution")
        scenes = [line.strip() for line in outline_text.split('\n') if line.strip() and (line[0].isdigit() or line.startswith('-'))]
        if not scenes: scenes = outline_text.split('\n\n')
        
        full_script = f"TITLE: {user_input[:50]}...\n\nOUTLINE:\n{outline_text}\n\nMODALITIES: SEQUENTIAL\n\n"
        progress = st.progress(0)
        
        for i, scene in enumerate(scenes[:min(len(scenes), num_pages // 2)]):
            st.caption(f"Writing Scene {i+1}...")
            progress.progress((i+1) / len(scenes))
            
            def gen_scene():
                return stream_azure_response_generic([
                    {"role": "system", "content": "Write a screenplay scene. Format: SCENE HEADING, ACTION, CHARACTER, DIALOGUE."},
                    {"role": "user", "content": f"Write this scene based on outline:\n{scene}\n\nContext: {user_input}"}
                ], deployment, token_tracker=token_tracker)
            
            scene_content = st.write_stream(gen_scene())
            full_script += f"\n\n{scene_content}\n\n"
            st.markdown("---")
            
        # Save to session state
        st.session_state.m1_script = full_script
        
        display_execution_time(start_time, token_tracker)
        st.rerun()

# -----------------------------------------------------------------------------
# 5. PAGE LOGIC: METHOD 2 (Iterative)
# -----------------------------------------------------------------------------
def render_method2():
    st.title("📝 Method 2: Iterative Expansion")
    st.markdown("**Overview**: Builds progressively, passing recent scene content as rolling memory.")

    # Check if script already exists
    if st.session_state.m2_script:
        if st.button("🔄 Generate New Script", key="m2_reset"):
            st.session_state.m2_script = ""
            st.session_state.m2_chat_history = []
            st.rerun()
        
        render_script_editor("m2_script", "m2_chat_history")
        return

    user_input = st.text_area("Script Description", placeholder="Enter your story concept...", height=100, key="m2_input")
    
    deployment = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
    
    col1, col2 = st.columns(2)
    with col1:
        num_pages = st.number_input("Target Pages", 10, 200, 140, key="m2_pages")
    with col2:
        context_window = st.number_input("Rolling Context (scenes)", 1, 5, 3, key="m2_window")

    if st.button("Generate Script (Iterative)", key="m2_btn"):
        if not user_input:
            st.error("Missing description.")
            return

        # Start Timer
        start_time = time.time()
        token_tracker = {'prompt': 0, 'completion': 0, 'total': 0}

        # Phase 1: Skeleton
        st.subheader("Phase 1: generating Skeleton")
        skeleton_ph = st.empty()
        
        def gen_skeleton():
            num_scenes = num_pages // 2
            return stream_azure_response_generic([
                {"role": "system", "content": "Create concise scene summaries."},
                {"role": "user", "content": f"Create {num_scenes} scene summaries for: {user_input}"}
            ], deployment, token_tracker=token_tracker)
            
        skeleton_text = ""
        with skeleton_ph.container():
            skeleton_text = st.write_stream(gen_skeleton())
            
        # Parse
        scene_summaries = [l.strip() for l in skeleton_text.split('\n') if l.strip() and (l[0].isdigit() or l.startswith('-'))]
        if not scene_summaries: scene_summaries = skeleton_text.split('\n\n')

        # Phase 2: Expansion
        st.subheader("Phase 2: Expanding Scenes")
        full_script = f"TITLE: {user_input[:50]}...\n\nSKELETON:\n{skeleton_text}\n\nMODALITIES: ITERATIVE\n\n"
        previous_scenes = []
        progress = st.progress(0)
        
        for i, summary in enumerate(scene_summaries[:num_pages//2]):
            st.caption(f"Expanding Scene {i+1} (Context: Last {min(i, context_window)} scenes)")
            progress.progress((i+1)/len(scene_summaries))
            
            # Context builder
            context = "\n".join(previous_scenes[-context_window:])
            
            def gen_expansion():
                return stream_azure_response_generic([
                    {"role": "system", "content": "Write full screenplay scene."},
                    {"role": "user", "content": f"Expand this summary:\n{summary}\n\nContext:\n{context}"}
                ], deployment, token_tracker=token_tracker)
            
            scene_text = st.write_stream(gen_expansion())
            previous_scenes.append(scene_text)
            full_script += f"\n\n{scene_text}\n\n"
            st.markdown("---")

        st.session_state.m2_script = full_script
        display_execution_time(start_time, token_tracker)
        st.rerun()

# -----------------------------------------------------------------------------
# 6. PAGE LOGIC: METHOD 3 (Chunk-Based)
# -----------------------------------------------------------------------------
def render_method3():
    st.title("📝 Method 3: Chunk-Based Acts")
    st.markdown("**Overview**: Divides into Acts, then Chunks. Best for structure.")

    # Check if script already exists
    if st.session_state.m3_script:
        if st.button("🔄 Generate New Script", key="m3_reset"):
            st.session_state.m3_script = ""
            st.session_state.m3_chat_history = []
            st.rerun()
        
        render_script_editor("m3_script", "m3_chat_history")
        return

    user_input = st.text_area("Script Description", placeholder="Story concept...", height=100, key="m3_input")
    
    deployment = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")

    c1, c2, c3 = st.columns(3)
    with c1:
        num_pages = st.number_input("Total Pages", 30, 200, 140, key="m3_pages")
    with c2:
        chunk_size = st.number_input("Chunk Size", 5, 20, 10, key="m3_chunk")
    with c3:
        act_struct = st.selectbox("Structure", ["3-Act (25/50/25)", "4-Act (25/25/25/25)", "5-Act (20/20/20/20/20)"], key="m3_struct")

    if st.button("Generate Script (Chunks)", key="m3_btn"):
        if not user_input: return

        # Start Timer
        start_time = time.time()
        token_tracker = {'prompt': 0, 'completion': 0, 'total': 0}

        # Calc pages
        if "3-Act" in act_struct: pages = [int(num_pages*0.25), int(num_pages*0.5), int(num_pages*0.25)]
        elif "4-Act" in act_struct: pages = [num_pages//4]*4
        else: pages = [num_pages//5]*5
        
        full_script = f"TITLE: {user_input[:50]}...\n\nSTRUCTURE: {act_struct}\n\nMODALITIES: CHUNK-BASED\n\n"
        
        for act_idx, act_len in enumerate(pages, 1):
            st.header(f"Act {act_idx} ({act_len} pages)")
            
            # Outline Act
            st.caption("Generating Act Outline...")
            def gen_act_out():
                return stream_azure_response_generic([
                    {"role": "system", "content": "Create act outline."},
                    {"role": "user", "content": f"Outline Act {act_idx} ({act_len} pages) for: {user_input}"}
                ], deployment, token_tracker=token_tracker)
            act_outline = st.write_stream(gen_act_out())
            
            # Chunks
            num_chunks = max(1, act_len // chunk_size)
            last_summary = ""
            
            for chunk_idx in range(num_chunks):
                st.subheader(f"Act {act_idx} - Chunk {chunk_idx+1}")
                def gen_chunk():
                    return stream_azure_response_generic([
                        {"role": "system", "content": "Write screenplay chunk."},
                        {"role": "user", "content": f"Write {chunk_size} pages for Act {act_idx}, Chunk {chunk_idx+1}.\nOutline: {act_outline}\nPrev Summary: {last_summary}"}
                    ], deployment, token_tracker=token_tracker)
                
                chunk_content = st.write_stream(gen_chunk())
                full_script += f"\n\n{chunk_content}\n\n"
                last_summary = chunk_content[-500:] # fast summary
                st.markdown("---")
                
        st.session_state.m3_script = full_script
        display_execution_time(start_time, token_tracker)
        st.rerun()

# -----------------------------------------------------------------------------
# 7. MAIN NAVIGATION ROUTER
# -----------------------------------------------------------------------------

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Initialize Script and Chat States
for i in range(1, 4):
    script_key = f"m{i}_script"
    chat_key = f"m{i}_chat_history"
    if script_key not in st.session_state:
        st.session_state[script_key] = ""
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

# Helper to change page
def set_page(p):
    st.session_state.page = p

# Render Content
if st.session_state.page == 'home':
    render_home()
else:
    # Render Navbar logic
    # We use columns for the navbar at the top
    c1, c2, c3, c4 = st.columns([1, 2, 2, 2])
    with c1:
        if st.button("🏠 Home"): set_page('home'); st.rerun()
    with c2:
        if st.button("Method 1: Sequential"): set_page('method1'); st.rerun()
    with c3:
        if st.button("Method 2: Iterative expansion"): set_page('method2'); st.rerun()
    with c4:
        if st.button("Method 3: Chunk-Based"): set_page('method3'); st.rerun()
    
    st.markdown("<hr style='margin: 0.5rem 0 2rem 0; border: 0; border-top: 1px solid rgba(255,255,255,0.1);'/>", unsafe_allow_html=True)

    if st.session_state.page == 'method1':
        render_method1()
    elif st.session_state.page == 'method2':
        render_method2()
    elif st.session_state.page == 'method3':
        render_method3()
