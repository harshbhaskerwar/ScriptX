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

def stream_azure_response_generic(messages, deployment_name, max_tokens=4000):
    """Generic stream wrapper for all methods"""
    try:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            stream=True,
            temperature=0.8,
            max_tokens=max_tokens
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"❌ LLM Error: {str(e)}")
        st.error(f"Generation failed: {str(e)}")
        raise e

def display_execution_time(start_time):
    end_time = time.time()
    execution_time = end_time - start_time
    st.markdown(f"""
    <div style="
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        padding: 20px;
        border-radius: 10px;
        color: black;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        margin-top: 30px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    ">
        ✅ Generation Complete!<br>
        <span style="font-size: 1rem; font-weight: normal;">Time Taken: {execution_time:.2f} seconds</span>
    </div>
    """, unsafe_allow_html=True)

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

    user_input = st.text_area("Script Description", placeholder="Enter your story concept...", height=100, key="m1_input")
    
    deployment = os.getenv("AZURE_LLM_DEPLOYMENT", "gpt-4o-mini")
    
    num_pages = st.number_input("Target Pages", 10, 200, 140, key="m1_pages")

    if st.button("Generate Script (Sequential)", key="m1_btn"):
        if not user_input:
            st.error("Please enter a description.")
            return

        # Start Timer
        start_time = time.time()

        # 1. Outline
        st.subheader("Phase 1: Master Outline")
        outline_ph = st.empty()
        
        def gen_outline():
            return stream_azure_response_generic([
                {"role": "system", "content": "You are an expert screenwriter."},
                {"role": "user", "content": f"Create a {num_pages}-page script outline for: {user_input}. Format as numbered list of scenes with summaries."}
            ], deployment)
            
        outline_text = ""
        with outline_ph.container():
            outline_text = st.write_stream(gen_outline())
        
        # 2. Scenes
        st.subheader("Phase 2: Scene Execution")
        scenes = [line.strip() for line in outline_text.split('\n') if line.strip() and (line[0].isdigit() or line.startswith('-'))]
        if not scenes: scenes = outline_text.split('\n\n')
        
        full_script = ""
        progress = st.progress(0)
        
        for i, scene in enumerate(scenes[:min(len(scenes), num_pages // 2)]):
            st.caption(f"Writing Scene {i+1}...")
            progress.progress((i+1) / len(scenes))
            
            def gen_scene():
                return stream_azure_response_generic([
                    {"role": "system", "content": "Write a screenplay scene. Format: SCENE HEADING, ACTION, CHARACTER, DIALOGUE."},
                    {"role": "user", "content": f"Write this scene based on outline:\n{scene}\n\nContext: {user_input}"}
                ], deployment)
            
            scene_content = st.write_stream(gen_scene())
            full_script += f"\n\n{scene_content}\n\n"
            st.markdown("---")
            
        display_execution_time(start_time)
        
        st.download_button("Download Script", full_script, "script_sequential.txt")

# -----------------------------------------------------------------------------
# 5. PAGE LOGIC: METHOD 2 (Iterative)
# -----------------------------------------------------------------------------
def render_method2():
    st.title("📝 Method 2: Iterative Expansion")
    st.markdown("**Overview**: Builds progressively, passing recent scene content as rolling memory.")

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

        # Phase 1: Skeleton
        st.subheader("Phase 1: generating Skeleton")
        skeleton_ph = st.empty()
        
        def gen_skeleton():
            num_scenes = num_pages // 2
            return stream_azure_response_generic([
                {"role": "system", "content": "Create concise scene summaries."},
                {"role": "user", "content": f"Create {num_scenes} scene summaries for: {user_input}"}
            ], deployment)
            
        skeleton_text = ""
        with skeleton_ph.container():
            skeleton_text = st.write_stream(gen_skeleton())
            
        # Parse
        scene_summaries = [l.strip() for l in skeleton_text.split('\n') if l.strip() and (l[0].isdigit() or l.startswith('-'))]
        if not scene_summaries: scene_summaries = skeleton_text.split('\n\n')

        # Phase 2: Expansion
        st.subheader("Phase 2: Expanding Scenes")
        full_script = ""
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
                ], deployment)
            
            scene_text = st.write_stream(gen_expansion())
            previous_scenes.append(scene_text)
            full_script += f"\n\n{scene_text}\n\n"
            st.markdown("---")

        display_execution_time(start_time)

        st.download_button("Download Script", full_script, "script_iterative.txt")

# -----------------------------------------------------------------------------
# 6. PAGE LOGIC: METHOD 3 (Chunk-Based)
# -----------------------------------------------------------------------------
def render_method3():
    st.title("📝 Method 3: Chunk-Based Acts")
    st.markdown("**Overview**: Divides into Acts, then Chunks. Best for structure.")

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

        # Calc pages
        if "3-Act" in act_struct: pages = [int(num_pages*0.25), int(num_pages*0.5), int(num_pages*0.25)]
        elif "4-Act" in act_struct: pages = [num_pages//4]*4
        else: pages = [num_pages//5]*5
        
        full_script = ""
        
        for act_idx, act_len in enumerate(pages, 1):
            st.header(f"Act {act_idx} ({act_len} pages)")
            
            # Outline Act
            st.caption("Generating Act Outline...")
            def gen_act_out():
                return stream_azure_response_generic([
                    {"role": "system", "content": "Create act outline."},
                    {"role": "user", "content": f"Outline Act {act_idx} ({act_len} pages) for: {user_input}"}
                ], deployment)
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
                    ], deployment)
                
                chunk_content = st.write_stream(gen_chunk())
                full_script += f"\n\n{chunk_content}\n\n"
                last_summary = chunk_content[-500:] # fast summary
                st.markdown("---")
                
        display_execution_time(start_time)

        st.download_button("Download Script", full_script, "script_chunks.txt")

# -----------------------------------------------------------------------------
# 7. MAIN NAVIGATION ROUTER
# -----------------------------------------------------------------------------

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'home'

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
