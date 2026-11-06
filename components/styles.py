"""
components/styles.py — Global CSS for the premium dark RAG UI.
"""

GLOBAL_CSS = """
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ──────────────────────────────────────── */
:root {
  --bg-primary:    #0e1117;
  --bg-secondary:  #161b27;
  --bg-tertiary:   #1e2433;
  --bg-card:       #1a1f2e;
  --bg-hover:      #212740;
  --border:        rgba(255,255,255,0.08);
  --border-accent: rgba(99,179,237,0.3);
  --accent:        #63b3ed;
  --accent-bright: #90cdf4;
  --accent-dim:    rgba(99,179,237,0.15);
  --text-primary:  #e8eaed;
  --text-secondary:#9aa0b0;
  --text-muted:    #6b7280;
  --success:       #68d391;
  --warning:       #f6ad55;
  --error:         #fc8181;
  --radius-sm:     6px;
  --radius-md:     10px;
  --radius-lg:     16px;
  --shadow-sm:     0 1px 3px rgba(0,0,0,0.4);
  --shadow-md:     0 4px 12px rgba(0,0,0,0.5);
  --shadow-lg:     0 8px 32px rgba(0,0,0,0.6);
}

/* ── Base ────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg-primary) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
  background: var(--bg-secondary) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1100px !important; }

/* ── Typography ──────────────────────────────────────────── */
h1, h2, h3, h4 { font-weight: 600; letter-spacing: -0.02em; }
h1 { font-size: 1.8rem; }
h2 { font-size: 1.3rem; }
code, pre {
  font-family: 'JetBrains Mono', monospace !important;
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}

/* ── Sidebar Nav Items ───────────────────────────────────── */
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 14px; margin: 2px 8px;
  border-radius: var(--radius-md);
  cursor: pointer; transition: all 0.15s ease;
  color: var(--text-secondary); font-size: 0.875rem; font-weight: 500;
  text-decoration: none; border: none; background: transparent;
}
.nav-item:hover { background: var(--bg-hover); color: var(--text-primary); }
.nav-item.active {
  background: var(--accent-dim);
  color: var(--accent-bright);
  border: 1px solid var(--border-accent);
}
.nav-icon { font-size: 1rem; width: 20px; text-align: center; }

/* ── Cards ───────────────────────────────────────────────── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  box-shadow: var(--shadow-sm);
  transition: border-color 0.2s ease;
}
.card:hover { border-color: var(--border-accent); }
.card-sm {
  background: var(--bg-tertiary);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 12px 16px;
}

/* ── Metric Cards ────────────────────────────────────────── */
.metric-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  text-align: center;
}
.metric-value { font-size: 2rem; font-weight: 700; color: var(--accent-bright); }
.metric-label { font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }

/* ── Chat bubbles ────────────────────────────────────────── */
.msg-user {
  background: var(--accent-dim);
  border: 1px solid var(--border-accent);
  border-radius: var(--radius-lg) var(--radius-lg) 4px var(--radius-lg);
  padding: 12px 16px; margin: 8px 0; max-width: 85%; margin-left: auto;
  color: var(--text-primary); font-size: 0.9rem;
}
.msg-assistant {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg) var(--radius-lg) var(--radius-lg) 4px;
  padding: 16px 20px; margin: 8px 0; max-width: 95%;
  color: var(--text-primary); font-size: 0.9rem; line-height: 1.65;
}
.msg-role {
  font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; margin-bottom: 6px; color: var(--text-muted);
}
.msg-role.user { color: var(--accent); }
.msg-role.assistant { color: var(--text-muted); }

/* ── Source citation pills ───────────────────────────────── */
.source-pill {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--bg-tertiary); border: 1px solid var(--border);
  border-radius: 20px; padding: 3px 10px;
  font-size: 0.72rem; color: var(--text-secondary);
  font-family: 'JetBrains Mono', monospace; margin: 2px;
}
.source-pill .page { color: var(--accent); }

/* ── Status badges ───────────────────────────────────────── */
.badge {
  display: inline-block; padding: 2px 10px; border-radius: 12px;
  font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
}
.badge-success  { background: rgba(104,211,145,0.15); color: var(--success); border: 1px solid rgba(104,211,145,0.3); }
.badge-warning  { background: rgba(246,173,85,0.15);  color: var(--warning); border: 1px solid rgba(246,173,85,0.3); }
.badge-error    { background: rgba(252,129,129,0.15); color: var(--error);   border: 1px solid rgba(252,129,129,0.3); }
.badge-info     { background: var(--accent-dim);      color: var(--accent);  border: 1px solid var(--border-accent); }
.badge-muted    { background: var(--bg-tertiary);     color: var(--text-muted); border: 1px solid var(--border); }

/* ── Chunk card (Explorer) ───────────────────────────────── */
.chunk-card {
  background: var(--bg-card); border: 1px solid var(--border);
  border-radius: var(--radius-md); padding: 14px 18px; margin: 6px 0;
  cursor: pointer; transition: border-color 0.15s ease;
}
.chunk-card:hover, .chunk-card.selected { border-color: var(--border-accent); background: var(--bg-hover); }
.chunk-index { font-size: 0.7rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; }
.chunk-preview { font-size: 0.82rem; color: var(--text-secondary); margin-top: 4px; line-height: 1.4; }
.chunk-meta-row { display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap; }

/* ── Score bar ───────────────────────────────────────────── */
.score-bar-wrap { background: var(--bg-tertiary); border-radius: 4px; height: 4px; width: 100%; }
.score-bar { height: 4px; border-radius: 4px; background: linear-gradient(90deg, var(--accent) 0%, var(--accent-bright) 100%); }

/* ── Input area ──────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--border-accent) !important;
  box-shadow: 0 0 0 3px var(--accent-dim) !important;
}

/* ── Buttons ─────────────────────────────────────────────── */
.stButton > button {
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  font-weight: 500 !important;
  transition: all 0.15s ease !important;
}
.stButton > button:hover {
  background: var(--bg-hover) !important;
  border-color: var(--border-accent) !important;
  color: var(--accent-bright) !important;
}
.stButton > button[kind="primary"] {
  background: var(--accent-dim) !important;
  border-color: var(--border-accent) !important;
  color: var(--accent-bright) !important;
}

/* ── Selectbox / MultiSelect ─────────────────────────────── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
  background: var(--bg-secondary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
}

/* ── Expander ────────────────────────────────────────────── */
.streamlit-expanderHeader {
  background: var(--bg-tertiary) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
}
.streamlit-expanderContent {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
}

/* ── Progress / Spinner ──────────────────────────────────── */
.stProgress > div > div > div > div {
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent-bright) 100%) !important;
}

/* ── Divider ─────────────────────────────────────────────── */
hr { border-color: var(--border) !important; margin: 16px 0 !important; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--bg-hover); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-accent); }

/* ── Welcome screen ──────────────────────────────────────── */
.welcome-hero {
  text-align: center; padding: 60px 40px;
  background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
  border: 1px solid var(--border); border-radius: var(--radius-lg);
  margin-bottom: 24px;
}
.welcome-title { font-size: 2.2rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px; }
.welcome-subtitle { font-size: 1rem; color: var(--text-secondary); max-width: 500px; margin: 0 auto 24px; }

/* ── Suggestion chip ─────────────────────────────────────── */
.suggestion-chip {
  display: inline-block; padding: 8px 16px;
  background: var(--bg-tertiary); border: 1px solid var(--border);
  border-radius: 20px; font-size: 0.8rem; color: var(--text-secondary);
  margin: 4px; cursor: pointer; transition: all 0.15s ease;
}
.suggestion-chip:hover { border-color: var(--border-accent); color: var(--accent-bright); background: var(--accent-dim); }

/* ── Sidebar branding ────────────────────────────────────── */
.brand-logo {
  display: flex; align-items: center; gap: 10px;
  padding: 18px 20px 12px; border-bottom: 1px solid var(--border); margin-bottom: 8px;
}
.brand-icon { font-size: 1.6rem; }
.brand-name { font-size: 1rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.01em; }
.brand-tagline { font-size: 0.65rem; color: var(--text-muted); letter-spacing: 0.05em; text-transform: uppercase; }

/* ── Upload area ─────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--bg-secondary) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius-lg) !important;
  transition: border-color 0.2s ease !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--border-accent) !important; }

/* ── Tab styling ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-secondary) !important;
  border-radius: var(--radius-md) !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 6px !important;
  color: var(--text-secondary) !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
}
.stTabs [aria-selected="true"] {
  background: var(--accent-dim) !important;
  color: var(--accent-bright) !important;
}
</style>
"""

