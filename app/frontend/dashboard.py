import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
from datetime import date, datetime, timedelta
import pulp

# Path magic to allow importing from src while running from app/frontend/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src import config
from src.inference.optimizer import generate_optimal_lineup
from src.models.predict import run_inference

# Logo assets — drop images here: app/frontend/assets/logos/
# Supported: .png  .jpg  .jpeg  .svg
_LOGOS_DIR = Path(__file__).resolve().parent / "assets" / "logos"

def _find_logo(name):
    for ext in ('.png', '.jpg', '.jpeg', '.svg'):
        p = _LOGOS_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None

def _show_logo(name, width, center=False):
    path = _find_logo(name)
    if path is None:
        return False
    import base64 as _b64
    mime = 'image/svg+xml' if path.suffix == '.svg' else f'image/{path.suffix.lstrip(".")}'
    b64 = _b64.b64encode(path.read_bytes()).decode()
    align = 'display:block;margin:auto;' if center else ''
    st.markdown(
        f'<img src="data:{mime};base64,{b64}" width="{width}" style="{align}"/>',
        unsafe_allow_html=True,
    )
    return True

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="WNBA Daily Fantasy Sports Roster Optimizer",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---
@st.cache_data
def load_data(filepath):
    """Safely loads CSVs with Streamlit caching for performance."""
    if Path(filepath).exists():
        return pd.read_csv(filepath)
    return pd.DataFrame()

def save_predicted_slate(dk_df, optimal_names, slate_date, platform='draftkings'):
    """
    Write all players from the current slate to slate_scores.csv.
    actual_pts is left blank to be filled in after the game.
    """
    rows = []
    for _, row in dk_df.iterrows():
        rows.append({
            'date':          slate_date,
            'platform':      platform,
            'player_name':   row['Name'],
            'position':      row['Position'],
            'salary':        row['Salary'],
            'predicted_pts': round(row['Predicted_Pts'], 2),
            'actual_pts':    None,
            'was_selected':  1 if row['Name'] in optimal_names else 0,
        })
    new_df = pd.DataFrame(rows)

    if config.SLATE_SCORES_PATH.exists():
        existing = pd.read_csv(config.SLATE_SCORES_PATH)
        # Avoid duplicate saves for the same date + platform
        dupe = (
            existing['date'].astype(str).eq(slate_date) &
            existing.get('platform', pd.Series(['draftkings'] * len(existing))).eq(platform)
        ).any()
        if dupe:
            return False, "Slate for this date and platform already saved."
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(config.SLATE_SCORES_PATH, index=False)
    return True, f"Saved {len(rows)} players for {slate_date} ({platform})."

def append_contest_log(slate_date, entry_fee, lineup_predicted, lineup_actual, cash_line, payout, notes, platform='draftkings'):
    """Append one contest summary row to contest_log.csv."""
    result = 'W' if lineup_actual >= cash_line else 'L'
    row = pd.DataFrame([{
        'date':                 slate_date,
        'platform':             platform,
        'entry_fee':            entry_fee,
        'lineup_predicted_pts': round(lineup_predicted, 2),
        'lineup_actual_pts':    round(lineup_actual, 2),
        'cash_line':            cash_line,
        'result':               result,
        'payout':               payout,
        'notes':                notes,
    }])
    if config.CONTEST_LOG_PATH.exists():
        existing = pd.read_csv(config.CONTEST_LOG_PATH)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row
    combined.to_csv(config.CONTEST_LOG_PATH, index=False)
    return result

def assign_slots(lineup_df, roster_slots):
    """Map optimizer output to ordered display slots (G, G, F, F, F, UTIL)."""
    ordered = [pos for pos, n in roster_slots.items() if pos != 'UTIL' for _ in range(n)]
    ordered += ['UTIL'] * roster_slots.get('UTIL', 0)
    result, used = [], set()
    for slot in ordered:
        for idx, row in lineup_df.iterrows():
            if idx in used:
                continue
            if slot == 'UTIL' or slot in str(row.get('Position', '')):
                entry = row.to_dict()
                entry['Slot'] = slot
                result.append(entry)
                used.add(idx)
                break
    return result


def slot_eligible_players(slate_df, slot, current_lineup, this_slot_idx):
    """Return players eligible for a slot's dropdown (position-filtered, no duplicates)."""
    occupied = {p['Name'] for i, p in enumerate(current_lineup) if i != this_slot_idx}
    eligible = slate_df[~slate_df['Name'].isin(occupied)]
    if slot != 'UTIL':
        eligible = eligible[eligible['Position'].str.contains(slot, na=False)]
    return eligible


# --- MAIN UI ---
if _find_logo('wnba'):
    _show_logo('wnba', 200, center=True)
st.markdown("<h1 style='text-align: center;'>WNBA Daily Fantasy Sports Roster Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Automated predictive pipeline and lineup optimization.</p>", unsafe_allow_html=True)

# Initialize the Tabs
tab_vault, tab_predict, tab_gamelog, tab_health = st.tabs([
    "Data Vault",
    "Daily Optimizer",
    "Game Day Log",
    "Model Health",
])


# ==========================================
# TAB 1: DAILY OPTIMIZER
# ==========================================
with tab_predict:
    st.header("Daily Optimizer")
    st.markdown("Upload a salary CSV for each platform to generate your optimal lineup.")

    today_str    = str(date.today())
    _scores_path = config.SLATE_SCORES_PATH
    if _scores_path.exists():
        _all_scores  = pd.read_csv(_scores_path)
        _today_saved = _all_scores[_all_scores['date'].astype(str) == today_str]
    else:
        _today_saved = pd.DataFrame()

    _all_platform_keys = list(config.PLATFORM_CONFIGS.keys())
    _saved_platforms   = (
        set(_today_saved['platform'].unique())
        if not _today_saved.empty and 'platform' in _today_saved.columns
        else set()
    )

    # --- Restore session state from slate_scores.csv on page reload ---
    # Runs on every script execution; guard prevents clobbering live session state.
    if not _today_saved.empty and 'platform' in _today_saved.columns:
        for _pk in _all_platform_keys:
            if f"lineup_{_pk}" in st.session_state:
                continue  # already loaded this session — don't overwrite
            _pk_rows = _today_saved[_today_saved['platform'] == _pk]
            if _pk_rows.empty:
                continue
            # Reconstruct the full slate DataFrame (all players)
            _restored_slate = _pk_rows.rename(columns={
                'player_name':   'Name',
                'position':      'Position',
                'salary':        'Salary',
                'predicted_pts': 'Predicted_Pts',
            })[['Name', 'Position', 'Salary', 'Predicted_Pts']].copy()
            _restored_slate['Injury_Status'] = ''
            st.session_state[f"slate_{_pk}"] = _restored_slate
            # Reconstruct lineup from selected players only
            _was_sel = _pk_rows.get('was_selected', pd.Series(0, index=_pk_rows.index))
            _sel_rows = _pk_rows[_was_sel == 1]
            if not _sel_rows.empty:
                _sel_df = _sel_rows.rename(columns={
                    'player_name':   'Name',
                    'position':      'Position',
                    'salary':        'Salary',
                    'predicted_pts': 'Predicted_Pts',
                }).copy()
                _sel_df['Injury_Status'] = ''
                _cfg_r = config.PLATFORM_CONFIGS[_pk]
                st.session_state[f"lineup_{_pk}"] = assign_slots(_sel_df, _cfg_r['roster_slots'])
                st.session_state[f"saved_{_pk}"] = True

    # --- Today's Slates Summary Tiles ---
    st.markdown(f"**Today's Slates** — {today_str}")
    _tile_cols = st.columns(len(_all_platform_keys))
    for _i, _pk in enumerate(_all_platform_keys):
        _display = config.PLATFORM_CONFIGS[_pk]['display_name']
        # Build logo HTML (base64) or empty string if no logo found
        _logo_path = _find_logo(_pk)
        if _logo_path:
            import base64 as _b64
            _mime = 'image/svg+xml' if _logo_path.suffix == '.svg' else f'image/{_logo_path.suffix.lstrip(".")}'
            _b64str = _b64.b64encode(_logo_path.read_bytes()).decode()
            _logo_html = f'<img src="data:{_mime};base64,{_b64str}" width="60" style="display:block;margin:0 auto 6px auto;"/>'
        else:
            _logo_html = ''
        with _tile_cols[_i]:
            if _pk in _saved_platforms:
                _n = len(_today_saved[_today_saved['platform'] == _pk])
                st.markdown(
                    f'<div style="background:#d4edda;border:1px solid #c3e6cb;border-radius:6px;padding:12px;text-align:center;">'
                    f'{_logo_html}<span style="color:#155724;">✅ {_display}<br><small>{_n} players saved</small></span></div>',
                    unsafe_allow_html=True,
                )
            elif f"lineup_{_pk}" in st.session_state:
                st.markdown(
                    f'<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;padding:12px;text-align:center;">'
                    f'{_logo_html}<span style="color:#856404;">⏳ {_display}<br><small>Lineup ready — not saved</small></span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:#cce5ff;border:1px solid #b8daff;border-radius:6px;padding:12px;text-align:center;">'
                    f'{_logo_html}<span style="color:#004085;">⬜ {_display}<br><small>Not yet uploaded</small></span></div>',
                    unsafe_allow_html=True,
                )

    # --- Compare All Platforms Panel ---
    _active_lineups = {
        pk: st.session_state[f"lineup_{pk}"]
        for pk in _all_platform_keys
        if f"lineup_{pk}" in st.session_state
    }
    if _active_lineups:
        st.divider()
        with st.expander("📊 Compare All Platforms", expanded=False):
            _cmp_cols = st.columns(len(_all_platform_keys))
            for _ci, _pk in enumerate(_all_platform_keys):
                _cfg = config.PLATFORM_CONFIGS[_pk]
                with _cmp_cols[_ci]:
                    st.markdown(f"**{_cfg['display_name']}**")
                    if _pk in _active_lineups:
                        _lu = _active_lineups[_pk]
                        _cmp_rows = [{
                            'Slot':   p.get('Slot', ''),
                            'Player': p.get('Name', ''),
                            'Salary': f"${int(p.get('Salary', 0)):,}",
                            'Proj':   f"{float(p.get('Predicted_Pts', 0)):.1f}",
                        } for p in _lu]
                        st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)
                        _t_sal  = sum(p.get('Salary', 0) for p in _lu)
                        _t_proj = sum(p.get('Predicted_Pts', 0) for p in _lu)
                        st.caption(f"${_t_sal:,} / ${_cfg['salary_cap']:,} cap  ·  {_t_proj:.1f} pts")
                    else:
                        st.caption("No lineup generated yet.")

    st.divider()

    # --- Platform Sub-Tabs ---
    _tab_labels     = [config.PLATFORM_CONFIGS[pk]['display_name'] for pk in _all_platform_keys]
    _platform_tabs  = st.tabs(_tab_labels)

    for _pk, _ptab in zip(_all_platform_keys, _platform_tabs):
        _cfg   = config.PLATFORM_CONFIGS[_pk]
        _col   = _cfg['csv_columns']
        _cap   = _cfg['salary_cap']
        _slots = _cfg['roster_slots']

        with _ptab:
            _has_slate  = f"slate_{_pk}"  in st.session_state
            _has_lineup = f"lineup_{_pk}" in st.session_state
            _is_saved   = st.session_state.get(f"saved_{_pk}", False)

            # ==== STEP 1: Upload Slate ====
            if _has_slate:
                _s1_slate  = st.session_state[f"slate_{_pk}"]
                _s1_n      = len(_s1_slate)
                _gi_col    = _col.get('game_info')
                _s1_games  = _s1_slate[_gi_col].nunique() if _gi_col and _gi_col in _s1_slate.columns else '?'
                _s1_title  = f"✅ 1. Upload Slate — {_s1_n} players, {_s1_games} games"
                _s1_open   = False
            else:
                _s1_title  = "1. Upload Slate"
                _s1_open   = True

            with st.expander(_s1_title, expanded=_s1_open):
                _uploaded = st.file_uploader(
                    f"Upload {_cfg['display_name']} Salaries (CSV)",
                    type=['csv'],
                    key=f"uploader_{_pk}",
                )

                # Fingerprint — clear stale state when a new file is dropped
                _fp = f"{_uploaded.name}_{_uploaded.size}" if _uploaded else None
                if _fp != st.session_state.get(f"fingerprint_{_pk}"):
                    for _k in [f"slate_{_pk}", f"lineup_{_pk}", f"saved_{_pk}"]:
                        st.session_state.pop(_k, None)
                    for _k in [k for k in st.session_state if k.startswith(f"edit_{_pk}_")]:
                        del st.session_state[_k]
                    st.session_state[f"fingerprint_{_pk}"] = _fp
                    if _fp is not None:
                        st.rerun()

                if _uploaded is not None:
                    _raw      = pd.read_csv(_uploaded)
                    _required = [c for c in [_col['name'], _col['salary'], _col['position']] if c]
                    _missing  = [c for c in _required if c not in _raw.columns]
                    if _missing:
                        st.error(f"Missing required columns: {_missing}")
                    else:
                        # Only store the raw slate once — Step 2 overwrites it with the
                        # inferred version (with Predicted_Pts). Never overwrite back.
                        _slate_was_new = f"slate_{_pk}" not in st.session_state
                        if _slate_was_new:
                            st.session_state[f"slate_{_pk}"] = _raw

                        # Slate stats (always computed from uploaded file for display)
                        _n_players = len(_raw)
                        _gi_col    = _col.get('game_info')
                        _n_games   = _raw[_gi_col].nunique() if _gi_col and _gi_col in _raw.columns else '?'
                        _slot_str  = ' / '.join(f"{n}{p}" for p, n in _slots.items())
                        st.markdown(f"**{_n_players} players · {_n_games} games · ${_cap:,} cap · {_slot_str}**")

                        # Injury summary
                        _inj_col   = _col.get('injury')
                        _start_col = _col.get('starting')
                        if _inj_col and _inj_col in _raw.columns:
                            _raw_inj = _raw[_inj_col].astype(str).str.strip()
                            _n_out = (_raw_inj == 'O').sum()
                            _n_d   = (_raw_inj == 'D').sum()
                            _n_q   = (_raw_inj == 'Q').sum()
                            if _n_out + _n_d + _n_q > 0:
                                st.warning(f"⚠️ {_n_out} Out · {_n_d} Doubtful · {_n_q} Questionable")
                        if _start_col and _start_col in _raw.columns:
                            _no_start = (_raw[_start_col].astype(str).str.strip().str.lower() == 'no').sum()
                            if _no_start > 0:
                                st.warning(f"⚠️ {_no_start} players listed as not starting")

                        with st.expander("🔍 Raw Slate", expanded=False):
                            _preview_cols = [c for c in [
                                _col.get('name'), _col.get('position'), _col.get('salary'),
                                _col.get('team'), _col.get('game_info'),
                                _col.get('injury'), _col.get('starting'),
                            ] if c and c in _raw.columns]
                            st.dataframe(_raw[_preview_cols], use_container_width=True, hide_index=True)

                        if _slate_was_new:
                            st.rerun()  # Update step headers now that slate is loaded

            # ==== STEP 2: Generate Lineup ====
            if _has_lineup:
                _s2_title = "✅ 2. Generate Lineup"
                _s2_open  = False
            elif _has_slate:
                _s2_title = "2. Generate Lineup"
                _s2_open  = True
            else:
                _s2_title = "2. Generate Lineup"
                _s2_open  = False

            with st.expander(_s2_title, expanded=_s2_open):
                if not _has_slate:
                    st.caption("Complete Step 1 first.")
                else:
                    _gen_col, _ = st.columns([1, 4])
                    if _gen_col.button("Generate Optimal Lineup", type="secondary",
                                       use_container_width=True, key=f"generate_{_pk}"):
                        with st.spinner("Running XGBoost inference and PuLP optimizer..."):
                            _inferred = run_inference(st.session_state[f"slate_{_pk}"], _cfg)
                            _results  = generate_optimal_lineup(_inferred, _cfg)
                            if _results["status"] == "Optimal":
                                st.session_state[f"slate_{_pk}"]  = _inferred
                                st.session_state[f"lineup_{_pk}"] = assign_slots(_results["lineup_df"], _slots)
                                st.session_state.pop(f"saved_{_pk}", None)
                                for _k in [k for k in st.session_state if k.startswith(f"edit_{_pk}_")]:
                                    del st.session_state[_k]
                                st.rerun()
                            else:
                                st.error(f"Optimizer returned: {_results['status']}")

            # ==== STEP 3: Review & Edit ====
            if _is_saved:
                _s3_title = "✅ 3. Review & Edit"
                _s3_open  = False
            elif _has_lineup:
                _s3_title = "3. Review & Edit"
                _s3_open  = True
            else:
                _s3_title = "3. Review & Edit"
                _s3_open  = False

            with st.expander(_s3_title, expanded=_s3_open):
                if not _has_lineup:
                    st.caption("Complete Step 2 first.")
                else:
                    _current_lineup = st.session_state[f"lineup_{_pk}"]
                    _slate_df       = st.session_state[f"slate_{_pk}"]

                    # Header row
                    _hdr = st.columns([0.7, 3.5, 0.7, 0.9, 1.0, 1.0, 0.8])
                    for _h, _lbl in zip(_hdr, ["**Slot**", "**Player**", "**Pos**",
                                               "**Team**", "**Salary**", "**Proj**", "**Inj**"]):
                        _h.markdown(_lbl)

                    for _i, _sp in enumerate(_current_lineup):
                        _slot         = _sp.get('Slot', '')
                        _current_name = _sp.get('Name', '')
                        _wkey         = f"edit_{_pk}_{_i}"

                        _elig_df    = slot_eligible_players(_slate_df, _slot, _current_lineup, _i)
                        _elig_names = _elig_df['Name'].tolist() if not _elig_df.empty else [_current_name]

                        # Detect selectbox change and apply swap before rendering
                        if _wkey in st.session_state and st.session_state[_wkey] != _current_name:
                            _sel = st.session_state[_wkey]
                            _new_rows = _elig_df[_elig_df['Name'] == _sel]
                            if not _new_rows.empty:
                                _new_p = _new_rows.iloc[0].to_dict()
                                _new_p['Slot'] = _slot
                                st.session_state[f"lineup_{_pk}"][_i] = _new_p
                            del st.session_state[_wkey]
                            st.rerun()

                        if _current_name not in _elig_names:
                            _elig_names = [_current_name] + _elig_names
                        _idx = _elig_names.index(_current_name)

                        _inj  = str(_sp.get('Injury_Status', '')).strip()
                        _badge = '🚫' if _inj in ('O', 'D') else ('❓' if _inj == 'Q' else '')
                        _team  = _sp.get('TeamAbbrev', _sp.get('Team', ''))

                        _row = st.columns([0.7, 3.5, 0.7, 0.9, 1.0, 1.0, 0.8])
                        _row[0].markdown(f"**{_slot}**")
                        with _row[1]:
                            st.selectbox("", options=_elig_names, index=_idx,
                                         key=_wkey, label_visibility="collapsed")
                        _row[2].markdown(_sp.get('Position', ''))
                        _row[3].markdown(str(_team))
                        _row[4].markdown(f"${int(_sp.get('Salary', 0)):,}")
                        _row[5].markdown(f"{float(_sp.get('Predicted_Pts', 0)):.1f}")
                        _row[6].markdown(_badge)

                    # Live totals
                    _t_sal  = sum(p.get('Salary', 0) for p in _current_lineup)
                    _t_proj = sum(p.get('Predicted_Pts', 0) for p in _current_lineup)
                    _over   = _t_sal - _cap
                    st.divider()
                    st.markdown(
                        f"**Total: ${_t_sal:,} / ${_cap:,} cap  ·  "
                        f"Remaining: ${_cap - _t_sal:,}  ·  Projected: {_t_proj:.1f} pts**"
                    )
                    if _over > 0:
                        st.error(f"⛔ Over cap by ${_over:,}! Fix in Step 3 before saving.")

            # ==== STEP 4: Save / Delete ====
            if _is_saved:
                _s4_title = "✅ 4. Saved"
                _s4_open  = False
            elif _has_lineup:
                _s4_title = "4. Save"
                _s4_open  = True
            else:
                _s4_title = "4. Save"
                _s4_open  = False

            with st.expander(_s4_title, expanded=_s4_open):
                if not _has_lineup:
                    st.caption("Complete Step 2 first.")
                elif _is_saved:
                    st.success(f"✅ Slate saved for {today_str} ({_cfg['display_name']}).")
                    if st.button("🗑️ Delete & Reset", key=f"delete_reset_{_pk}"):
                        if _scores_path.exists():
                            _ex  = pd.read_csv(_scores_path)
                            _ex  = _ex[~(
                                _ex['date'].astype(str).eq(today_str) &
                                _ex.get('platform', pd.Series(['draftkings'] * len(_ex))).eq(_pk)
                            )]
                            _ex.to_csv(_scores_path, index=False)
                        st.session_state.pop(f"lineup_{_pk}", None)
                        st.session_state.pop(f"saved_{_pk}",  None)
                        for _k in [k for k in st.session_state if k.startswith(f"edit_{_pk}_")]:
                            del st.session_state[_k]
                        st.cache_data.clear()
                        st.rerun()
                else:
                    _cur_lu   = st.session_state[f"lineup_{_pk}"]
                    _t_sal    = sum(p.get('Salary', 0) for p in _cur_lu)
                    _over     = _t_sal - _cap
                    _c_save, _c_del = st.columns([3, 1])
                    with _c_save:
                        if _over > 0:
                            st.caption(f"⛔ Over cap by ${_over:,} — fix in Step 3 first.")
                        if st.button("💾 Save Predicted Slate", type="primary",
                                     use_container_width=True, disabled=(_over > 0),
                                     key=f"save_btn_{_pk}"):
                            _optimal_names = {p['Name'] for p in _cur_lu}
                            # Strip any existing rows for this date+platform first
                            if _scores_path.exists():
                                _ex = pd.read_csv(_scores_path)
                                _ex = _ex[~(
                                    _ex['date'].astype(str).eq(today_str) &
                                    _ex.get('platform', pd.Series(['draftkings'] * len(_ex))).eq(_pk)
                                )]
                                _ex.to_csv(_scores_path, index=False)
                            _ok, _msg = save_predicted_slate(
                                st.session_state[f"slate_{_pk}"], _optimal_names, today_str, platform=_pk
                            )
                            if _ok:
                                st.session_state[f"saved_{_pk}"] = True
                                st.cache_data.clear()
                                st.rerun()
                            else:
                                st.warning(f"⚠️ {_msg}")
                    with _c_del:
                        if st.button("🗑️ Delete Lineup", use_container_width=True,
                                     key=f"delete_btn_{_pk}"):
                            st.session_state.pop(f"lineup_{_pk}", None)
                            for _k in [k for k in st.session_state if k.startswith(f"edit_{_pk}_")]:
                                del st.session_state[_k]
                            st.rerun()


# ==========================================
# TAB 2: DATA VAULT
# ==========================================
with tab_vault:
    st.header(f"Data Vault — {config.CURRENT_SEASON} Season")

    # --- Section 1: WNBA Game Logs ---
    st.subheader("WNBA Game Logs")
    gamelogs_df = load_data(config.RAW_DATA_DIR / f"wnba_{config.CURRENT_SEASON}_gamelogs.csv")
    if gamelogs_df.empty:
        st.warning(f"No game log file found for {config.CURRENT_SEASON}.")
    else:
        _gl_col1, _gl_col2, _gl_col3 = st.columns(3)
        _gl_col1.metric("Game Dates Logged", gamelogs_df['GAME_DATE'].nunique())
        _gl_col2.metric("Players w/ Game Appearances", gamelogs_df['PLAYER_NAME'].nunique())
        _gl_col3.metric("Most Recent Game", gamelogs_df['GAME_DATE'].max())

        _by_date = (
            gamelogs_df.groupby('GAME_DATE')
            .agg(Games=('GAME_ID', 'nunique'), Players=('PLAYER_NAME', 'nunique'))
            .reset_index()
            .rename(columns={'GAME_DATE': 'Game Date'})
            .sort_values('Game Date', ascending=False)
            .head(5)
        )
        st.dataframe(_by_date, use_container_width=True, hide_index=True)
        st.caption(
            "Data is ingested nightly via GitHub Actions (2 AM EST). "
            "New rows appear here automatically after each game day."
        )

    st.divider()

    # --- Section 2: Players ---
    st.subheader("Players")
    vault_df   = load_data(config.ROSTERS_DATA_DIR / f"player_vault_{config.CURRENT_SEASON}.csv")
    proxies_df = load_data(config.PROCESSED_DATA_DIR / "rookie_proxies.csv")

    ncaa_count = len(proxies_df[proxies_df['ORIGIN_LEAGUE'] == 'NCAA']) if not proxies_df.empty else 0
    intl_count = len(proxies_df[proxies_df['ORIGIN_LEAGUE'] == 'INTL']) if not proxies_df.empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Currently Rostered ({config.CURRENT_SEASON})", len(vault_df) if not vault_df.empty else 0)
    col2.metric("NCAA Proxies Loaded", ncaa_count)
    col3.metric("INTL Proxies Loaded", intl_count)

    st.subheader("Rookie Translation Proxies")
    col_ncaa, col_intl = st.columns(2)

    with col_ncaa:
        st.markdown(f"**NCAA Draft Targets ({config.NCAA_ROOKIE_TAX} Tax)**")
        ncaa_df = proxies_df[proxies_df['ORIGIN_LEAGUE'] == 'NCAA'] if not proxies_df.empty else pd.DataFrame()
        if not ncaa_df.empty:
            st.dataframe(ncaa_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No NCAA proxies found.")

    with col_intl:
        st.markdown(f"**International Pro Targets ({config.INTL_PRO_TAX} Tax)**")
        intl_df = proxies_df[proxies_df['ORIGIN_LEAGUE'] == 'INTL'] if not proxies_df.empty else pd.DataFrame()
        if not intl_df.empty:
            st.dataframe(intl_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No International proxies found.")

    st.divider()

    # --- Section 3: Feature Engineering (unchanged) ---
    golden_df = load_data(config.PROCESSED_DATA_DIR / "training_features.csv")
    st.subheader("Feature Engineering (Golden Table)")
    st.markdown("A live view of the processed features currently fed to the XGBoost model.")
    if not golden_df.empty:
        display_df = golden_df.sort_values(by='GAME_DATE', ascending=False).head(100)
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ==========================================
# TAB 3: GAME DAY LOG
# ==========================================
with tab_gamelog:
    st.header("Game Day Log")
    st.markdown("Track contest entries, log post-game results, and monitor daily P&L.")

    contest_df    = load_data(config.CONTEST_LOG_PATH)
    slate_scores_df = load_data(config.SLATE_SCORES_PATH)

    # --- KPI ROW (only when data exists) ---
    if not contest_df.empty:
        total_contests = len(contest_df)
        overall_wr     = (contest_df['result'] == 'W').mean() * 100
        total_pnl      = contest_df['payout'].sum() - contest_df['entry_fee'].sum()
        last10_wr      = (contest_df.tail(10)['result'] == 'W').mean() * 100

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Contests Entered", total_contests)
        k2.metric("Overall Win Rate", f"{overall_wr:.1f}%")
        k3.metric("Cumulative P&L", f"${total_pnl:+.2f}")
        k4.metric("Last 10 Win Rate", f"{last10_wr:.1f}%")
        st.divider()

    # --- INCOMPLETE SLATE BANNER ---
    if not slate_scores_df.empty:
        slate_scores_df['actual_pts'] = pd.to_numeric(slate_scores_df['actual_pts'], errors='coerce')
        incomplete = slate_scores_df[
            slate_scores_df['actual_pts'].isna() &
            (pd.to_datetime(slate_scores_df['date']).dt.date < date.today())
        ]
        # Warn per (date, platform) pair so the user knows exactly what's pending
        if 'platform' in incomplete.columns:
            incomplete_pairs = incomplete.drop_duplicates(subset=['date', 'platform'])[['date', 'platform']].values.tolist()
            incomplete_pairs = sorted(incomplete_pairs)
        else:
            incomplete_pairs = [[d, 'draftkings'] for d in sorted(incomplete['date'].unique())]

        if incomplete_pairs:
            for pending_date, pending_platform in incomplete_pairs:
                _plabel = config.PLATFORM_CONFIGS.get(pending_platform, {}).get('display_name', pending_platform)
                st.warning(f"⚠️ Unlogged results: **{pending_date}** — **{_plabel}**")

    st.divider()

    # --- TWO-COLUMN LAYOUT ---
    col_form, col_history = st.columns([1, 1], gap="large")

    # ---- LEFT: SCORE ENTRY FORM ----
    with col_form:
        st.subheader("Log Results")

        # Determine which date to pre-fill (most recent incomplete, or today)
        if not slate_scores_df.empty and incomplete_pairs:
            default_date = pd.to_datetime(incomplete_pairs[-1][0]).date()
        else:
            default_date = date.today()

        log_date     = st.date_input("Contest Date", value=default_date)
        log_date_str = str(log_date)

        # Find which platforms have saved slates for this date
        if not slate_scores_df.empty and 'platform' in slate_scores_df.columns:
            _date_platforms = slate_scores_df[
                slate_scores_df['date'].astype(str) == log_date_str
            ]['platform'].unique().tolist()
        else:
            _date_platforms = []

        if _date_platforms:
            log_platform = st.selectbox(
                "Platform",
                options=_date_platforms,
                format_func=lambda k: config.PLATFORM_CONFIGS.get(k, {}).get('display_name', k),
                key='log_platform_select',
            )
        else:
            # No saved slate yet — show a disabled selector with all platforms
            log_platform = st.selectbox(
                "Platform",
                options=list(config.PLATFORM_CONFIGS.keys()),
                format_func=lambda k: config.PLATFORM_CONFIGS[k]['display_name'],
                key='log_platform_select',
            )
        log_platform_label = config.PLATFORM_CONFIGS.get(log_platform, {}).get('display_name', log_platform)

        # Load players for selected date + platform from slate_scores
        _date_in_scores = (
            not slate_scores_df.empty
            and log_date_str in slate_scores_df['date'].astype(str).values
        )
        if _date_in_scores:
            _mask = slate_scores_df['date'].astype(str) == log_date_str
            if 'platform' in slate_scores_df.columns:
                _mask &= slate_scores_df['platform'] == log_platform
            day_players = slate_scores_df[_mask].sort_values(
                ['was_selected', 'salary'], ascending=[False, False]
            ).drop_duplicates(subset=['player_name']).reset_index(drop=True)
        else:
            day_players = pd.DataFrame()

        if not day_players.empty:
            st.markdown(f"**Enter actual FPTS — {log_platform_label} / {log_date_str}**")
            st.caption("★ = selected in optimal lineup. Fill all players for accurate model feedback.")

            actual_scores = {}
            for _, row in day_players.iterrows():
                label = f"{'★ ' if row['was_selected'] else ''}{row['player_name']} ({row['position']}, ${int(row['salary']):,}) — Predicted: {row['predicted_pts']:.1f}"
                _safe_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in row['player_name'])
                actual_scores[row['player_name']] = st.number_input(
                    label, min_value=0.0, max_value=100.0,
                    value=float(row['actual_pts']) if pd.notna(row.get('actual_pts')) else 0.0,
                    step=0.5, key=f"score_{log_platform}_{_safe_name}_{log_date_str}"
                )
        else:
            st.info(f"No saved **{log_platform_label}** slate for {log_date_str}. Run the optimizer and click **Save Predicted Slate** on game day first.")
            actual_scores = {}

        st.divider()

        cash_line  = st.number_input(f"{log_platform_label} Cash Line (FPTS)", min_value=0.0, value=150.0, step=0.5)
        entry_fee  = st.number_input("Entry Fee ($)", min_value=0.0, value=5.00, step=0.5)
        payout     = st.number_input("Payout ($)", min_value=0.0, value=0.0, step=0.5)
        notes      = st.text_area("Notes (optional)", placeholder="e.g. Missed A'ja Wilson injury scratch")

        if st.button("✅ Submit Results", type="primary", use_container_width=True) and actual_scores:
            # Update actual_pts in slate_scores.csv, scoped to date + platform
            scores_full = pd.read_csv(config.SLATE_SCORES_PATH)
            for player_name, actual in actual_scores.items():
                mask = (
                    (scores_full['date'].astype(str) == log_date_str) &
                    (scores_full['player_name'] == player_name)
                )
                if 'platform' in scores_full.columns:
                    mask &= scores_full['platform'] == log_platform
                scores_full.loc[mask, 'actual_pts'] = actual
            scores_full.to_csv(config.SLATE_SCORES_PATH, index=False)

            # Calculate lineup actual score from selected players
            selected_names = day_players.loc[day_players['was_selected'] == 1, 'player_name'].tolist()
            lineup_actual  = sum(actual_scores.get(p, 0) for p in selected_names)
            lineup_predicted = day_players[day_players['was_selected'] == 1]['predicted_pts'].sum()

            result = append_contest_log(
                log_date_str, entry_fee, lineup_predicted,
                lineup_actual, cash_line, payout, notes,
                platform=log_platform,
            )

            st.cache_data.clear()
            if result == 'W':
                st.success(f"✅ WIN logged! Lineup scored {lineup_actual:.1f} pts (cash line: {cash_line})")
            else:
                st.error(f"❌ LOSS logged. Lineup scored {lineup_actual:.1f} pts (cash line: {cash_line})")

    # ---- RIGHT: ROLLING PERFORMANCE TABLE ----
    with col_history:
        st.subheader("Contest History")

        if not contest_df.empty:
            display_log = contest_df.sort_values('date', ascending=False).copy()
            display_log['P&L'] = display_log['payout'] - display_log['entry_fee']
            display_log['P&L'] = display_log['P&L'].apply(lambda x: f"${x:+.2f}")
            rename_map = {
                'date': 'Date',
                'lineup_predicted_pts': 'Predicted',
                'lineup_actual_pts': 'Actual',
                'cash_line': 'Cash Line',
                'result': 'Result',
            }
            if 'platform' in display_log.columns:
                rename_map['platform'] = 'Platform'
            display_log = display_log.rename(columns=rename_map)
            show_cols = ['Date'] + (['Platform'] if 'Platform' in display_log.columns else []) + \
                        ['Predicted', 'Actual', 'Cash Line', 'Result', 'P&L']
            display_log = display_log[show_cols]

            st.dataframe(
                display_log,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Result': st.column_config.TextColumn('Result'),
                    'Predicted': st.column_config.NumberColumn('Predicted', format="%.1f"),
                    'Actual': st.column_config.NumberColumn('Actual', format="%.1f"),
                }
            )

            # Totals row
            wins  = (contest_df['result'] == 'W').sum()
            total = len(contest_df)
            net   = contest_df['payout'].sum() - contest_df['entry_fee'].sum()
            st.markdown(f"**{wins}W / {total - wins}L** &nbsp;|&nbsp; Net P&L: **${net:+.2f}**")
        else:
            st.info("No contests logged yet. Results will appear here after your first submission.")


# ==========================================
# TAB 4: MODEL HEALTH
# ==========================================
with tab_health:
    st.header("Model Health")

    contest_df = load_data(config.CONTEST_LOG_PATH)
    golden_df  = load_data(config.PROCESSED_DATA_DIR / "training_features.csv")

    MIN_CONTESTS_FOR_HEALTH = 10

    if contest_df.empty or len(contest_df) < MIN_CONTESTS_FOR_HEALTH:
        remaining = MIN_CONTESTS_FOR_HEALTH - (len(contest_df) if not contest_df.empty else 0)
        st.info(f"📊 Keep logging — health signals appear after {MIN_CONTESTS_FOR_HEALTH} contests. ({remaining} more to go)")
    else:
        contest_df['date'] = pd.to_datetime(contest_df['date'])
        contest_df = contest_df.sort_values('date').reset_index(drop=True)
        contest_df['win_flag'] = (contest_df['result'] == 'W').astype(int)

        # --- RETRAIN RECOMMENDATION ---
        last_retrained = pd.to_datetime(config.LAST_RETRAINED)
        days_since     = (datetime.today() - last_retrained).days
        rolling10_wr   = contest_df.tail(10)['win_flag'].mean() * 100

        needs_retrain = days_since > 28 or rolling10_wr < 70

        st.subheader("Model Status")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Last Retrained", config.LAST_RETRAINED, delta=f"{days_since}d ago", delta_color="inverse")
        col_b.metric("Last 10 Win Rate", f"{rolling10_wr:.1f}%")
        col_c.metric("Overall Win Rate", f"{contest_df['win_flag'].mean() * 100:.1f}%")

        if needs_retrain:
            reasons = []
            if days_since > 28:
                reasons.append(f"model is {days_since} days old (threshold: 28)")
            if rolling10_wr < 70:
                reasons.append(f"last-10 win rate is {rolling10_wr:.1f}% (threshold: 70%)")
            st.warning(f"⚠️ **Consider retraining.** Reason(s): {'; '.join(reasons)}")
        else:
            st.success("✅ Model health looks good. No retraining needed.")

        st.divider()

        # --- ROLLING WIN RATE CHART ---
        st.subheader("Rolling Win Rate (10-Contest Window)")
        if len(contest_df) >= 10:
            contest_df['rolling_wr'] = contest_df['win_flag'].rolling(10).mean() * 100
            chart_df = contest_df.dropna(subset=['rolling_wr'])[['date', 'rolling_wr']].set_index('date')
            st.line_chart(chart_df)
        else:
            st.info("Need 10+ contests for rolling chart.")

        st.divider()

        # --- FEATURE DRIFT (lightweight) ---
        st.subheader("Scoring Environment Drift")
        if not golden_df.empty and 'FPTS_SEASON_AVG' in golden_df.columns and 'GAME_DATE' in golden_df.columns:
            golden_df['GAME_DATE'] = pd.to_datetime(golden_df['GAME_DATE'])
            training_mean = golden_df['FPTS_SEASON_AVG'].mean()
            training_std  = golden_df['FPTS_SEASON_AVG'].std()

            cutoff = pd.Timestamp.today() - pd.Timedelta(days=14)
            recent = golden_df[golden_df['GAME_DATE'] >= cutoff]

            if not recent.empty:
                recent_mean = recent['FPTS_SEASON_AVG'].mean()
                z_score     = abs(recent_mean - training_mean) / training_std if training_std > 0 else 0

                col_d, col_e = st.columns(2)
                col_d.metric("Training Baseline Avg FPTS", f"{training_mean:.2f}")
                col_e.metric("Recent 2-Week Avg FPTS", f"{recent_mean:.2f}",
                             delta=f"{recent_mean - training_mean:+.2f}")

                if z_score > 1.5:
                    st.warning(f"⚠️ Scoring environment may have shifted (z={z_score:.1f}). "
                               "Consider checking for rule changes or roster disruptions.")
                else:
                    st.success(f"✅ Scoring distribution looks stable (z={z_score:.1f}).")
            else:
                st.info("No recent game data found for drift check.")

        st.divider()

        # --- RETRAINING WORKFLOW REMINDER ---
        with st.expander("🔧 How to Retrain the Model"):
            st.code("""
# 1. Run grid search (logs all runs to DagsHub)
python src/models/tune.py

# 2. Go to DagsHub, compare runs, download the best model.ubj

# 3. Place downloaded model in src/models/production/model.ubj
#    (or run train.py to re-train with updated XGB_PARAMS in config.py)
python src/models/train.py

# 4. Update LAST_RETRAINED in src/config.py to today's date
#    LAST_RETRAINED = 'YYYY-MM-DD'

# 5. Restart the dashboard
streamlit run app/frontend/dashboard.py
""", language="bash")
