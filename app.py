import streamlit as st
from streamlit_mic_recorder import speech_to_text
import pandas as pd
import google.generativeai as genai
import os


# --- STEP 1: 初始化 Session State (記憶抽屜) ---
if 'voice_output' not in st.session_state:
    st.session_state['voice_output'] = ""
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = None
if 'ai_keywords_display' not in st.session_state:
    st.session_state['ai_keywords_display'] = []
if 'user_query_display' not in st.session_state:
    st.session_state['user_query_display'] = ""
if 'request_history' not in st.session_state:
    st.session_state['request_history'] = []
if 'recommendations' not in st.session_state:
    st.session_state['recommendations'] = pd.DataFrame()
if 'recommendation_title' not in st.session_state:
    st.session_state['recommendation_title'] = "猜你也喜歡"


st.set_page_config(page_title="智能點歌台", page_icon="🎵")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1080px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1 {
        font-size: 2.25rem;
        letter-spacing: 0;
    }

    h3 {
        font-size: 1.15rem;
        margin-bottom: 0.25rem;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 0.5rem;
        border-bottom: 1px solid rgba(120, 120, 120, 0.22);
    }

    [data-testid="stTabs"] [role="tab"] {
        padding: 0.75rem 1rem;
        border-radius: 0.5rem 0.5rem 0 0;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        background: rgba(255, 75, 75, 0.09);
    }

    [data-testid="stButton"] button,
    [data-testid="stLinkButton"] a {
        border-radius: 0.45rem;
        font-weight: 600;
    }

    [data-testid="stTextInput"] input {
        border-radius: 0.45rem;
    }

    [data-testid="stMetric"] {
        background: rgba(120, 120, 120, 0.08);
        border: 1px solid rgba(120, 120, 120, 0.16);
        border-radius: 0.5rem;
        padding: 0.65rem 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("智能點歌台 🎵")
st.caption("AI 語意搜尋 + 最近點歌偏好推薦 App")


# --- STEP 2: AI 配置與資料庫讀取 ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('models/gemini-3.1-flash-lite')


def get_ai_keywords(user_query):
    prompt = f"""
    你是一個強大的音樂語意分析大腦。
    請理解用戶輸入的「情境或句子」，將其轉化為 3 個與音樂相關的標籤（風格、情緒、特徵）。
    用戶輸入：'{user_query}'

    【嚴格規則】：
    1. 語意轉化：如果用戶輸入完整句子（例如「我要聽很難唱的歌」），絕對禁止把整句話當作關鍵字！你必須提取出背後的意境，例如：高音, 炫技, 高難度。
    2. 情緒轉化：如果輸入（例如「我覺得很悲傷」），請轉化為：悲傷, 抒情, 催淚。
    3. 如果包含歌手/歌名，將其保留為第一個標籤，再補上 2 個風格詞。
    4. 嚴禁多餘描述，只回傳關鍵字並用英文逗號隔開，不要有任何引號與標點符號。
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"API_ERROR: {e}"


@st.cache_data
def load_data():
    file_path = "songs_with_tags.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        return df
    return None


df = load_data()

with st.sidebar:
    st.header("最近點歌")
    if st.session_state['request_history']:
        for item in reversed(st.session_state['request_history'][-10:]):
            st.write(f"{item['song']} - {item['artist']}")

        if st.button("清除點歌紀錄", use_container_width=True):
            st.session_state['request_history'] = []
            st.session_state['recommendations'] = pd.DataFrame()
            st.session_state['recommendation_title'] = "猜你也喜歡"
            st.rerun()
    else:
        st.caption("還沒有點歌紀錄。從搜尋結果加入幾首歌後，這裡會開始累積偏好。")

    st.divider()
    st.caption("推薦會優先參考最近 10 首歌，並排除已點過的歌曲。")


# --- STEP 3: 推薦系統工具 ---
def first_existing_column(dataframe, candidates):
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).lower().strip()


def split_terms(value):
    text = normalize_text(value)
    for symbol in ["，", "、", "/", "|", ";", "；", " "]:
        text = text.replace(symbol, ",")
    return [term.strip() for term in text.split(",") if term.strip()]


def terms_are_related(left, right):
    if not left or not right:
        return False
    return left == right or left in right or right in left


def get_song_id(row):
    if 'song_id' in row and pd.notna(row['song_id']):
        return str(row['song_id'])
    return f"{row.get('artist', '')}::{row.get('song', '')}"


def infer_era(row):
    era_column = first_existing_column(pd.DataFrame([row]), ['era', '年代'])
    if era_column:
        era_value = normalize_text(row.get(era_column, ""))
        if era_value:
            return era_value

    year_column = first_existing_column(pd.DataFrame([row]), ['year', '年份', 'release_year'])
    if not year_column:
        return ""

    try:
        year = int(float(row.get(year_column)))
        return f"{year // 10 * 10}s"
    except (TypeError, ValueError):
        return ""


def get_numeric_feature(row, candidates):
    for column in candidates:
        if column in row and pd.notna(row[column]):
            try:
                return float(row[column])
            except (TypeError, ValueError):
                return None
    return None


def build_feature_profile(rows):
    profile = {
        "eras": {},
        "languages": {},
        "terms": {},
        "bpms": [],
        "energies": [],
    }

    for _, row in rows.iterrows():
        era = infer_era(row)
        if era:
            profile["eras"][era] = profile["eras"].get(era, 0) + 1

        language_column = first_existing_column(pd.DataFrame([row]), ['language', '語言'])
        if language_column:
            language = normalize_text(row.get(language_column, ""))
            if language:
                profile["languages"][language] = profile["languages"].get(language, 0) + 1

        tag_text = row.get('AI_Keywords', "")
        genre_column = first_existing_column(pd.DataFrame([row]), ['genre', 'genres', '曲風'])
        mood_column = first_existing_column(pd.DataFrame([row]), ['mood', 'moods', '情緒'])
        scene_column = first_existing_column(pd.DataFrame([row]), ['scene', 'scenes', '場景'])

        terms = split_terms(tag_text)
        if genre_column:
            terms.extend(split_terms(row.get(genre_column, "")))
        if mood_column:
            terms.extend(split_terms(row.get(mood_column, "")))
        if scene_column:
            terms.extend(split_terms(row.get(scene_column, "")))

        for term in terms:
            profile["terms"][term] = profile["terms"].get(term, 0) + 1

        bpm = get_numeric_feature(row, ['bpm', 'BPM', 'tempo'])
        if bpm:
            profile["bpms"].append(bpm)

        energy = get_numeric_feature(row, ['energy', '能量'])
        if energy is not None:
            profile["energies"].append(energy)

    return profile


def top_key(counter):
    if not counter:
        return ""
    return max(counter.items(), key=lambda item: item[1])[0]


def average(values):
    if not values:
        return None
    return sum(values) / len(values)


def create_recommendation_title(profile):
    era = top_key(profile["eras"])
    terms = [
        term
        for term, _ in sorted(profile["terms"].items(), key=lambda item: item[1], reverse=True)[:2]
    ]

    title_parts = []
    if era:
        title_parts.append(era.replace("s", "年代"))
    title_parts.extend(terms)

    if title_parts:
        return "猜你也喜歡：" + " ".join(title_parts)
    return "猜你也喜歡"


def create_recommendation_reason(row, profile):
    reasons = []
    era = infer_era(row)
    preferred_era = top_key(profile["eras"])
    if era and preferred_era and era == preferred_era:
        reasons.append(era.replace("s", "年代"))

    row_terms = set(split_terms(row.get('AI_Keywords', "")))
    matched_terms = [
        term
        for term, _ in sorted(profile["terms"].items(), key=lambda item: item[1], reverse=True)
        if term in row_terms
    ][:2]
    reasons.extend(matched_terms)

    preferred_bpm = average(profile["bpms"])
    row_bpm = get_numeric_feature(row, ['bpm', 'BPM', 'tempo'])
    if preferred_bpm and row_bpm and abs(preferred_bpm - row_bpm) <= 12:
        reasons.append("節奏接近")

    if not reasons:
        return "和最近點歌風格相近"
    return "、".join(reasons)


def recommendation_score(row, profile, requested_ids):
    song_id = get_song_id(row)
    if song_id in requested_ids:
        return -9999.0

    score = 0.0
    row_terms = set(split_terms(row.get('AI_Keywords', "")))

    genre_column = first_existing_column(pd.DataFrame([row]), ['genre', 'genres', '曲風'])
    mood_column = first_existing_column(pd.DataFrame([row]), ['mood', 'moods', '情緒'])
    scene_column = first_existing_column(pd.DataFrame([row]), ['scene', 'scenes', '場景'])
    if genre_column:
        row_terms.update(split_terms(row.get(genre_column, "")))
    if mood_column:
        row_terms.update(split_terms(row.get(mood_column, "")))
    if scene_column:
        row_terms.update(split_terms(row.get(scene_column, "")))

    song_text = normalize_text(row.get('song', ""))
    artist_text = normalize_text(row.get('artist', ""))
    for term, count in profile["terms"].items():
        has_related_term = any(terms_are_related(term, row_term) for row_term in row_terms)
        if has_related_term or term in song_text or term in artist_text:
            score += min(25, 8 * count)

    era = infer_era(row)
    preferred_era = top_key(profile["eras"])
    if era and preferred_era and era == preferred_era:
        score += 20

    language_column = first_existing_column(pd.DataFrame([row]), ['language', '語言'])
    preferred_language = top_key(profile["languages"])
    if language_column and preferred_language:
        language = normalize_text(row.get(language_column, ""))
        if language == preferred_language:
            score += 15

    preferred_bpm = average(profile["bpms"])
    row_bpm = get_numeric_feature(row, ['bpm', 'BPM', 'tempo'])
    if preferred_bpm and row_bpm:
        bpm_distance = abs(preferred_bpm - row_bpm)
        score += max(0, 12 - bpm_distance / 5)

    preferred_energy = average(profile["energies"])
    row_energy = get_numeric_feature(row, ['energy', '能量'])
    if preferred_energy is not None and row_energy is not None:
        score += max(0, 10 - abs(preferred_energy - row_energy) * 10)

    popularity = get_numeric_feature(row, ['popularity', '熱門度', 'hot_score'])
    if popularity is not None:
        score += min(10, popularity * 10 if popularity <= 1 else popularity / 10)

    return round(score, 2)


def refresh_recommendations(dataframe):
    history = st.session_state['request_history'][-10:]
    if dataframe is None or not history:
        st.session_state['recommendations'] = pd.DataFrame()
        return

    requested_ids = {item['song_id'] for item in history}
    history_rows = dataframe[
        dataframe.apply(lambda row: get_song_id(row) in requested_ids, axis=1)
    ]

    if history_rows.empty:
        st.session_state['recommendations'] = pd.DataFrame()
        return

    profile = build_feature_profile(history_rows.tail(10))
    candidate_df = dataframe.copy()
    candidate_df['recommendation_score'] = candidate_df.apply(
        lambda row: recommendation_score(row, profile, requested_ids),
        axis=1
    )
    candidate_df['recommendation_reason'] = candidate_df.apply(
        lambda row: create_recommendation_reason(row, profile),
        axis=1
    )
    recommendations = candidate_df[candidate_df['recommendation_score'] > 0]

    if recommendations.empty:
        fallback_df = candidate_df[
            ~candidate_df.apply(lambda row: get_song_id(row) in requested_ids, axis=1)
        ].copy()
        fallback_df['recommendation_score'] = 1.0
        fallback_df['recommendation_reason'] = "先推薦資料庫中尚未點過的歌曲"
        recommendations = fallback_df

    recommendations = recommendations.sort_values(by='recommendation_score', ascending=False)

    st.session_state['recommendation_title'] = create_recommendation_title(profile)
    st.session_state['recommendations'] = recommendations.head(10)


def add_to_request_history(row):
    song_id = get_song_id(row)
    st.session_state['request_history'].append({
        "song_id": song_id,
        "song": row.get('song', ''),
        "artist": row.get('artist', ''),
    })

    # 只保留最近 30 首，避免 session_state 越長越大。
    st.session_state['request_history'] = st.session_state['request_history'][-30:]
    refresh_recommendations(df)


def render_song_card(row, show_match_score=True, show_recommendation_score=False, key_prefix="song"):
    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            is_exact = show_match_score and row.get('match_score', 0) == 100.0
            st.subheader(f"🎯 {row['song']}" if is_exact else row['song'])
            st.write(f"🎤 {row['artist']}")

            display_keywords = []
            if st.session_state['user_query_display']:
                display_keywords.append(st.session_state['user_query_display'])
            for keyword in st.session_state['ai_keywords_display']:
                if keyword not in display_keywords:
                    display_keywords.append(keyword)

            matched_tags = [k for k in display_keywords if k.lower() in str(row.get('AI_Keywords', '')).lower()]
            if matched_tags:
                st.info(f"✨ 標籤命中：{', '.join(matched_tags)}")

            if show_match_score:
                st.caption(f"契合度：{int(row['match_score'])}%")
            if show_recommendation_score:
                st.caption(f"推薦分數：{row['recommendation_score']}")
                if 'recommendation_reason' in row and pd.notna(row['recommendation_reason']):
                    st.caption(f"推薦原因：{row['recommendation_reason']}")

        with c2:
            st.write(" ")
            yt_url = f"https://www.youtube.com/watch?v={row['youtube_id']}"
            st.link_button("▶️ 播放影片", yt_url, use_container_width=True)
            if st.button("➕ 加入點歌", key=f"{key_prefix}_request_{get_song_id(row)}", use_container_width=True):
                add_to_request_history(row)
                st.success("已加入點歌紀錄")
                st.rerun()


def render_recommendation_section():
    if df is None or not st.session_state['request_history']:
        st.info("先到「智能搜尋」加入幾首歌，這裡就會開始出現推薦。")
        return

    total_requests = len(st.session_state['request_history'])
    recommendation_count = len(st.session_state['recommendations'])
    metric_a, metric_b = st.columns(2)
    metric_a.metric("最近點歌", f"{total_requests} 首")
    metric_b.metric("本次推薦", f"{recommendation_count} 首")

    st.subheader(st.session_state['recommendation_title'])

    recent_labels = [
        f"{item['song']} - {item['artist']}"
        for item in st.session_state['request_history'][-5:]
    ]
    st.caption("根據最近點歌：" + "、".join(recent_labels))

    recommendations = st.session_state['recommendations']
    if not recommendations.empty:
        st.caption("推薦依據包含 AI 標籤、年代、語言、BPM、energy 與熱門度；CSV 沒有的欄位會自動略過。")
        for _, row in recommendations.iterrows():
            render_song_card(row, show_match_score=False, show_recommendation_score=True, key_prefix="recommendation")
    else:
        st.info("已記錄點歌，但目前還沒有足夠資料產生推薦。可以再加入幾首歌，或確認 CSV 的 AI_Keywords 欄位有標籤。")


def render_queue_section():
    if not st.session_state['request_history']:
        st.info("目前還沒有點歌。到「智能搜尋」或「猜你也喜歡」加入歌曲後，這裡會顯示最近點歌。")
        return

    st.subheader("已點歌單")
    queue_df = pd.DataFrame(st.session_state['request_history'])
    queue_df = queue_df.tail(30).iloc[::-1].reset_index(drop=True)
    queue_df.index = queue_df.index + 1
    st.dataframe(
        queue_df[['song', 'artist']].rename(columns={'song': '歌曲', 'artist': '歌手'}),
        use_container_width=True
    )

    if st.button("清除全部點歌紀錄", type="secondary", use_container_width=True):
        st.session_state['request_history'] = []
        st.session_state['recommendations'] = pd.DataFrame()
        st.session_state['recommendation_title'] = "猜你也喜歡"
        st.rerun()


def run_search(query):
    if df is None:
        st.error("找不到歌曲資料庫 (CSV)。")
        return

    with st.spinner('🤖 AI 正在精準媒合中...'):
        try:
            res_text = get_ai_keywords(query)

            if "API_ERROR" in res_text or not res_text or res_text.strip() == "":
                ai_keywords = []
            else:
                cleaned_res = res_text.replace("`", "").replace("'", "").replace('"', "")
                ai_keywords = [k.strip() for k in cleaned_res.split(',') if k.strip()]

            original_keyword = query.strip()
            search_keywords = []
            if original_keyword:
                search_keywords.append(original_keyword)
            for keyword in ai_keywords:
                if keyword and keyword not in search_keywords:
                    search_keywords.append(keyword)

            st.session_state['user_query_display'] = original_keyword
            st.session_state['ai_keywords_display'] = ai_keywords

            temp_df = df.copy()
            user_query_lower = query.lower().strip()

            def calculate_score(row):
                artist_val = str(row['artist']).lower().strip()
                song_val = str(row['song']).lower().strip()
                tag_val = str(row['AI_Keywords']).lower().strip()

                if user_query_lower in song_val or song_val in user_query_lower or user_query_lower in artist_val:
                    return 100.0

                if user_query_lower and user_query_lower in tag_val:
                    return 85.0

                if search_keywords:
                    tag_matches = 0
                    for k in search_keywords:
                        keyword = k.lower().strip()
                        if keyword in tag_val or keyword in song_val:
                            tag_matches += 1

                    if tag_matches == 1:
                        return 40.0
                    if tag_matches == 2:
                        return 70.0
                    if tag_matches >= 3:
                        return 90.0

                return 0.0

            temp_df['match_score'] = temp_df.apply(calculate_score, axis=1)
            temp_df = temp_df.reset_index()
            final_results = temp_df[temp_df['match_score'] > 0].sort_values(
                by=['match_score', 'index'],
                ascending=[False, True]
            )
            st.session_state['search_results'] = final_results

        except Exception as e:
            st.error(f"搜尋過程中發生錯誤：{e}")


tab_search, tab_recommend, tab_queue = st.tabs(["智能搜尋", "猜你也喜歡", "已點歌單"])


with tab_search:
    st.subheader("想唱什麼？")
    col1, col2 = st.columns([6, 4])

    with col1:
        query = st.text_input(
            "search_input",
            value=st.session_state['voice_output'],
            placeholder="搜尋歌手、歌名或心情 (例如：王菲 空靈)...",
            label_visibility="collapsed"
        )

    with col2:
        voice_text = speech_to_text(
            language='zh-TW',
            start_prompt="🎙️ 語音輸入",
            stop_prompt="🛑 輸入中...完成請點我",
            key='mic_recorder'
        )

    if st.session_state.get('mic_recorder') is not None and not voice_text:
        st.markdown("💬 :red[語音輸入中...請開始說話]")

    if voice_text and voice_text != st.session_state['voice_output']:
        st.session_state['voice_output'] = voice_text
        st.rerun()

    search_trigger = st.button("🔍 開始搜尋", type="primary", use_container_width=True)

    if search_trigger and query:
        run_search(query)

    if st.session_state['search_results'] is not None:
        display_keywords = []
        if st.session_state['user_query_display']:
            display_keywords.append(st.session_state['user_query_display'])
        for keyword in st.session_state['ai_keywords_display']:
            if keyword not in display_keywords:
                display_keywords.append(keyword)

        if display_keywords:
            formatted_keywords = ", ".join([f"`{k}`" for k in display_keywords])
            st.markdown(f"💡 **搜尋 / AI 關鍵字：** {formatted_keywords}")
            st.write("---")

        results = st.session_state['search_results']

        if not results.empty:
            st.success(f"🔍 為妳精選了 {len(results)} 首歌曲（僅顯示前 50 首）：")

            for _, row in results.head(50).iterrows():
                render_song_card(row, show_match_score=True, key_prefix="search")
        else:
            st.warning("💔 沒找到符合的歌曲，換個關鍵字試試看？")
    elif df is not None:
        st.info(f"目前資料庫共有 {len(df)} 首歌。請輸入心情或歌手開始點歌！")


with tab_recommend:
    st.subheader("依照最近點歌推薦")
    render_recommendation_section()


with tab_queue:
    render_queue_section()
