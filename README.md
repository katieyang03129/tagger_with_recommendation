# 智能點歌台

這是一個使用 Streamlit 製作的智能點歌系統，支援：

- 文字搜尋歌曲
- 語音輸入搜尋
- Gemini AI 解析使用者的心情、情境與音樂需求
- 依照歌曲 `AI_Keywords` 做智能媒合
- 根據最近點歌紀錄產生「猜你也喜歡」推薦

## 檔案結構

```txt
.
├── app.py
├── songs_with_tags.csv
├── requirements.txt
├── README.md
├── .gitignore
└── versions/
```

## CSV 欄位

`songs_with_tags.csv` 至少需要以下欄位：

```csv
song,artist,AI_Keywords,youtube_id
```

推薦系統也支援以下可選欄位；有提供會讓推薦更準：

```csv
song_id,year,era,language,genre,mood,scene,bpm,energy,popularity
```

也支援部分中文欄位名稱，例如：

```csv
年份,年代,語言,曲風,情緒,場景,能量,熱門度
```

## 本機執行

安裝套件：

```bash
pip install -r requirements.txt
```

建立本機 secret：

```txt
.streamlit/secrets.toml
```

內容：

```toml
GEMINI_API_KEY = "你的 Gemini API Key"
```

啟動：

```bash
streamlit run app.py
```

## 部署到 Streamlit Cloud

1. 將此資料夾內容上傳到 GitHub。
2. 到 Streamlit Cloud 建立 App。
3. Main file path 設定為：

```txt
app.py
```

4. 在 Streamlit Cloud 的 Secrets 設定：

```toml
GEMINI_API_KEY = "你的 Gemini API Key"
```

注意：不要把 `.streamlit/secrets.toml` 上傳到 GitHub。
