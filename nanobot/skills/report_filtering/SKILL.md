---
name: report_filtering
description: 教導 AI 如何根據用戶要求，精準從特定的報告中提取資料。
metadata: '{"nanobot": {"always": true}}'
---

# 報告過濾指南 (Dynamic Report Filtering)

當用戶的問題中指定了「特定的報告」(例如：2023 Annual Report, ESG Report, 某某公司的 Q1 財報) 時，你必須執行以下步驟來模擬 `SELECT * WHERE` 操作：

## 步驟 1：確認報告存在 (Find the Report)
先使用 `search_knowledge_base` 工具，用關鍵字搜尋該報告是否存在。
例如：查詢 "Do we have the 2023 Annual Report for Tencent?"

## 步驟 2：精準限定範圍 (Strict Contextual Query)
當你確認報告存在後，再次使用 `search_knowledge_base` 工具查詢具體數據時，**必須在你的 Query 中加上嚴格的限制詞**。
- ❌ 錯誤 Query範例："What is the revenue of Tencent?" (這會混雜所有年份的數據)
- ✅ 正確 Query範例："According strictly to the 2023 Annual Report of Tencent, what is the exact revenue? Do not use data from 2022 or 2024."

## 步驟 3：防幻覺機制 (Anti-Hallucination)
如果你在指定的報告中找不到該數據，即使你在其他報告知道答案，你也**必須**回答：「在您指定的 [報告名稱] 中，沒有提及此數據。」絕不允許跨報告拼湊數據。