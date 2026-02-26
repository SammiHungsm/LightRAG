# Batch Data Analyzer (Map-Reduce)

## 任務描述
當用戶要求對比、排名、或者總結多個實體嘅數據時，採用 Map-Reduce 策略。

## 執行流程
1. **Map (分配任務)**：
   - 根據攞到嘅名單，對每一間公司/實體分別調用 `access_financial_data`。
   - 準確提取目標財務指標（例如：Revenue, Liabilities, EPS 等）。
2. **計算處理**：
   - 涉及多個數字嘅加減乘除、增長率或平均值，必須調用 `python_sandbox`，嚴禁自行心算。
3. **Reduce (匯總結果)**：
   - 將所有 Map 階段攞到嘅數據整合。
   - 使用 **Markdown 表格** 展示橫向對比。
   - 根據數據畀出專業分析結論（粵語）。