# LLM 微調權重

依本儲存庫其他週的做法，GitHub 不提交基底模型、LoRA safetensors、中間 checkpoint 與 tokenizer 複本。
原始權重雜湊、參數變化 L2、訓練步數、設定與全部 history 保留在 training.json；before.json / after.json 保留實際測試回答。
在原本的本機實驗，權重仍保留於 week6_homework/quiz3/Output/llm/adapter/，沒有刪除或變更。
GitHub 上的唯讀核對使用 ../../../verify_saved_results.py；要實際生成微調後答案，須重新訓練或放回同一組 adapter。
